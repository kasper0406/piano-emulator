//! A listening ladder between the Salamander recording and the engine's render
//! of the same note, one rung per hypothesis about what is missing.
//!
//! `TUNING_REPORT.md` measures the frequencies and the decays and finds them
//! right, and the instrument still sounds "sinusoidal and artificial". Two of
//! the report's findings are candidates for that — §3's excitation spectrum,
//! 5–10 dB rougher per partial than any smooth comb and *not* shared between
//! notes, and §2's ~3–4 dB of envelope residual on real unisons — and neither
//! has ever been listened to on its own. Everything here exists to make each
//! candidate audible in isolation, at matched level, on the same note.
//!
//! The ladder runs from the recording to the engine, and the rung that matters
//! is `01`: an additive resynthesis from the *measured* trajectories, every
//! tracked partial rendered as a sinusoid following its own measured `a_k(t)`
//! and `f_k(t)`. That rung is this project's synthesis machinery driven by the
//! recording's exact per-partial content, so it bounds what per-partial fitting
//! can ever deliver. If `01` already sounds artificial next to `00`, no amount
//! of better fitting will fix the instrument, and the missing sound is
//! something a per-partial model does not represent at all. If `01` sounds like
//! the piano, then everything between `01` and `07` is a *parameter* error and
//! the ladder says which parameter.
//!
//! ```text
//! cargo run --release --example timbre_ladder -- \
//!     [data/salamander] [presets/salamander-c5.toml] [data/cache/salamander] [renders/timbre-ladder]
//! ```
//!
//! # How each rung is made
//!
//! Rungs `01`–`03` and `08` come out of one additive renderer (`render_additive`)
//! that takes a list of partials, each with a frequency law and an amplitude
//! law, and integrates the phase sample by sample. Amplitude interpolation
//! between track points is done on `ln a`, not on `a`: a partial's envelope is
//! an exponential and linear interpolation of it in the linear domain is a
//! systematic 1–2 dB error between frames, which is the same size as the effect
//! under test.
//!
//! Rungs `04`, `05` and `09` come out of a second renderer (`ModalNote`) that
//! builds the engine's own `ModalBank`s directly from the preset, with the same
//! formulas `engine::string::PianoString::new` uses — partial layout, the
//! `sigma0 + sigma1 (f/1000)^2` damping law, the vertical/horizontal split, the
//! unison detune and per-string sigma scale, the bridge admittance's `Re Y`
//! per-partial correction, the unison bridge coupling — driven by the real
//! `engine::Hammer` and radiated through the real `engine::Soundboard`. Nothing
//! in the engine is modified; what varies between the three is one line each
//! (the per-partial input gain `g_k`, or a per-block retune of every mode).
//! `09` is the control that makes `04` and `05` readable: it is that renderer
//! with nothing changed, so `04 − 09` is the roughness alone and `05 − 09` is
//! the linewidth alone. It is *not* the engine (no duplex, no resonance bus, no
//! mechanism noise, no felt limiter), which is what `07` is for.
//!
//! Rungs `06` and `07` are the shipped engine through its public API, `06` with
//! the recording's attack residual mixed on top.
//!
//! # The attack residual
//!
//! `06` and `08` need "the recording minus its tracked partials" over the first
//! 150 ms, which needs a resynthesis *phase-locked* to the recording — the
//! ladder's own rung `01` has free phases and would not cancel anything. So the
//! source is analysed a second time, per channel, by projecting it onto
//! `e^{i 2 pi f_k t}` in short hopped windows (`track_complex`); the
//! interpolated complex amplitudes are resynthesized and subtracted. What
//! survives is everything at the strike that is not a tracked partial's steady
//! sinusoid: the hammer's own noise, the knock of the action, and whatever the
//! partials do inside a window that a sinusoid does not.
//!
//! The same analysis gives each partial its stereo balance — the tracker's
//! trajectories are measured on the mono sum, so without this the additive
//! rungs would be mono against a stereo recording, and a width difference is
//! exactly the kind of thing that gets heard as "artificial" and blamed on the
//! timbre.

use std::path::{Path, PathBuf};

use piano_emulator::hammer::{Hammer, MAX_SKEW_SAMPLES};
use piano_emulator::modal::ModalBank;
use piano_emulator::preset::{Preset as EnginePreset, Voicing};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::resonance::BridgeFilter;
use piano_emulator::soundboard::{pan_for_key, Soundboard};
use piano_emulator::string::{contact_taper as engine_taper, StringParams};
use piano_emulator::types::{db_to_amp, Event, BLOCK};
use piano_tuner::estimate::decay::DecayFit;
use piano_tuner::estimate::FitSpan;
use piano_tuner::numeric::{poly_eval, weighted_polyfit};
use piano_tuner::pipeline::analyze_trajectories;
use piano_tuner::preset::equal_temperament;
use piano_tuner::survey::{trajectories_for, SurveyConfig};
use piano_tuner::synth::SplitMix64;
use piano_tuner::trajectory::PartialTrack;
use piano_tuner::{audio, NoteAnalysis, Sample, SampleLibrary, SAMPLE_RATE};

/// The three keys the ladder is rendered for: the middle of the compass, where
/// `TUNING_REPORT.md` §4 says the engine is already right and the complaint is
/// therefore hardest to explain; the tenor, where a note has thirty partials
/// and three strings; and two octaves up, where a note has six partials and the
/// report's halo finding bites.
const KEYS: [u8; 3] = [60, 45, 84];

/// Velocity every rung is struck at. The source is the library layer whose
/// band contains it.
const VELOCITY: u8 = 90;

/// Silence before the strike in every file, in frames — four engine blocks, so
/// the offline modal renderer's block grid lines up with the engine's.
const PREROLL: usize = 4 * BLOCK;

const NOTE_FRAMES: usize = 4 * SAMPLE_RATE as usize;
const TOTAL_FRAMES: usize = PREROLL + NOTE_FRAMES;

const SR: f64 = SAMPLE_RATE as f64;

/// Window every rung's level is matched over, in seconds since the strike.
/// The prompt sound, past the strike transient and inside the part of the
/// record every rung actually models.
const MATCH_LO_S: f64 = 0.2;
const MATCH_HI_S: f64 = 2.0;

/// How much of the recording's attack residual is kept, and the raised-cosine
/// fade that takes it out.
const RESIDUAL_S: f64 = 0.15;
const RESIDUAL_FADE_S: f64 = 0.05;

/// How long the phase-locked analysis runs: past the residual, and past the
/// level-matching window so the per-partial stereo balance is measured over
/// the same span the levels are.
const ANALYSIS_S: f64 = 2.1;
const ANALYSIS_HOP_S: f64 = 0.005;

/// Rise time given to a partial at the strike in the additive rungs. A tracked
/// envelope is extrapolated back to `t = 0` as a step, and a step at full
/// amplitude is a click; a real partial takes about this long to come up under
/// the hammer.
const ATTACK_RAMP_S: f64 = 0.002;

/// Fades applied to every finished file, so no rung can click at either edge.
const FADE_IN_S: f64 = 0.002;
const FADE_OUT_S: f64 = 0.030;

/// Standard deviation of the slow per-partial detune walk in `05`, in cents,
/// and its correlation time in seconds. One to three cents over seconds is
/// what gives a partial a finite linewidth without becoming a vibrato.
const WALK_CENTS: f64 = 2.0;
const WALK_TAU_S: f64 = 1.5;

/// `REFERENCE_F0` of `engine::string`, which is private there: the pitch the
/// preset's `excitation_scale` was calibrated at.
const REFERENCE_F0: f32 = 261.6256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = args
        .next()
        .unwrap_or_else(|| "presets/salamander-c5.toml".into());
    let cache = PathBuf::from(args.next().unwrap_or_else(|| "data/cache/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/timbre-ladder".into()));

    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let preset = EnginePreset::load(Path::new(&preset_path))?;
    let config = SurveyConfig {
        cache_dir: Some(cache),
        ..SurveyConfig::default()
    };

    std::fs::create_dir_all(&out)?;
    let mut notes: Vec<KeyReport> = Vec::new();
    for key in KEYS {
        let report = render_key(key, &library, &preset, &config, &out)?;
        println!("{}", report.line());
        notes.push(report);
    }
    write_readme(&out, &notes, &preset_path)?;
    println!("wrote {}", out.display());
    Ok(())
}

// --------------------------------------------------------------- one key

/// What one key's ladder came to, for the README and for stdout.
struct KeyReport {
    key: u8,
    name: String,
    layer: u8,
    partials: usize,
    /// RMS of the measured excitation spectrum around the smooth comb, dB —
    /// `TUNING_REPORT.md` §3's number, recomputed here because it is the size
    /// of what rungs `02`/`04` move.
    roughness_db: f64,
    /// Whether the strike-comb fit ran, or the preset's own comb stood in.
    fitted_comb: bool,
    /// Level of the attack residual against the source, over the residual
    /// window, dB.
    residual_db: f64,
    /// First instant the tracker has a measurement at, in ms after the strike:
    /// half an analysis window, and the reason the residual is what it is.
    first_frame_ms: f64,
}

impl KeyReport {
    fn line(&self) -> String {
        format!(
            "{:>3} {:<3} layer {:>2}  {:>2} partials  roughness {:>5.2} dB  comb {}  \
             residual {:>6.1} dB  first frame {:>5.1} ms",
            self.key,
            self.name,
            self.layer,
            self.partials,
            self.roughness_db,
            if self.fitted_comb { "fitted " } else { "preset " },
            self.residual_db,
            self.first_frame_ms,
        )
    }
}

fn render_key(
    key: u8,
    library: &SampleLibrary,
    preset: &EnginePreset,
    config: &SurveyConfig,
    out: &Path,
) -> Result<KeyReport, Box<dyn std::error::Error>> {
    let name = note_name(key);
    let dir = out.join(&name);
    std::fs::create_dir_all(&dir)?;

    let sample = layer_for(library, key, VELOCITY)?;
    let note_config = config.note_config(equal_temperament(key))?;
    let trajectories = trajectories_for(sample, &note_config, config)?;
    let onset_s = trajectories.onset_s;
    let window_s = trajectories.window_s;
    let span = FitSpan::from_trajectories(&trajectories);
    let analysis = analyze_trajectories(trajectories, &note_config)?;

    // The recording, on the engine's clock, cut so that frame `PREROLL` is the
    // strike. Everything else in this function shares that clock.
    let recording = audio::load_at(&sample.path, SAMPLE_RATE)?;
    let (source_l, source_r) = cut_source(&recording, onset_s);

    let partials = build_partials(&analysis, span, preset, key);
    if partials.is_empty() {
        return Err(format!("{name}: no partial survived the decay fit").into());
    }
    let freqs: Vec<f64> = partials.iter().map(|p| p.frequency_hz).collect();

    // One phase-locked analysis of the source serves two purposes: the attack
    // residual, and the per-partial stereo balance the mono trajectories do
    // not carry.
    let window_n = analysis_window(&freqs);
    let hop_n = (ANALYSIS_HOP_S * SR).round() as usize;
    let hops = ((ANALYSIS_S * SR) as usize) / hop_n + 1;
    let locked_l = track_complex(&source_l[PREROLL..], &freqs, window_n, hop_n, hops);
    let locked_r = track_complex(&source_r[PREROLL..], &freqs, window_n, hop_n, hops);
    let balance = stereo_balance(&locked_l, &locked_r, hop_n, hops);
    let partials: Vec<Partial> = partials
        .into_iter()
        .zip(balance)
        .map(|(p, (gl, gr))| Partial { gain_l: gl, gain_r: gr, ..p })
        .collect();

    let (attack_l, attack_r) = attack_residual(
        &source_l,
        &source_r,
        &freqs,
        &locked_l,
        &locked_r,
        hop_n,
        hops,
    );

    // ---- the rungs

    let source = (source_l, source_r);
    let resynth_full = render_additive(&partials, Shape::MeasuredBoth);
    let meas_amp_law_decay = render_additive(&partials, Shape::LawDecay);
    let smooth_amp_meas_decay = render_additive(&partials, Shape::SmoothAmplitude);

    let modal_control = ModalNote::new(preset, key, None).render(preset, key, None);
    let roughness: Vec<f64> = partials.iter().map(|p| p.roughness).collect();
    let engine_rough = ModalNote::new(preset, key, Some(&roughness)).render(preset, key, None);
    let engine_linewidth = ModalNote::new(preset, key, None).render(preset, key, Some(key));

    let engine = render_engine(preset, key);

    // The residual is mixed at the level it has against the recording, so a
    // rung it is added to gets it in the same proportion the piano does.
    let source_level = match_rms(&source);
    let engine_scale = match_rms(&engine) / source_level;
    let resynth_scale = match_rms(&resynth_full) / source_level;
    let engine_attack = mixed(&engine, &attack_l, &attack_r, engine_scale);
    let resynth_plus_attack = mixed(&resynth_full, &attack_l, &attack_r, resynth_scale);

    let rungs: Vec<(&str, Stereo)> = vec![
        ("00_source", source),
        ("01_resynth_full", resynth_full),
        ("02_meas_amp_law_decay", meas_amp_law_decay),
        ("03_smooth_amp_meas_decay", smooth_amp_meas_decay),
        ("04_engine_rough", engine_rough),
        ("05_engine_linewidth", engine_linewidth),
        ("06_engine_attack", engine_attack),
        ("07_engine", engine),
        ("08_resynth_plus_attack", resynth_plus_attack),
        ("09_engine_modal_control", modal_control),
    ];
    write_matched(&dir, &rungs)?;

    let keep = PREROLL + (RESIDUAL_S * SR) as usize;
    let residual_db = 20.0
        * (rms(&attack_l, &attack_r, PREROLL, keep)
            / rms(&rungs[0].1 .0, &rungs[0].1 .1, PREROLL, keep))
        .log10();

    Ok(KeyReport {
        key,
        name,
        layer: sample.layer,
        partials: partials.len(),
        roughness_db: rms_db(partials.iter().map(|p| p.roughness)),
        fitted_comb: analysis.strike.is_some(),
        residual_db,
        first_frame_ms: 500.0 * window_s,
    })
}

/// The library layer of `key` that a strike at `velocity` would trigger.
fn layer_for(
    library: &SampleLibrary,
    key: u8,
    velocity: u8,
) -> Result<&Sample, Box<dyn std::error::Error>> {
    library
        .layers(key)
        .iter()
        .find(|s| (s.lovel..=s.hivel).contains(&velocity))
        .ok_or_else(|| format!("key {key} has no layer covering velocity {velocity}").into())
}

/// The recording, resampled if needed, cut so that frame [`PREROLL`] is the
/// strike and padded to [`TOTAL_FRAMES`].
fn cut_source(recording: &audio::Audio, onset_s: f64) -> Stereo {
    let start = (onset_s * SR).round() as isize - PREROLL as isize;
    let channel = |i: usize| -> Vec<f32> {
        let source = &recording.channels[i.min(recording.channel_count() - 1)];
        (0..TOTAL_FRAMES)
            .map(|n| {
                let index = start + n as isize;
                if index < 0 {
                    0.0
                } else {
                    source.get(index as usize).copied().unwrap_or(0.0)
                }
            })
            .collect()
    };
    (channel(0), channel(1))
}

// -------------------------------------------------------- the partial model

/// One tracked partial, with everything the additive rungs need to render it.
#[derive(Clone)]
struct Partial {
    /// Median measured frequency: the fixed pitch the smooth rungs use, and
    /// the carrier the phase-locked analysis projects onto.
    frequency_hz: f64,
    /// `(t since strike, ln a)` for every usable measurement, ascending.
    envelope: Vec<(f64, f64)>,
    /// `(t since strike, f)` for the same frames.
    glide: Vec<(f64, f64)>,
    /// The two-exponential fit: the engine's own envelope law, per partial.
    fit: DecayFit,
    /// Multiplier putting the fit onto the first measured point, so the
    /// extrapolation into the unmeasured head of the note is continuous.
    head_scale: f64,
    /// The same at the tail, for partials whose track ends before the render.
    tail_scale: f64,
    /// The measured excitation `a_k(0)` — §3's rough number.
    a0_rough: f64,
    /// What a smooth envelope times the strike comb puts at the same partial.
    a0_smooth: f64,
    /// `a0_rough / a0_smooth`: the per-partial roughness, as a linear ratio.
    roughness: f64,
    gain_l: f64,
    gain_r: f64,
}

impl Partial {
    /// The measured envelope, interpolated on `ln a` and extrapolated at both
    /// ends by the fitted law scaled to meet the data.
    fn measured_amplitude(&self, t: f64) -> f64 {
        let (t_first, ln_first) = self.envelope[0];
        let (t_last, _) = *self.envelope.last().expect("non-empty");
        if t <= t_first {
            // The head of a note is not measured: a frame is timestamped at
            // the centre of its window, so nothing before half a window exists.
            // The fit is the only statement about it there is, and it is
            // capped: a fit is allowed to extrapolate 40 dB (`DecayConfig`),
            // and 20 dB of unmeasured rise is already more than any partial of
            // a struck note does.
            let modelled = self.head_scale * self.fit.modulated_amplitude_at(t.max(0.0));
            return modelled.min(ln_first.exp() * 10.0);
        }
        if t >= t_last {
            return self.tail_scale * self.fit.modulated_amplitude_at(t);
        }
        let i = self
            .envelope
            .partition_point(|&(u, _)| u <= t)
            .clamp(1, self.envelope.len() - 1);
        let (t0, y0) = self.envelope[i - 1];
        let (t1, y1) = self.envelope[i];
        let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        (y0 * (1.0 - u) + y1 * u).exp()
    }

    /// The measured frequency, linearly interpolated and held at both ends.
    fn measured_frequency(&self, t: f64) -> f64 {
        let (t_first, f_first) = self.glide[0];
        let (t_last, f_last) = *self.glide.last().expect("non-empty");
        if t <= t_first {
            return f_first;
        }
        if t >= t_last {
            return f_last;
        }
        let i = self
            .glide
            .partition_point(|&(u, _)| u <= t)
            .clamp(1, self.glide.len() - 1);
        let (t0, f0) = self.glide[i - 1];
        let (t1, f1) = self.glide[i];
        let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
        f0 * (1.0 - u) + f1 * u
    }

    /// The engine's smooth two-polarization law, normalised to the measured
    /// excitation at the strike.
    fn law_amplitude(&self, t: f64) -> f64 {
        let at_zero = self.fit.amplitude_at(0.0);
        if at_zero <= 0.0 {
            return 0.0;
        }
        self.a0_rough * self.fit.amplitude_at(t.max(0.0)) / at_zero
    }
}

/// Assembles the partial models: the tracked envelope of each partial, its
/// fitted law, and where it stands against the smooth comb.
fn build_partials(
    analysis: &NoteAnalysis,
    span: FitSpan,
    preset: &EnginePreset,
    key: u8,
) -> Vec<Partial> {
    let onset = span.onset_s;
    let start = span.start_s;
    let smooth = smooth_comb(analysis, preset, key);

    let mut out = Vec::new();
    for fit in &analysis.decays.partials {
        let Some(track) = analysis.trajectories.track(fit.k) else {
            continue;
        };
        let (envelope, glide) = usable_points(track, onset, start);
        if envelope.len() < 2 {
            continue;
        }
        let a0_rough = fit.initial_amplitude();
        let a0_smooth = smooth(fit.k);
        if !(a0_rough.is_finite() && a0_rough > 0.0 && a0_smooth.is_finite() && a0_smooth > 0.0) {
            continue;
        }
        let (t_first, ln_first) = envelope[0];
        let (t_last, ln_last) = *envelope.last().expect("checked above");
        let head = fit.modulated_amplitude_at(t_first);
        let tail = fit.modulated_amplitude_at(t_last);
        out.push(Partial {
            frequency_hz: fit.frequency_hz,
            head_scale: if head > 0.0 { ln_first.exp() / head } else { 1.0 },
            tail_scale: if tail > 0.0 { ln_last.exp() / tail } else { 1.0 },
            envelope,
            glide,
            fit: *fit,
            a0_rough,
            a0_smooth,
            roughness: a0_rough / a0_smooth,
            gain_l: 1.0,
            gain_r: 1.0,
        });
    }
    out
}

/// The measurements of one track that an envelope may be built from: past the
/// first window that lies wholly after the strike, positive, and on the clock
/// the render uses.
type Envelope = (Vec<(f64, f64)>, Vec<(f64, f64)>);

fn usable_points(track: &PartialTrack, onset: f64, start: f64) -> Envelope {
    let mut envelope = Vec::new();
    let mut glide = Vec::new();
    for point in &track.points {
        if point.time_s < start || point.amplitude <= 0.0 || !point.frequency_hz.is_finite() {
            continue;
        }
        let t = point.time_s - onset;
        envelope.push((t, point.amplitude.ln()));
        glide.push((t, point.frequency_hz));
    }
    (envelope, glide)
}

/// What a smooth spectral envelope times the strike comb puts at partial `k`.
///
/// The fitted comb where the strike estimator ran (it needs eight partials, and
/// `TUNING_REPORT.md` §1 says C6 has six), and otherwise the comb the *preset*
/// will actually play — `sin(k pi x)` at the note's tabulated strike position
/// with its contact taper — under a degree-2 polynomial in `ln k` fitted to the
/// measured spectrum. Either way this is the smoothest thing the engine can
/// produce at this note, and the ratio to it is §3's roughness.
fn smooth_comb(
    analysis: &NoteAnalysis,
    preset: &EnginePreset,
    key: u8,
) -> Box<dyn Fn(u32) -> f64> {
    if let Some(strike) = analysis.strike.clone() {
        return Box::new(move |k| strike.amplitude(k));
    }
    let params = preset.string_params(key);
    let position = f64::from(params.strike_position);
    let width = f64::from(params.contact_width);
    let comb = move |k: u32| -> f64 {
        let kf = f64::from(k);
        let sine = (kf * std::f64::consts::PI * position).sin();
        // The same softened null the strike estimator uses: a real null is
        // never empty, and a zero here would be minus infinity in the log fit.
        let softened = (sine * sine + 0.05f64 * 0.05).sqrt();
        softened * f64::from(engine_taper(k as usize, width as f32))
    };
    let spectrum = analysis.decays.excitation_spectrum();
    let points: Vec<(f64, f64)> = spectrum
        .iter()
        .filter(|&&(k, a)| a > 0.0 && comb(k) > 0.0)
        .map(|&(k, a)| (f64::from(k).ln(), a.ln() - comb(k).ln()))
        .collect();
    let degree = 2.min(points.len().saturating_sub(1));
    let x: Vec<f64> = points.iter().map(|p| p.0).collect();
    let y: Vec<f64> = points.iter().map(|p| p.1).collect();
    let coefficients = weighted_polyfit(&x, &y, &vec![1.0; x.len()], degree)
        .unwrap_or_else(|| vec![y.iter().sum::<f64>() / y.len().max(1) as f64]);
    Box::new(move |k| poly_eval(&coefficients, f64::from(k).ln()).exp() * comb(k))
}

// ------------------------------------------------------- additive rendering

type Stereo = (Vec<f32>, Vec<f32>);

/// Which of the four combinations of amplitude and envelope a rung is.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// `01`: the measured excitation and the measured envelope — the recording's
    /// own per-partial content.
    MeasuredBoth,
    /// `02`: the measured excitation, the fitted two-exponential envelope.
    LawDecay,
    /// `03`: the smooth comb's excitation, the measured envelope.
    SmoothAmplitude,
}

/// Sums the partials into a stereo buffer, integrating each one's phase sample
/// by sample so a measured frequency glide is followed exactly.
///
/// Phases are free — the ladder is about timbre, and a free phase is what any
/// resynthesis from `(f_k, a_k)` alone has — but they are *seeded*: two runs of
/// this example produce identical files, which is what makes an A/B between two
/// versions of the ladder meaningful.
fn render_additive(partials: &[Partial], shape: Shape) -> Stereo {
    let mut left = vec![0.0f32; TOTAL_FRAMES];
    let mut right = vec![0.0f32; TOTAL_FRAMES];
    let mut rng = SplitMix64::new(0x71_6d_62_72_5f_6c_64_00);
    for partial in partials {
        let mut phase = rng.next_f64() * std::f64::consts::TAU;
        let scale = match shape {
            Shape::SmoothAmplitude => partial.a0_smooth / partial.a0_rough,
            _ => 1.0,
        };
        for n in PREROLL..TOTAL_FRAMES {
            let t = (n - PREROLL) as f64 / SR;
            let (amplitude, frequency) = match shape {
                Shape::LawDecay => (partial.law_amplitude(t), partial.frequency_hz),
                _ => (
                    scale * partial.measured_amplitude(t),
                    partial.measured_frequency(t),
                ),
            };
            let value = amplitude * attack_ramp(t) * phase.sin();
            left[n] += (value * partial.gain_l) as f32;
            right[n] += (value * partial.gain_r) as f32;
            phase += std::f64::consts::TAU * frequency / SR;
        }
    }
    (left, right)
}

/// The rise the additive rungs give a partial at the strike. Raised cosine, so
/// the waveform and its slope both start at zero and nothing clicks.
fn attack_ramp(t: f64) -> f64 {
    if t >= ATTACK_RAMP_S {
        1.0
    } else if t <= 0.0 {
        0.0
    } else {
        0.5 - 0.5 * (std::f64::consts::PI * t / ATTACK_RAMP_S).cos()
    }
}

// ------------------------------------------------- phase-locked analysis

/// The phase-locked projection and its resynthesis, both from
/// [`piano_tuner::residual`]: they were written here and are now the estimator's
/// as well, so the residual the ladder listens to and the one
/// `estimate::attack` fits `[noise.strike]` from are the same subtraction.
fn track_complex(
    signal: &[f32],
    freqs: &[f64],
    window_n: usize,
    hop_n: usize,
    hops: usize,
) -> Vec<Vec<(f64, f64)>> {
    piano_tuner::residual::track_complex(signal, SR, freqs, window_n, hop_n, hops)
}

fn resynth_locked(
    freqs: &[f64],
    coefficients: &[Vec<(f64, f64)>],
    hop_n: usize,
    hops: usize,
    frames: usize,
) -> Vec<f64> {
    piano_tuner::residual::resynth_locked(freqs, SR, coefficients, hop_n, hops, frames)
}

/// Per-partial stereo balance, as a pair of gains whose mean is 1.
///
/// The trajectories are measured on the mono sum, so a partial rendered at its
/// tracked amplitude with these two gains has the recording's balance and the
/// mono sum the tracker measured. Measured over the level-matching window
/// rather than at the strike: `TUNING_REPORT.md` §5 says the balance *drifts*
/// by 1–6 dB over that span, and one static number per partial is all a
/// per-partial additive model can carry.
fn stereo_balance(
    left: &[Vec<(f64, f64)>],
    right: &[Vec<(f64, f64)>],
    hop_n: usize,
    hops: usize,
) -> Vec<(f64, f64)> {
    let lo = ((MATCH_LO_S * SR) as usize / hop_n).min(hops.saturating_sub(1));
    let hi = (((MATCH_HI_S * SR) as usize / hop_n) + 1).min(hops);
    left.iter()
        .zip(right)
        .map(|(l, r)| {
            let energy = |hopped: &Vec<(f64, f64)>| -> f64 {
                hopped[lo..hi]
                    .iter()
                    .map(|&(re, im)| re * re + im * im)
                    .sum::<f64>()
                    .sqrt()
            };
            let (el, er) = (energy(l), energy(r));
            if !(el > 0.0 && er > 0.0) {
                return (1.0, 1.0);
            }
            let ratio = el / er;
            (2.0 * ratio / (1.0 + ratio), 2.0 / (1.0 + ratio))
        })
        .collect()
}

/// The recording minus its tracked partials over the attack, faded out.
///
/// Returned on the render's clock (frame [`PREROLL`] is the strike) and zero
/// everywhere past the fade, so it can simply be added to any rung.
fn attack_residual(
    left: &[f32],
    right: &[f32],
    freqs: &[f64],
    locked_l: &[Vec<(f64, f64)>],
    locked_r: &[Vec<(f64, f64)>],
    hop_n: usize,
    hops: usize,
) -> (Vec<f32>, Vec<f32>) {
    let end = ((RESIDUAL_S + RESIDUAL_FADE_S) * SR) as usize;
    let keep = (RESIDUAL_S * SR) as usize;
    let mut out = (vec![0.0f32; TOTAL_FRAMES], vec![0.0f32; TOTAL_FRAMES]);
    for (channel, (source, locked)) in [(left, locked_l), (right, locked_r)].into_iter().enumerate() {
        let modelled = resynth_locked(freqs, locked, hop_n, hops, end);
        let target = if channel == 0 { &mut out.0 } else { &mut out.1 };
        for n in 0..end {
            let residual = f64::from(source[PREROLL + n]) - modelled[n];
            let fade = if n <= keep {
                1.0
            } else {
                let u = (n - keep) as f64 / (end - keep) as f64;
                0.5 + 0.5 * (std::f64::consts::PI * u).cos()
            };
            target[PREROLL + n] = (residual * fade) as f32;
        }
    }
    out
}

/// A window long enough to separate the fundamental from its neighbour and
/// short enough to see the attack: four periods of the lowest partial, held
/// between 20 and 40 ms.
fn analysis_window(freqs: &[f64]) -> usize {
    piano_tuner::residual::locked_window(freqs, SR)
}

// ------------------------------------------------- the engine, offline

/// The engine's own modal machinery for one note, assembled here from the
/// preset instead of by `PianoString::new`, so that a rung may change one thing
/// about it without changing the engine.
///
/// Every formula is the engine's, copied term for term from
/// `engine::string::PianoString::new`: the partial layout, the damping law and
/// its vertical/horizontal split, the bridge admittance's per-partial `Re Y`
/// correction, the unison detune and per-string sigma scale, the strike comb
/// with its contact taper, and the one-block-late bridge coupling inside the
/// group. What is *not* here is everything outside the string — the duplex
/// segments, the sympathetic bus, the mechanism noise, the felt limiter — which
/// is what makes rung `09` the control for `04` and `05` rather than a second
/// copy of `07`.
struct ModalNote {
    strings: Vec<ModalStrings>,
    group_previous: [f32; BLOCK],
    coupling: f32,
    shares: Vec<f32>,
    /// Every mode as it was built, per string, so the linewidth rung can
    /// retune around it without re-deriving anything.
    base: Vec<Vec<Mode>>,
    partials: usize,
}

/// One mode of one unison string, in both polarizations.
#[derive(Clone, Copy)]
struct Mode {
    freq_v: f32,
    sigma_v: f32,
    gain_v: f32,
    freq_h: f32,
    sigma_h: f32,
    gain_h: f32,
}

struct ModalStrings {
    vertical: ModalBank,
    horizontal: ModalBank,
    excitation: [f32; BLOCK],
    previous: [f32; BLOCK],
}

impl ModalNote {
    /// `roughness`, when given, multiplies the input gain of partial `k` — the
    /// per-partial excitation gain replaced by the measured `a_k(0)` scatter.
    /// Partials the recording never measured keep the engine's own gain.
    fn new(preset: &EnginePreset, key: u8, roughness: Option<&[f64]>) -> Self {
        let params = preset.string_params(key);
        let voicing = &preset.voicing;
        let partials = params.partial_count();
        let radiated = radiated_damping(&params, voicing, partials);
        let output_scale =
            voicing.excitation_scale * params.bridge_gain * params.f0 / REFERENCE_F0;
        let vertical_factor = voicing.vertical_decay_factor();
        let horizontal_gain = db_to_amp(voicing.horizontal_gain_db);

        let mut strings = Vec::with_capacity(params.unison);
        let mut base = Vec::with_capacity(params.unison);
        for (i, &polarization_offset) in voicing
            .horizontal_offset_hz
            .iter()
            .take(params.unison)
            .enumerate()
        {
            let detune = voicing.detune_ratio(i, params.unison, params.detune_cents);
            let sigma_scale = voicing.sigma_scale(i, params.unison);
            let mut vertical = ModalBank::with_capacity(partials);
            let mut horizontal = ModalBank::with_capacity(partials);
            let mut modes = Vec::with_capacity(partials);
            for k in 1..=partials {
                let f = params.partial_freq(k) * detune;
                let sigma =
                    params.partial_sigma(k) * vertical_factor * sigma_scale * radiated[k - 1];
                let rough = roughness
                    .and_then(|r| r.get(k - 1))
                    .copied()
                    .unwrap_or(1.0) as f32;
                let g = output_scale
                    * (k as f32 * std::f32::consts::PI * params.strike_position).sin()
                    * engine_taper(k, params.contact_width)
                    * rough
                    / SAMPLE_RATE as f32;
                let mode = Mode {
                    freq_v: f,
                    sigma_v: sigma,
                    gain_v: g,
                    freq_h: f + polarization_offset,
                    sigma_h: sigma * voicing.horizontal_decay_ratio,
                    gain_h: g * horizontal_gain,
                };
                vertical.push_mode(mode.freq_v, mode.sigma_v, mode.gain_v);
                horizontal.push_mode(mode.freq_h, mode.sigma_h, mode.gain_h);
                modes.push(mode);
            }
            strings.push(ModalStrings {
                vertical,
                horizontal,
                excitation: [0.0; BLOCK],
                previous: [0.0; BLOCK],
            });
            base.push(modes);
        }
        ModalNote {
            group_previous: [0.0; BLOCK],
            coupling: voicing.unison_coupling / output_scale,
            shares: (0..params.unison)
                .map(|i| voicing.strike_share(i, params.unison))
                .collect(),
            strings,
            base,
            partials,
        }
    }

    /// Strikes the note and radiates it through the engine's soundboard.
    ///
    /// `walk_seed`, when given, gives every mode of every string an independent
    /// Ornstein-Uhlenbeck detune of [`WALK_CENTS`] cents with a [`WALK_TAU_S`]
    /// correlation time, updated once per block. A single pole at a fixed
    /// frequency radiates a line of zero width; a string on a real board does
    /// not, and this is the cheapest thing that gives one a width without
    /// giving it a vibrato.
    fn render(mut self, preset: &EnginePreset, key: u8, walk_seed: Option<u8>) -> Stereo {
        let mut hammer = Hammer::new(preset.hammer_params(key));
        let mut board = Soundboard::new(&preset.soundboard);
        let pan = pan_for_key(key);
        let spread = preset.pan_spread(key) * if key % 2 == 0 { 1.0 } else { -1.0 };
        let (pan_v, pan_h) = (pan - spread, pan + spread);

        let mut walk = walk_seed.map(|seed| Walk::new(u64::from(seed), self.walk_len()));
        let mut left = vec![0.0f32; TOTAL_FRAMES];
        let mut right = vec![0.0f32; TOTAL_FRAMES];
        let mut vertical = [0.0f32; BLOCK];
        let mut horizontal = [0.0f32; BLOCK];
        let strings = self.strings.len();

        let mut start = 0usize;
        while start < TOTAL_FRAMES {
            let end = (start + BLOCK).min(TOTAL_FRAMES);
            if start == PREROLL {
                hammer.strike_midi(VELOCITY);
            }
            if let Some(walk) = walk.as_mut() {
                walk.advance();
            }
            if let Some(walk) = walk.as_ref() {
                self.retune(walk);
            }
            board.begin_block();
            if hammer.is_active() {
                for s in 0..strings {
                    let skew = s * MAX_SKEW_SAMPLES / strings.max(1);
                    let share = self.shares[s];
                    hammer.add_pulse(&mut self.strings[s].excitation, skew, share);
                }
                hammer.advance(BLOCK);
            }
            vertical.fill(0.0);
            horizontal.fill(0.0);
            self.process(&mut vertical, &mut horizontal);
            board.add_voice(&vertical, pan_v);
            board.add_voice(&horizontal, pan_h);

            let mut block_l = [0.0f32; BLOCK];
            let mut block_r = [0.0f32; BLOCK];
            board.process(&mut block_l, &mut block_r);
            left[start..end].copy_from_slice(&block_l[..end - start]);
            right[start..end].copy_from_slice(&block_r[..end - start]);
            start = end;
        }
        (left, right)
    }

    fn walk_len(&self) -> usize {
        2 * self.strings.len() * self.partials
    }

    /// Applies the current walk offsets to every mode, keeping each resonator's
    /// state — `ModalBank::set_mode` redefines a mode in place, which is how a
    /// retuned string goes on ringing instead of restarting.
    fn retune(&mut self, walk: &Walk) {
        let mut index = 0;
        for (s, string) in self.strings.iter_mut().enumerate() {
            for k in 0..self.partials {
                let mode = self.base[s][k];
                let (ratio_v, ratio_h) = (walk.ratio(index), walk.ratio(index + 1));
                index += 2;
                string
                    .vertical
                    .set_mode(k, mode.freq_v * ratio_v, mode.sigma_v, mode.gain_v);
                string
                    .horizontal
                    .set_mode(k, mode.freq_h * ratio_h, mode.sigma_h, mode.gain_h);
            }
        }
    }

    /// The unison bridge coupling and the two polarizations, exactly as
    /// `PianoString::process_split` runs them.
    fn process(&mut self, out_v: &mut [f32], out_h: &mut [f32]) {
        if self.strings.len() == 1 {
            let s = &mut self.strings[0];
            s.vertical.process_add(&s.excitation, out_v);
            s.horizontal.process_add(&s.excitation, out_h);
            s.excitation.fill(0.0);
            return;
        }
        for s in &mut self.strings {
            for ((e, &sum), &own) in s
                .excitation
                .iter_mut()
                .zip(&self.group_previous)
                .zip(&s.previous)
            {
                *e += self.coupling * (sum - own);
            }
        }
        self.group_previous.fill(0.0);
        let mut vertical = [0.0f32; BLOCK];
        for s in &mut self.strings {
            vertical.fill(0.0);
            s.previous.fill(0.0);
            s.vertical.process_add(&s.excitation, &mut vertical);
            s.horizontal.process_add(&s.excitation, &mut s.previous);
            s.excitation.fill(0.0);
            for i in 0..BLOCK {
                out_v[i] += vertical[i];
                out_h[i] += s.previous[i];
                s.previous[i] += vertical[i];
                self.group_previous[i] += s.previous[i];
            }
        }
    }
}

/// `engine::string::radiated_damping`, which is private there: the per-partial
/// multiplier the bridge admittance's *fluctuation* puts on a partial's decay.
fn radiated_damping(params: &StringParams, voicing: &Voicing, partials: usize) -> Vec<f32> {
    let share = match &voicing.bridge {
        Some(bridge) if bridge.radiated_share > 0.0 => bridge.radiated_share,
        _ => return vec![1.0; partials],
    };
    let modes = BridgeFilter::peaks_only(voicing.bridge.as_ref().expect("checked above"));
    (1..=partials)
        .map(|k| {
            let excess = modes.magnitude(params.partial_freq(k)) - 1.0;
            (1.0 + share * excess).clamp(0.25, 4.0)
        })
        .collect()
}

/// One independent Ornstein-Uhlenbeck detune per mode, in cents, advanced once
/// per block. Seeded, so the rung is the same file every run.
struct Walk {
    state: Vec<f64>,
    rng: SplitMix64,
    decay: f64,
    kick: f64,
}

impl Walk {
    fn new(seed: u64, len: usize) -> Self {
        let dt = BLOCK as f64 / SR;
        let decay = (-dt / WALK_TAU_S).exp();
        Walk {
            state: vec![0.0; len],
            rng: SplitMix64::new(0x_9E37_79B9 ^ seed),
            decay,
            kick: WALK_CENTS * (1.0 - decay * decay).sqrt(),
        }
    }

    fn advance(&mut self) {
        for x in self.state.iter_mut() {
            *x = *x * self.decay + self.kick * self.rng.normal();
        }
    }

    /// The frequency ratio the walk asks for at index `i`.
    fn ratio(&self, i: usize) -> f32 {
        let cents = self.state.get(i).copied().unwrap_or(0.0);
        (cents / 1200.0 * std::f64::consts::LN_2).exp() as f32
    }
}

/// The shipped engine's render of the note, through its public API.
fn render_engine(preset: &EnginePreset, key: u8) -> Stereo {
    let events = [RenderEvent::new(
        PREROLL as f32 / SAMPLE_RATE as f32,
        Event::NoteOn { key, vel: VELOCITY },
    )];
    let (left, right) = render_to_buffer(preset, &events, TOTAL_FRAMES as f32 / SAMPLE_RATE as f32);
    (fit_length(left), fit_length(right))
}

fn fit_length(mut channel: Vec<f32>) -> Vec<f32> {
    channel.resize(TOTAL_FRAMES, 0.0);
    channel
}

/// One rung with the attack residual added at `scale` — the ratio between that
/// rung's level and the recording's, so the residual arrives in the same
/// proportion the piano has it.
fn mixed(base: &Stereo, attack_l: &[f32], attack_r: &[f32], scale: f64) -> Stereo {
    let add = |channel: &[f32], residual: &[f32]| -> Vec<f32> {
        channel
            .iter()
            .zip(residual)
            .map(|(&x, &r)| x + (f64::from(r) * scale) as f32)
            .collect()
    };
    (add(&base.0, attack_l), add(&base.1, attack_r))
}

// ------------------------------------------------------ levels and output

/// RMS of a rung over the level-matching window, both channels together.
fn match_rms(audio: &Stereo) -> f64 {
    rms(
        &audio.0,
        &audio.1,
        PREROLL + (MATCH_LO_S * SR) as usize,
        PREROLL + (MATCH_HI_S * SR) as usize,
    )
}

fn rms(left: &[f32], right: &[f32], from: usize, to: usize) -> f64 {
    let to = to.min(left.len()).min(right.len());
    if to <= from {
        return 0.0;
    }
    let sum: f64 = (from..to)
        .map(|i| f64::from(left[i]).powi(2) + f64::from(right[i]).powi(2))
        .sum();
    (sum / (2 * (to - from)) as f64).sqrt()
}

/// RMS of a set of ratios, in dB — `TUNING_REPORT.md` §3's statistic.
fn rms_db(values: impl Iterator<Item = f64>) -> f64 {
    let db: Vec<f64> = values
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| 20.0 * v.log10())
        .collect();
    if db.is_empty() {
        return 0.0;
    }
    (db.iter().map(|d| d * d).sum::<f64>() / db.len() as f64).sqrt()
}

/// Writes the ladder: every rung matched to `00` over the level window, then a
/// single common gain if anything would clip.
///
/// One common guard, not one per file: the whole point of the ladder is that
/// the rungs are at the same level, so nothing may be scaled on its own after
/// the match.
fn write_matched(dir: &Path, rungs: &[(&str, Stereo)]) -> Result<(), Box<dyn std::error::Error>> {
    let reference = match_rms(&rungs[0].1);
    if reference <= 0.0 || !reference.is_finite() {
        return Err(format!("{}: the source is silent", dir.display()).into());
    }
    let gains: Vec<f64> = rungs
        .iter()
        .map(|(_, audio)| {
            let level = match_rms(audio);
            if level > 0.0 {
                reference / level
            } else {
                0.0
            }
        })
        .collect();
    // Headroom: the source sits where the library put it, and a rung with more
    // crest factor than it must not clip against the same RMS.
    let peak = rungs
        .iter()
        .zip(&gains)
        .map(|((_, (l, r)), &g)| {
            l.iter()
                .chain(r.iter())
                .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()))
                * g
        })
        .fold(0.0f64, f64::max);
    let common = if peak > 0.891 { 0.891 / peak } else { 1.0 };

    for ((name, audio), &gain) in rungs.iter().zip(&gains) {
        write_wav(&dir.join(format!("{name}.wav")), audio, gain * common)?;
    }
    Ok(())
}

fn write_wav(path: &Path, (left, right): &Stereo, gain: f64) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let fade_in = (FADE_IN_S * SR) as usize;
    let fade_out = (FADE_OUT_S * SR) as usize;
    let frames = left.len().min(right.len());
    let mut writer = hound::WavWriter::create(path, spec)?;
    for n in 0..frames {
        let mut envelope = gain;
        if n < fade_in {
            envelope *= 0.5 - 0.5 * (std::f64::consts::PI * n as f64 / fade_in as f64).cos();
        }
        if n + fade_out > frames {
            let u = (n + fade_out - frames) as f64 / fade_out as f64;
            envelope *= 0.5 + 0.5 * (std::f64::consts::PI * u).cos();
        }
        writer.write_sample((f64::from(left[n]) * envelope) as f32)?;
        writer.write_sample((f64::from(right[n]) * envelope) as f32)?;
    }
    writer.finalize()
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B",
    ];
    format!("{}{}", NAMES[usize::from(key) % 12], i32::from(key) / 12 - 1)
}

// ------------------------------------------------------------------ README

fn write_readme(
    out: &Path,
    notes: &[KeyReport],
    preset_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    text.push_str(
        "# timbre-ladder\n\n\
         A level-matched listening ladder from the Salamander recording to the engine's render \
         of the same note, one rung per hypothesis about what the engine is missing. Written by \
         `cargo run --release --example timbre_ladder`; one directory per key, ten files each, \
         4 s, 48 kHz stereo float.\n\n\
         Every rung is RMS-matched to `00` over 0.2-2 s after the strike, and one common gain is \
         applied to the whole group afterwards if anything would clip - so levels are comparable \
         *between* rungs and only between rungs of the same key.\n\n\
         ## The rungs\n\n\
         - `00_source` - the Salamander recording itself, cut to the strike. The reference.\n\
         - `01_resynth_full` - additive resynthesis from the measured trajectories: every tracked \
         partial as a sinusoid following its own measured `a_k(t)` and `f_k(t)`, phases free. \
         The decisive rung - this project's synthesis machinery driven by the recording's exact \
         per-partial content.\n\
         - `02_meas_amp_law_decay` - the measured `a_k(0)` per partial, but every envelope \
         replaced by the fitted two-polarization exponential law: what is lost by describing an \
         envelope with four numbers.\n\
         - `03_smooth_amp_meas_decay` - the mirror: the smooth strike comb's amplitudes with the \
         measured envelopes, so the only thing missing is the excitation roughness.\n\
         - `04_engine_rough` - the engine's modal parameters, rendered offline, with each \
         partial's input gain multiplied by the measured `a_k(0)` roughness.\n\
         - `05_engine_linewidth` - the same, unmodified, plus an independent seeded random-walk \
         detune per partial (2 cents, 1.5 s correlation), giving every partial a finite linewidth.\n\
         - `06_engine_attack` - the shipped engine's render plus the recording's attack residual \
         (source minus phase-locked partial resynthesis, first 150 ms, faded), mixed at the level \
         it has in the recording.\n\
         - `07_engine` - the shipped engine's render of the note, through its public API.\n\
         - `08_resynth_plus_attack` - `01` plus the same attack residual: the ceiling of what \
         per-partial fitting plus a recorded transient could ever deliver.\n\
         - `09_engine_modal_control` - the control for `04` and `05`: the same offline modal \
         renderer with nothing changed. It is the string and the soundboard only - no duplex, no \
         sympathetic bus, no mechanism noise - so `04 - 09` is the roughness alone and `05 - 09` \
         the linewidth alone, while `09 - 07` is everything the engine has beside the string.\n\n\
         ## How to listen\n\n\
         `00` against `01` first. If `01` already sounds synthetic, the missing sound is not a \
         per-partial parameter and rungs `02`-`05` cannot matter; if it does not, the ladder from \
         `01` down to `07` says which parameter it is.\n\n\
         ## What was rendered\n\n\
         | key | layer | partials | excitation roughness | strike comb | attack residual | \
         first tracked frame |\n\
         |:--|--:|--:|--:|:--|--:|--:|\n",
    );
    for note in notes {
        text.push_str(&format!(
            "| {} | {} | {} | {:.2} dB | {} | {:.1} dB | {:.0} ms |\n",
            note.name,
            note.layer,
            note.partials,
            note.roughness_db,
            if note.fitted_comb {
                "fitted"
            } else {
                "preset's own"
            },
            note.residual_db,
            note.first_frame_ms,
        ));
    }
    text.push_str(&format!(
        "\nSource: the Salamander Grand Piano library at velocity {VELOCITY} (nearest layer), \
         preset `{preset_path}`, trajectories from the survey's cache.\n\n\
         ## Limits of the ladder\n\n\
         - **The head of the note is modelled, not measured.** The tracker times a frame at the \
         centre of its analysis window, so nothing exists before half a window (the last column \
         above: 43 ms at C4 and C6, 85 ms at A2). Inside that stretch the additive rungs follow \
         the fitted two-exponential law extrapolated back to the strike, scaled to meet the first \
         real measurement, and given a 2 ms raised-cosine rise.\n\
         - **The additive rungs carry one static stereo balance per partial**, measured over \
         0.2-2 s: the trajectories come from the mono sum, and one number per partial cannot \
         reproduce the 1-6 dB of balance *drift* `TUNING_REPORT.md` \u{a7}5 measures.\n\
         - **The additive rungs are only the tracked partials.** Where the recording's late \
         energy is not the struck string - `TUNING_REPORT.md` \u{a7}4 puts C6's between-partial \
         energy one second in at -22 dB against the engine's -48 - rungs `01`-`03` do not have \
         it, and matching their level to `00` over 0.2-2 s then lifts the partials they do have \
         by a few dB.\n\
         - **`04`, `05` and `09` are the string and the board only.** No duplex, no sympathetic \
         bus, no mechanism noise, no felt limiter: compare them with each other, and use `07` for \
         what the whole engine does.\n"
    ));
    std::fs::write(out.join("README.md"), text)?;
    Ok(())
}
