//! Accuracy of the partial tracker on synthetic signals — sums of decaying
//! sinusoids whose frequencies, amplitudes and decay rates are known exactly.
//!
//! The bar `TUNING.md` sets for the analysis front end is 0.1 Hz on frequency
//! and 5 % on the amplitude envelope at 40 dB SNR, over the range where a
//! partial actually stands above the noise. Every test below states its own
//! measured margin in its assertion message, so a regression says how far it
//! moved rather than only that it moved.

use piano_tuner::synth::{Partial, Tone};
use piano_tuner::{InharmonicModel, PartialTracker, Stft, StftConfig, TrackerConfig};

const SAMPLE_RATE: f64 = 48_000.0;

/// 2^14 samples (341 ms) advanced by 10 ms, transformed at 2^15. Shorter than
/// the 2^16 `TUNING.md` specifies for real recordings — the synthetic notes
/// here are seconds long, not tens of seconds, and the shorter window keeps
/// the test suite quick without changing anything it measures.
fn test_config() -> TrackerConfig {
    TrackerConfig {
        stft: StftConfig::padded(1 << 14, 480, 2).unwrap(),
        ..TrackerConfig::default()
    }
}

/// The largest spectral peak white noise alone produces through the same
/// transform: the level below which no measurement can be trusted. Deriving it
/// by measurement rather than by formula keeps the tests honest about the
/// window they actually use.
fn noise_peak_amplitude(config: &TrackerConfig, level: f64, seed: u64, frames: usize) -> f64 {
    let mut noise = vec![0.0f32; frames];
    piano_tuner::synth::add_white_noise(&mut noise, level, seed);
    let stft = Stft::new(config.stft).unwrap();
    let mut peak = 0.0f64;
    stft.for_each_frame(&noise, SAMPLE_RATE, |_, magnitude| {
        for &m in magnitude {
            peak = peak.max(f64::from(m));
        }
    });
    peak
}

struct Accuracy {
    frames: usize,
    max_frequency_error_hz: f64,
    max_amplitude_error: f64,
    median_frequency_error_hz: f64,
    median_amplitude_error: f64,
}

/// Compare a recovered track against the partial that produced it, over the
/// frames where the true envelope is at least `margin` times `floor`.
fn compare(track: &piano_tuner::PartialTrack, truth: &Partial, onset_s: f64, floor: f64, margin: f64) -> Accuracy {
    let mut frequency = Vec::new();
    let mut amplitude = Vec::new();
    for point in &track.points {
        let expected = truth.amplitude_at(point.time_s - onset_s);
        if expected < floor * margin {
            continue;
        }
        frequency.push((point.frequency_hz - truth.frequency_hz).abs());
        amplitude.push((point.amplitude / expected - 1.0).abs());
    }
    let median = |v: &mut Vec<f64>| {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Accuracy {
        frames: frequency.len(),
        max_frequency_error_hz: frequency.iter().copied().fold(0.0, f64::max),
        max_amplitude_error: amplitude.iter().copied().fold(0.0, f64::max),
        median_frequency_error_hz: median(&mut frequency.clone()),
        median_amplitude_error: median(&mut amplitude.clone()),
    }
}

#[test]
fn a_single_decaying_sinusoid_is_recovered_exactly() {
    // 441.37 Hz sits between bins at every window length used here, so no
    // part of the accuracy can come from the peak happening to land on a bin.
    let truth = Partial::new(1, 441.37, 0.5, 2.0).with_phase(0.9);
    let tone = Tone::new(SAMPLE_RATE, 3.0, vec![truth]);
    let signal = tone.render();

    let tracker = PartialTracker::new(test_config()).unwrap();
    let trajectories = tracker.track(&signal, SAMPLE_RATE, InharmonicModel::harmonic(441.0));

    let track = trajectories.track(1).expect("no fundamental");
    let accuracy = compare(track, &truth, 0.0, 1e-9, 1.0);
    assert!(accuracy.frames > 250, "only {} frames", accuracy.frames);
    // Measured: 4.6e-3 Hz worst case, i.e. 3e-3 of a bin at this window.
    assert!(
        accuracy.max_frequency_error_hz < 0.02,
        "frequency error {:.2e} Hz",
        accuracy.max_frequency_error_hz
    );
    // The measured envelope is the window-weighted mean of a 2.0/s decay, 6 %
    // above the true value at the window centre before compensation.
    assert!(
        accuracy.max_amplitude_error < 0.002,
        "amplitude error {:.3} %",
        100.0 * accuracy.max_amplitude_error
    );
}

#[test]
fn the_decay_compensation_removes_the_windowing_bias() {
    let truth = Partial::new(1, 441.37, 0.5, 6.0);
    let tone = Tone::new(SAMPLE_RATE, 1.5, vec![truth]);
    let signal = tone.render();

    let compensated = PartialTracker::new(test_config()).unwrap();
    let raw = PartialTracker::new(TrackerConfig {
        decay_compensation: false,
        ..test_config()
    })
    .unwrap();

    let seed = InharmonicModel::harmonic(441.0);
    let with = compare(
        compensated
            .track(&signal, SAMPLE_RATE, seed)
            .track(1)
            .unwrap(),
        &truth,
        0.0,
        1e-9,
        1.0,
    );
    let without = compare(
        raw.track(&signal, SAMPLE_RATE, seed).track(1).unwrap(),
        &truth,
        0.0,
        1e-9,
        1.0,
    );

    // A 6/s decay through a 341 ms window reads 9 % high uncompensated
    // (G(1.02) = 1.09); the correction has to bring that inside 1 %.
    assert!(
        without.median_amplitude_error > 0.05,
        "uncompensated error only {:.2} % — is the bias still there?",
        100.0 * without.median_amplitude_error
    );
    assert!(
        with.max_amplitude_error < 0.01,
        "compensated error {:.2} %",
        100.0 * with.max_amplitude_error
    );
}

#[test]
fn an_inharmonic_partial_series_is_recovered_at_40_db_snr() {
    // A2-ish: 12 partials of a stiff string, decays running from 1.0/s on the
    // fundamental to 8.4/s on the twelfth.
    let seed = InharmonicModel::new(110.31, 3.7e-4);
    let tone = Tone::from_model(seed, 12, 0.7, 1.2, SAMPLE_RATE, 4.0);
    let config = test_config();
    let signal = tone.render_with_noise(40.0, 20_260_813);

    let noise_level = piano_tuner::synth::rms(&tone.render()) * 10f64.powf(-40.0 / 20.0);
    let floor = noise_peak_amplitude(&config, noise_level, 20_260_813, tone.frames());

    let tracker = PartialTracker::new(config).unwrap();
    // Seed the search 12 cents flat of the truth: the tracker must find the
    // partials from an approximate model, which is all an estimator ever has.
    let trajectories = tracker.track(
        &signal,
        SAMPLE_RATE,
        InharmonicModel::new(seed.f0_hz * 0.993, 3.0e-4),
    );

    let (mut worst_f, mut worst_a) = (0.0f64, 0.0f64);
    for k in 1..=12u32 {
        let truth = tone.partial(k).unwrap();
        let track = trajectories
            .track(k)
            .unwrap_or_else(|| panic!("partial {k} was not tracked"));
        // 20x the worst noise peak: below that the noise alone can move an
        // amplitude by 5 %, so there is nothing left for the tracker to spend.
        let accuracy = compare(track, truth, 0.0, floor, 20.0);
        assert!(
            accuracy.frames >= 20,
            "partial {k}: only {} usable frames",
            accuracy.frames
        );
        assert!(
            accuracy.max_frequency_error_hz < 0.1,
            "partial {k} ({:.1} Hz): frequency error {:.4} Hz over {} frames",
            truth.frequency_hz,
            accuracy.max_frequency_error_hz,
            accuracy.frames
        );
        assert!(
            accuracy.max_amplitude_error < 0.05,
            "partial {k} ({:.1} Hz): amplitude error {:.2} % over {} frames",
            truth.frequency_hz,
            100.0 * accuracy.max_amplitude_error,
            accuracy.frames
        );
        worst_f = worst_f.max(accuracy.max_frequency_error_hz);
        worst_a = worst_a.max(accuracy.max_amplitude_error);
        println!(
            "k={k:2} f={:8.2} Hz  frames {:3}  |df| max {:.4} Hz median {:.4} Hz  \
             |da/a| max {:5.2} % median {:5.2} %",
            truth.frequency_hz,
            accuracy.frames,
            accuracy.max_frequency_error_hz,
            accuracy.median_frequency_error_hz,
            100.0 * accuracy.max_amplitude_error,
            100.0 * accuracy.median_amplitude_error
        );
    }
    println!("worst over all partials: {worst_f:.4} Hz, {:.2} %", 100.0 * worst_a);
}

#[test]
fn the_default_window_resolves_a_bass_note() {
    // A0: partials 27.6 Hz apart and decays measured in tens of seconds. This
    // is the case `TUNING.md`'s >= 2^16 window exists for — at the 341 ms
    // window the rest of these tests use, a Hann main lobe is 11.7 Hz wide and
    // the low partials of a detuned unison would merge.
    let seed = InharmonicModel::new(27.57, 1.1e-4);
    let tone = Tone::from_model(seed, 20, 0.28, 0.9, SAMPLE_RATE, 6.0);
    let signal = tone.render_with_noise(50.0, 5);

    let tracker = PartialTracker::new(TrackerConfig::default()).unwrap();
    let trajectories = tracker.track(&signal, SAMPLE_RATE, seed);

    for k in 1..=20u32 {
        let truth = tone.partial(k).unwrap();
        let track = trajectories
            .track(k)
            .unwrap_or_else(|| panic!("partial {k} ({:.1} Hz) was not tracked", truth.frequency_hz));
        let accuracy = compare(track, truth, 0.0, 1e-4, 1.0);
        assert!(
            accuracy.max_frequency_error_hz < 0.1,
            "partial {k} ({:.2} Hz): frequency error {:.4} Hz",
            truth.frequency_hz,
            accuracy.max_frequency_error_hz
        );
        assert!(
            accuracy.max_amplitude_error < 0.05,
            "partial {k} ({:.2} Hz): amplitude error {:.2} %",
            truth.frequency_hz,
            100.0 * accuracy.max_amplitude_error
        );
    }
}

#[test]
fn the_measured_frequencies_follow_the_string_and_not_the_seed() {
    // The seed is a semitone-flat guess with no inharmonicity at all; the
    // recovered frequencies must be the ones that were rendered, so that an
    // f0/B fit downstream sees the string rather than its own prior.
    let truth = InharmonicModel::new(261.63, 8e-4);
    let tone = Tone::from_model(truth, 8, 0.8, 1.0, SAMPLE_RATE, 2.5);
    let signal = tone.render();

    let tracker = PartialTracker::new(test_config()).unwrap();
    let trajectories = tracker.track(&signal, SAMPLE_RATE, InharmonicModel::harmonic(258.0));

    for k in 1..=8u32 {
        let track = trajectories.track(k).unwrap_or_else(|| panic!("partial {k} missing"));
        let measured = track.weighted_frequency().unwrap();
        let expected = truth.partial(k);
        assert!(
            (measured - expected).abs() < 0.05,
            "partial {k}: measured {measured:.3} Hz, rendered {expected:.3} Hz, \
             seed said {:.3} Hz",
            trajectories.seed.partial(k)
        );
    }
}

#[test]
fn a_note_that_starts_late_is_tracked_from_its_onset() {
    let truth = InharmonicModel::new(196.4, 2e-4);
    let tone = Tone::from_model(truth, 6, 0.9, 1.0, SAMPLE_RATE, 3.0).with_onset(0.25);
    let signal = tone.render_with_noise(60.0, 11);

    let tracker = PartialTracker::new(test_config()).unwrap();
    let trajectories = tracker.track(&signal, SAMPLE_RATE, truth);

    assert!(
        (trajectories.onset_s - 0.25).abs() < 0.01,
        "onset reported at {:.4} s",
        trajectories.onset_s
    );
    for k in 1..=6u32 {
        let track = trajectories.track(k).unwrap_or_else(|| panic!("partial {k} missing"));
        let partial = tone.partial(k).unwrap();
        // Only frames whose whole window sits after the strike can carry a
        // meaningful envelope; before that the window is half silence.
        let accuracy = compare(track, partial, tone.onset_s, 1e-4, 1.0);
        let usable: Vec<_> = track
            .points
            .iter()
            .filter(|p| p.time_s > tone.onset_s + 0.5 * trajectories.window_s)
            .collect();
        assert!(usable.len() > 100, "partial {k}: {} usable frames", usable.len());
        assert!(
            accuracy.median_amplitude_error < 0.05,
            "partial {k}: median amplitude error {:.2} %",
            100.0 * accuracy.median_amplitude_error
        );
    }
}

#[test]
fn trajectories_survive_a_round_trip_through_the_cache() {
    let truth = InharmonicModel::new(329.9, 5e-4);
    let tone = Tone::from_model(truth, 5, 1.0, 1.5, SAMPLE_RATE, 1.5);
    let tracker = PartialTracker::new(test_config()).unwrap();
    let trajectories = tracker
        .track(&tone.render(), SAMPLE_RATE, truth)
        .with_source("synthetic E4")
        .with_note(piano_tuner::NoteId::layer(64, 12));

    let path = std::env::temp_dir().join("piano-tuner-trajectory-cache.json");
    trajectories.write_json(&path).unwrap();
    let back = piano_tuner::NoteTrajectories::read_json(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(back.source, "synthetic E4");
    assert_eq!(back.note, trajectories.note);
    assert_eq!(back.point_count(), trajectories.point_count());
    for (a, b) in back.tracks.iter().zip(&trajectories.tracks) {
        assert_eq!(a.k, b.k);
        assert_eq!(a.points, b.points);
    }
}
