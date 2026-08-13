//! Decoding and resampling against a checked-in FLAC fixture.
//!
//! `tests/fixtures/tone-44k1-stereo.flac` is 0.5 s of 16-bit stereo at 44.1 kHz:
//! a 1000 Hz sine at amplitude 0.5 on the left, 2500 Hz at 0.25 on the right.
//! It is deliberately not produced by this crate — it was encoded by `flac`
//! (through ffmpeg) from PCM this test regenerates — so the test exercises a
//! real encoder's output rather than a round trip through our own assumptions.

use std::f64::consts::PI;
use std::path::PathBuf;

use piano_tuner::{audio, InharmonicModel, PartialTracker, StftConfig, TrackerConfig};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tone-44k1-stereo.flac")
}

/// The PCM the fixture was encoded from, as f32 in the loader's convention.
fn reference(freq: f64, amplitude: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / 44_100.0;
            let quantised = (amplitude * (2.0 * PI * freq * t).sin() * 32767.0).round();
            (quantised / 32768.0) as f32
        })
        .collect()
}

#[test]
fn a_flac_decodes_sample_for_sample() {
    let decoded = audio::load(fixture()).unwrap();
    assert_eq!(decoded.sample_rate, 44_100);
    assert_eq!(decoded.channel_count(), 2);
    assert_eq!(decoded.frames(), 22_050);
    assert!((decoded.duration_s() - 0.5).abs() < 1e-9);

    // FLAC is lossless, so this is equality, not a tolerance.
    assert_eq!(decoded.channels[0], reference(1000.0, 0.5, 22_050));
    assert_eq!(decoded.channels[1], reference(2500.0, 0.25, 22_050));
}

#[test]
fn loading_at_the_engines_rate_resamples_and_keeps_both_tones() {
    let decoded = audio::load_at(fixture(), 48_000).unwrap();
    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(decoded.frames(), 24_000);

    // Track both tones out of the mono mix; the mean of the two channels
    // halves each amplitude.
    let tracker = PartialTracker::new(TrackerConfig {
        stft: StftConfig::padded(1 << 13, 480, 4).unwrap(),
        max_partials: 3,
        ..TrackerConfig::default()
    })
    .unwrap();
    let signal = decoded.mono();
    for (freq, amplitude) in [(1000.0f64, 0.25f64), (2500.0, 0.125)] {
        let trajectories = tracker.track(&signal, 48_000.0, InharmonicModel::harmonic(freq));
        let track = trajectories.track(1).expect("tone not tracked");
        let measured = track.weighted_frequency().unwrap();
        let peak = track.peak().unwrap().amplitude;
        assert!((measured - freq).abs() < 0.05, "{freq} Hz read as {measured:.4} Hz");
        assert!(
            (peak - amplitude).abs() < 0.01 * amplitude,
            "{freq} Hz read at amplitude {peak:.5}, expected {amplitude}"
        );
    }
}

#[test]
fn an_unknown_extension_is_refused_rather_than_guessed() {
    assert!(audio::load("recording.mp3").is_err());
    assert!(audio::load("recording").is_err());
}
