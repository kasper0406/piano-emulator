//! The caches under `data/cache/` are only worth having if a hit is the same
//! answer as a miss. These are the tests that say so.
//!
//! Two things are cached (`piano_tuner::cache`, `DECISIONS.md` 284): the
//! *reference* renders the `bench` and `compass` subcommands
//! score the engine against, as 32-bit float WAV; and the self-calibration
//! gate's corpus of tracked notes, as the compact encoding
//! `NoteTrajectories` implements. Both claim to be **bit-identical** across the
//! round trip rather than merely close, because both stand behind a measurement
//! that is quoted to hundredths of a decibel and diffed between runs.

use std::path::{Path, PathBuf};

use piano_tuner::cache::{self, Cacheable};
use piano_tuner::realism::PHRASE_SET_VERSION;
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::trajectory::{InharmonicModel, NoteId, NoteTrajectories, PartialTrack, TrackPoint};
use piano_tuner::{realism, Audio, Sampler, SAMPLE_RATE};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// The Salamander library, if this working tree has it. It is 707 MiB and
/// gitignored, so a checkout that has not run `data/fetch_salamander.sh` skips
/// the render half of this file rather than failing it — exactly as the
/// subcommands that need it do.
fn sfz() -> Option<PathBuf> {
    let path = repo()
        .join("data/salamander")
        .join("SalamanderGrandPiano-V3+20200602.sfz");
    path.exists().then_some(path)
}

/// One phrase, rendered through the sampler twice: once cold into a fresh cache
/// directory, and once out of it. Every sample of the two has to be the same
/// float, not a close one — a cache that quietly rounds would move every number
/// in `REALISM.md` by an amount nobody could account for.
#[test]
fn a_cached_reference_render_is_the_render_it_replaced() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the render half");
        return;
    };
    let phrase = realism::phrase_set()
        .into_iter()
        .next()
        .expect("the phrase set is not empty");

    let mut key = cache::Fingerprint::new();
    key.str("tests/reference_cache")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)
        .expect("the sfz is readable")
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(PHRASE_SET_VERSION))
        .str(phrase.name)
        .f64(phrase.duration_s);

    // Its own directory, removed first, so the test measures a real miss
    // followed by a real hit and cannot be fooled by a neighbour's entry.
    let dir = repo().join("data/cache/reference/tests");
    let path = dir.join(format!("{}-{}.wav", phrase.name, key.hex()));
    let _ = std::fs::remove_file(&path);

    let render = || {
        let mut sampler = Sampler::new(&sfz)?;
        sampler.render(&phrase.events, phrase.duration_s)
    };
    let fresh = cache::audio(&path, render).expect("the reference renders");
    assert!(path.exists(), "the miss did not write an entry");
    let hit = cache::audio(&path, || panic!("the second call must not re-render"))
        .expect("the entry reloads");

    assert_eq!(fresh.sample_rate, hit.sample_rate);
    assert_eq!(fresh.channels.len(), hit.channels.len());
    for (c, (a, b)) in fresh.channels.iter().zip(&hit.channels).enumerate() {
        assert_eq!(a.len(), b.len(), "channel {c} changed length");
        // Bitwise, not `==`: the point is that no float moved at all.
        let moved = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(moved, 0, "channel {c}: {moved} samples came back different");
    }

    // And the same render again from scratch is the same bytes, which is the
    // property the cache key rests on: the reference is a function of its
    // inputs and of nothing else.
    let again = render().expect("the reference renders twice");
    for (c, (a, b)) in fresh.channels.iter().zip(&again.channels).enumerate() {
        let moved = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(moved, 0, "channel {c}: the sampler is not reproducible");
    }
}

/// A phrase the *pedal* lifts a chord under, rendered twice in one process.
///
/// This is the regression for `DECISIONS.md` 284's determinism finding: the
/// order sounding keys were released in used to be a `HashMap`'s, so a pedal-up
/// summed a chord's release voices in an order that was reseeded per process and
/// two runs of `bench` wrote reference audio an ulp apart. Nothing in
/// `REALISM.md` moved, which is exactly why it went unnoticed.
#[test]
fn the_pedal_releases_a_chord_in_the_same_order_every_time() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping");
        return;
    };
    let phrase = realism::phrase_set()
        .into_iter()
        .find(|p| p.name.contains("pedal"))
        .expect("the phrase set has a pedalled phrase");

    let render = || -> Audio {
        let mut sampler = Sampler::new(&sfz).expect("the sfz loads");
        sampler
            .render(&phrase.events, phrase.duration_s)
            .expect("the phrase renders")
    };
    let (a, b) = (render(), render());
    for (c, (x, y)) in a.channels.iter().zip(&b.channels).enumerate() {
        let moved = x
            .iter()
            .zip(y)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        assert_eq!(
            moved, 0,
            "channel {c}: {moved} samples of a pedalled phrase differ between two renders"
        );
    }
}

/// The gate's corpus encoding, over a value with every field of every struct
/// set to something a default would not produce — including the two floats no
/// text format survives, a signed zero and a subnormal.
#[test]
fn a_corpus_entry_decodes_to_the_bits_that_were_encoded() {
    let value = NoteTrajectories {
        source: "a recording with a \"quoted\" name".into(),
        note: Some(NoteId::layer(33, 7)),
        sample_rate: 48_000.0,
        window_s: 0.170_666_666_666_666_66,
        hop_s: 0.01,
        seed: InharmonicModel {
            f0_hz: 54.971_327_053_275_004,
            b: 2.053_456_281_414_472e-4,
            b4: -2.0e-8,
        },
        onset_s: 0.049,
        tracks: vec![
            PartialTrack {
                k: 1,
                points: vec![
                    TrackPoint {
                        time_s: 0.085_333_333_333_333_33,
                        frequency_hz: 53.221_034_169_106_65,
                        amplitude: 1.296_260_257_505_008e-3,
                    },
                    TrackPoint {
                        time_s: -0.0,
                        frequency_hz: f64::MIN_POSITIVE / 2.0,
                        amplitude: f64::MAX,
                    },
                ],
            },
            // A partial that found nothing is still a track.
            PartialTrack { k: 80, points: Vec::new() },
        ],
    };

    let back = NoteTrajectories::decode(&value.encode()).expect("the entry decodes");
    assert_eq!(back.source, value.source);
    assert_eq!(back.note, value.note);
    assert_eq!(back.sample_rate.to_bits(), value.sample_rate.to_bits());
    assert_eq!(back.window_s.to_bits(), value.window_s.to_bits());
    assert_eq!(back.hop_s.to_bits(), value.hop_s.to_bits());
    assert_eq!(back.seed.f0_hz.to_bits(), value.seed.f0_hz.to_bits());
    assert_eq!(back.seed.b.to_bits(), value.seed.b.to_bits());
    assert_eq!(back.seed.b4.to_bits(), value.seed.b4.to_bits());
    assert_eq!(back.onset_s.to_bits(), value.onset_s.to_bits());
    assert_eq!(back.tracks.len(), value.tracks.len());
    for (a, b) in back.tracks.iter().zip(&value.tracks) {
        assert_eq!(a.k, b.k);
        assert_eq!(a.points.len(), b.points.len());
        for (p, q) in a.points.iter().zip(&b.points) {
            // `-0.0 == 0.0` in Rust, so the comparison is on the bits.
            assert_eq!(p.time_s.to_bits(), q.time_s.to_bits());
            assert_eq!(p.frequency_hz.to_bits(), q.frequency_hz.to_bits());
            assert_eq!(p.amplitude.to_bits(), q.amplitude.to_bits());
        }
    }
}

/// Nothing a corrupt or foreign file can contain is allowed to be read as a
/// corpus entry, because a silent misread would be a wrong measurement rather
/// than a slow one.
#[test]
fn a_damaged_corpus_entry_is_a_miss_and_not_a_panic() {
    let value = NoteTrajectories {
        source: String::new(),
        note: None,
        sample_rate: 48_000.0,
        window_s: 0.17,
        hop_s: 0.01,
        seed: InharmonicModel::new(440.0, 1e-4),
        onset_s: 0.05,
        tracks: vec![PartialTrack {
            k: 1,
            points: vec![TrackPoint { time_s: 0.0, frequency_hz: 440.0, amplitude: 0.5 }],
        }],
    };
    let good = value.encode();
    assert!(NoteTrajectories::decode(&good).is_some());

    assert!(NoteTrajectories::decode(&[]).is_none(), "empty");
    assert!(NoteTrajectories::decode(b"not a corpus entry at all").is_none(), "foreign");
    for cut in 1..good.len() {
        assert!(
            NoteTrajectories::decode(&good[..cut]).is_none(),
            "a file truncated to {cut} bytes decoded"
        );
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert!(NoteTrajectories::decode(&trailing).is_none(), "trailing bytes");
    // A length field that claims more than the file holds must not be trusted.
    let mut absurd = good.clone();
    let at = absurd.len() - 8 - 24 - 4 - 8;
    absurd[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(NoteTrajectories::decode(&absurd).is_none(), "an absurd count");
}

/// The cache directories are the ones `README.md` and `DECISIONS.md` 284 name,
/// and they are under the gitignored `data/`.
#[test]
fn the_caches_live_where_they_are_documented_to() {
    assert!(cache::reference_dir(Path::new("data/salamander")).ends_with("data/cache/reference"));
    assert!(cache::calibration_dir(Path::new("/repo")).ends_with("data/cache/calibration"));
}
