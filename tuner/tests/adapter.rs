//! The library-description layer, and the one property it must not cost.
//!
//! `piano_tuner::adapter` generalizes what the factory assumed about a sample
//! library — that it ships an SFZ, that its keys are minor thirds, that it has
//! sixteen velocity layers, that it is already at 48 kHz. Generalizing it
//! touches nothing in [`Sampler`], which is why the Salamander reference is
//! expected to be bit-exact; **expected** is not a proof, and the boards of
//! this repository are all barred against numbers measured through that
//! reference, so it gets one.
//!
//! [`the_salamander_reference_render_is_bit_exact`] renders the six benchmark
//! phrases — `realism::phrase_set`, the same list `bench`, `compass`, `melody`
//! and `stereo` all drive both sides from — through the sampler on the shipped
//! Salamander SFZ and hashes every sample of every channel. The hash is
//! pinned. Any change that moves one sample of the reference moves it, which
//! makes this the falsification the adapter work did not otherwise have: an
//! adapter that had quietly re-read Salamander through the generated path, or
//! a `Bands`/`Layout` refactor that had leaked into `library.rs`'s opcode
//! reader, would fail here and nowhere else until a board moved.
//!
//! The corpus is 707 MiB and gitignored, so every test here skips itself
//! without it, as the other corpus tests do.

use std::path::{Path, PathBuf};

use piano_tuner::adapter::{Bands, Layout, LibrarySpec, Source};
use piano_tuner::library::MechanismKind;
use piano_tuner::realism::phrase_set;
use piano_tuner::{Audio, SampleLibrary, Sampler, SamplerConfig};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the tuner sits in the workspace")
        .to_path_buf()
}

fn salamander_sfz() -> PathBuf {
    repo()
        .join("data/salamander")
        .join("SalamanderGrandPiano-V3+20200602.sfz")
}

/// FNV-1a over the raw bit patterns of every sample, channel by channel.
///
/// Not a cryptographic digest and does not need to be: what it has to do is
/// change when any sample changes, and be reproducible across machines and
/// process runs. It reads the `f32` bits rather than the value so that a
/// signed zero or a NaN payload cannot slip through as "equal".
fn hash_audio(audio: &Audio) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for byte in audio.sample_rate.to_le_bytes() {
        eat(byte);
    }
    for channel in &audio.channels {
        for byte in (channel.len() as u64).to_le_bytes() {
            eat(byte);
        }
        for sample in channel {
            for byte in sample.to_bits().to_le_bytes() {
                eat(byte);
            }
        }
    }
    hash
}

/// **The bit-exactness pin.** FNV-1a over the sampler's render of all six
/// benchmark phrases from the shipped Salamander SFZ, at the default sampler
/// configuration.
///
/// Measured 2026-08-20, on the tree the adapter landed in, and identical to
/// the same measurement on the tree without the adapter in it (`DECISIONS.md`
/// 517). **Do not update this constant to make a test pass.** It moving means
/// the reference every board in this repository is barred against has moved,
/// which is either a `SAMPLER_VERSION` bump the author owes an explanation
/// for, or a bug.
const SALAMANDER_REFERENCE_HASH: u64 = 4_099_710_989_447_995_248;

#[test]
fn the_salamander_reference_render_is_bit_exact() {
    let sfz = salamander_sfz();
    if !sfz.is_file() {
        eprintln!("no data/salamander in this tree; skipping the bit-exactness pin");
        return;
    }
    let mut sampler = Sampler::with_config(&sfz, SamplerConfig::default()).expect("the SFZ loads");
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut names = Vec::new();
    for phrase in phrase_set() {
        let audio = sampler
            .render(&phrase.events, phrase.duration_s)
            .expect("the reference renders");
        // Fold each phrase's hash into the running one, so the pin is a
        // function of all six and of the order they are rendered in.
        hash ^= hash_audio(&audio);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        names.push(phrase.name);
    }
    assert_eq!(names.len(), 6, "the benchmark phrase set changed shape");
    assert_eq!(
        hash, SALAMANDER_REFERENCE_HASH,
        "the Salamander reference moved. Every bar in this repository was \
         measured through it; see this file's header before touching the pin."
    );
}

/// The other half of the same claim, and the cheaper one: the adapter refuses
/// to generate a map for a library that ships one, so there is no path by
/// which Salamander could be played from anything but its own file.
#[test]
fn salamander_is_never_generated() {
    let spec = LibrarySpec::find("salamander").expect("salamander is described");
    assert!(matches!(spec.source, Source::Shipped(_)));
    assert!(
        spec.emit_sfz(&repo().join("data/salamander")).is_err(),
        "a shipped map must not be replaceable by a generated one"
    );
}

/// The two new libraries, checked against their own trees: the description is
/// a claim about what is on disk, and this is where the claim is tested.
///
/// Skips without the tree, like every corpus test here.
#[test]
fn the_new_libraries_are_complete_where_they_are_fetched() {
    for (id, dir, notes, mechanism) in [
        ("bitklavier-piano-bar", "data/bitklavier-piano-bar", 480, 182),
        ("vcsl-knight-upright", "data/vcsl-knight-upright", 90, 53),
    ] {
        let root = repo().join(dir);
        if !root.is_dir() {
            eprintln!("no {dir} in this tree; skipping");
            continue;
        }
        let spec = LibrarySpec::find(id).expect("described");
        let scan = spec.scan(&root);
        assert_eq!(
            scan.present_notes(),
            notes,
            "{id}: {} of {} note recordings present; the first missing is {:?}",
            scan.present_notes(),
            scan.notes.len(),
            scan.missing_notes().first().map(|n| &n.relative)
        );
        assert_eq!(scan.present_mechanism(), mechanism, "{id}: mechanism census");
        assert_eq!(scan.recorded_keys().len(), spec.layout.keys().len());
    }
}

/// A generated map has to be readable by the two things that read maps, and
/// has to say the same thing to both. `library.rs` decides what is *fitted*;
/// `sampler.rs` decides what is *rendered*; a map they disagree about would
/// fit one instrument and score against another.
#[test]
fn the_generated_maps_read_the_same_to_the_survey_and_to_the_player() {
    for (id, dir, keys, layers) in [
        ("bitklavier-piano-bar", "data/bitklavier-piano-bar", 30, 16),
        ("vcsl-knight-upright", "data/vcsl-knight-upright", 45, 2),
    ] {
        let sfz = repo().join(dir).join(format!("{id}.sfz"));
        if !sfz.is_file() {
            eprintln!("no {dir} in this tree; skipping");
            continue;
        }
        let library = SampleLibrary::from_sfz(&sfz).expect("the generated map parses");
        assert_eq!(library.key_count(), keys, "{id}: recorded keys");
        assert_eq!(library.sample_count(), keys * layers, "{id}: recordings");
        for key in library.keys() {
            assert_eq!(library.layers(key).len(), layers, "{id}: key {key}");
        }
        // No gain and no velocity law asserted, on any region, anywhere.
        assert!(
            library.samples().all(|s| s.volume_db == 0.0),
            "{id}: a generated map states a gain"
        );

        let player = piano_tuner::Instrument::from_sfz(&sfz).expect("the player reads it");
        assert!(
            player.ignored_opcodes().is_empty(),
            "{id}: the player skipped {:?}",
            player.ignored_opcodes()
        );
        // The player's recorded-key set is the survey's, which is the
        // property `Instrument::rerouted` and the evaluation policy both rest
        // on.
        let mut played: Vec<u8> = player
            .regions()
            .iter()
            .filter_map(|r| r.recorded_key())
            .collect();
        played.sort_unstable();
        played.dedup();
        let surveyed: Vec<u8> = library.keys().collect();
        for key in &surveyed {
            assert!(played.contains(key), "{id}: {key} is surveyed but not played");
        }
    }
}

/// The mechanism census, per library, and the two places the libraries differ
/// from Salamander in a way a stage has to know about.
#[test]
fn the_mechanism_each_library_offers_is_what_its_description_says() {
    /// One library's census: its id, the tree it lives in, and how many
    /// recordings of each mechanism kind its description promises.
    type Census = (&'static str, &'static str, [(MechanismKind, usize); 4]);
    let cases: [Census; 2] = [
        (
            "bitklavier-piano-bar",
            "data/bitklavier-piano-bar",
            [
                (MechanismKind::KeyOff, 88),
                (MechanismKind::StringResonance, 90),
                (MechanismKind::PedalDown, 2),
                (MechanismKind::PedalUp, 2),
            ],
        ),
        (
            "vcsl-knight-upright",
            "data/vcsl-knight-upright",
            [
                // The upright ships NO unpitched key-off group at all: its 45
                // releases are the strings still ringing. `noise`'s key-off
                // term has no material here, and that is a property of the
                // library rather than of the stage.
                (MechanismKind::KeyOff, 0),
                (MechanismKind::StringResonance, 45),
                (MechanismKind::PedalDown, 4),
                (MechanismKind::PedalUp, 4),
            ],
        ),
    ];
    for (id, dir, expected) in cases {
        let sfz = repo().join(dir).join(format!("{id}.sfz"));
        if !sfz.is_file() {
            eprintln!("no {dir} in this tree; skipping");
            continue;
        }
        let library = SampleLibrary::from_sfz(&sfz).expect("parses");
        for (kind, count) in expected {
            assert_eq!(
                library.mechanism_of(kind).len(),
                count,
                "{id}: {kind:?}"
            );
        }
    }
}

/// The velocity bands are the abscissa of every velocity fit, so what they are
/// is a decision and not an implementation detail (`DECISIONS.md` 519).
#[test]
fn the_velocity_abscissa_of_each_library_is_pinned() {
    let salamander = LibrarySpec::find("salamander").unwrap();
    let midpoints: Vec<u8> = salamander
        .bands
        .bands()
        .iter()
        .map(|&(lo, hi)| ((u16::from(lo) + u16::from(hi)) / 2) as u8)
        .collect();
    // Salamander's own bands, and they are markedly uneven: the softest is
    // twenty-six velocities wide and the third is two.
    assert_eq!(
        midpoints,
        vec![13, 30, 35, 40, 45, 48, 53, 60, 68, 76, 84, 92, 100, 108, 116, 124]
    );

    // bitKlavier's sixteen were built "relatively evenly distributed across
    // the dynamic range" and it ships no map, so even bands are what is
    // asserted — and they are NOT Salamander's.
    let grand = LibrarySpec::find("bitklavier-piano-bar").unwrap();
    assert_eq!(grand.bands, Bands::Even(16));
    let grand_midpoints: Vec<u8> = grand
        .bands
        .bands()
        .iter()
        .map(|&(lo, hi)| ((u16::from(lo) + u16::from(hi)) / 2) as u8)
        .collect();
    assert_ne!(grand_midpoints, midpoints);
    assert_eq!(grand_midpoints[0], 4);
    assert_eq!(grand_midpoints[15], 123);

    // The upright's two.
    let upright = LibrarySpec::find("vcsl-knight-upright").unwrap();
    assert_eq!(upright.bands.bands(), vec![(1, 63), (64, 127)]);
}

/// The layouts, and the tie-break the whole-tone library forced.
#[test]
fn the_layouts_are_the_recorded_key_sets_the_policy_scores() {
    let grand = LibrarySpec::find("bitklavier-piano-bar").unwrap();
    let salamander = LibrarySpec::find("salamander").unwrap();
    assert_eq!(
        grand.layout.keys(),
        salamander.layout.keys(),
        "bitKlavier deliberately records the same thirty keys Salamander does, \
         which is why its preset is scoreable against exactly the ladder the \
         evaluation policy already uses"
    );

    let upright = LibrarySpec::find("vcsl-knight-upright").unwrap();
    let keys = upright.layout.keys();
    assert_eq!(keys.len(), 45);
    assert_eq!(keys[0], 21);
    assert_eq!(keys[keys.len() - 1], 108);
    // Whole tones tie at every other key, and the tie goes down — which is
    // what VCSL's own generated map does.
    let spans = upright.layout.spans();
    assert_eq!(spans[&21], (21, 22));
    assert_eq!(spans[&23], (23, 24));
    // Half of Salamander's reroute distance: +-1 semitone rather than +-2.
    let worst = spans
        .iter()
        .map(|(&key, &(lo, hi))| lo.abs_diff(key).max(hi.abs_diff(key)))
        .max()
        .unwrap();
    assert_eq!(worst, 1);
    let salamander_worst = Layout::Interval { lo: 21, step: 3, extra: &[] }
        .spans()
        .iter()
        .map(|(&key, &(lo, hi))| lo.abs_diff(key).max(hi.abs_diff(key)))
        .max()
        .unwrap();
    assert_eq!(salamander_worst, 1);
}

/// A preset built from a library that was resampled says so. The rate a
/// measurement passed through is part of that measurement.
#[test]
fn a_resampled_library_declares_it() {
    for spec in LibrarySpec::all() {
        if spec.is_native_rate() {
            continue;
        }
        assert!(
            spec.caveats
                .iter()
                .any(|c| c.contains("RESAMPLED") || c.contains("resampled")),
            "{}: resampled and does not say so in its caveats",
            spec.id
        );
        let root = repo().join("data").join(spec.id);
        if !root.is_dir() {
            continue;
        }
        let sfz = root.join(format!("{}.sfz", spec.id));
        if !sfz.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&sfz).unwrap();
        assert!(
            text.contains("resampled once, offline"),
            "{}: the generated map does not carry the rate note",
            spec.id
        );
        // And the files it names really are at the engine's rate.
        let library = SampleLibrary::from_sfz(&sfz).unwrap();
        let first = library.samples().next().expect("a recording");
        let audio = piano_tuner::audio::load(&first.path).unwrap();
        assert_eq!(audio.sample_rate, piano_tuner::SAMPLE_RATE);
    }
}

/// Anything under `data/` that a preset was estimated from has a fetch script
/// carrying its licence, and the script is checked in even though the data is
/// not. This is the standing rule of `ATTRIBUTION.md`, tested rather than
/// remembered.
#[test]
fn every_described_library_has_a_checked_in_fetch_script_carrying_its_licence() {
    for spec in LibrarySpec::all() {
        let script = repo().join("data").join(match spec.id {
            "salamander" => "fetch_salamander.sh".to_string(),
            "bitklavier-piano-bar" => "fetch_bitklavier.sh".to_string(),
            id => format!("fetch_{}.sh", id.replace('-', "_")),
        });
        assert!(
            script.is_file(),
            "{}: no fetch script at {}",
            spec.id,
            script.display()
        );
        let text = std::fs::read_to_string(&script).unwrap();
        // The licence name and its URL, both, in the script itself.
        let (name, url) = spec
            .licence
            .split_once(" — ")
            .unwrap_or((spec.licence, spec.licence));
        assert!(
            mentions(&text, name),
            "{}: the fetch script does not name {name}",
            spec.id
        );
        assert!(
            mentions(&text, url.trim().trim_start_matches("https").trim_start_matches("http")),
            "{}: the fetch script does not carry the licence URL",
            spec.id
        );
        assert!(
            text.contains("sha256") || text.contains("sha1"),
            "{}: the fetch script pins no checksum",
            spec.id
        );
    }
}

/// And the same rule one level up: ATTRIBUTION.md names every library a
/// shipped preset was estimated from.
#[test]
fn attribution_names_every_library_a_preset_ships_from() {
    let attribution = std::fs::read_to_string(repo().join("ATTRIBUTION.md")).unwrap();
    for spec in LibrarySpec::all() {
        let (name, url) = spec
            .licence
            .split_once(" — ")
            .unwrap_or((spec.licence, spec.licence));
        assert!(
            mentions(&attribution, name),
            "ATTRIBUTION.md does not record {}'s licence ({name})",
            spec.id
        );
        assert!(
            mentions(&attribution, url.trim()),
            "ATTRIBUTION.md does not carry {}'s licence URL",
            spec.id
        );
        assert!(
            attribution.contains(spec.source_url),
            "ATTRIBUTION.md does not carry {}'s source",
            spec.id
        );
    }
}

/// A preset file names the library it was estimated from, in a field a reader
/// of the file can find. Runs over whatever presets exist, so a new one that
/// forgets its provenance fails here.
#[test]
fn every_measured_preset_names_its_library() {
    let dir = repo().join("presets");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // The hand-tuned v1 instrument was authored, not measured.
        if name == "default.toml" {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("description"),
            "{name}: no description field"
        );
        let names_a_library = LibrarySpec::all().iter().any(|spec| {
            let (licence, _) = spec
                .licence
                .split_once(" — ")
                .unwrap_or((spec.licence, spec.licence));
            mentions(&text, licence)
        });
        assert!(
            names_a_library,
            "{name}: names no library licence; a measured preset carries its \
             provenance in the file, not only in ATTRIBUTION.md"
        );
    }
}

/// **The falsification for the false-beat bank guard** (`DECISIONS.md` 522).
///
/// A `notes.false_beat` row names a partial and asks the engine to modulate
/// it; the schema rejects a row for a partial the key's string bank does not
/// have, and it is right to, because there is nothing there to modulate. The
/// recording is the authority on *which* partials beat, and at the top of the
/// compass it resolves partials the bank does not hold — so stage 1 of `fit`
/// has to intersect the two, and before this guard it did not.
///
/// This could not happen on a minor-third library. It happened on the first
/// whole-tone one, at key 108, as a **panic** inside `solve_on_the_render`'s
/// first render rather than as a quietly wrong number — which is the good
/// failure and the reason it is worth a test rather than a silent fix.
///
/// On the code before the guard this test fails by `Preset::from_toml`
/// returning `Invalid("notes.false_beat[87][2].k is 5, but key 108 has
/// partials 1..=4")`; after it, every row of every measured preset names a
/// partial its own key really has.
#[test]
fn no_measured_preset_asks_the_engine_to_beat_a_partial_it_does_not_have() {
    for name in ["concert-grand-d.toml", "upright-parlour.toml"] {
        let path = repo().join("presets").join(name);
        assert!(
            path.is_file(),
            "{name} is a shipped preset and is missing from presets/"
        );
        let text = std::fs::read_to_string(&path).unwrap();
        let engine = piano_emulator::preset::Preset::from_toml(&text)
            .unwrap_or_else(|e| panic!("{name} does not load into the engine: {e:?}"));
        for key in 21..=108u8 {
            let bank = engine.string_params(key).partial_count() as u32;
            let index = usize::from(key) - 21;
            let rows = engine
                .notes
                .false_beat
                .get(index)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            for row in rows {
                assert!(
                    u32::from(row.k) >= 1 && u32::from(row.k) <= bank,
                    "{name}: key {key} carries a false-beat row for partial {} \
                     and its bank has {bank}",
                    row.k
                );
            }
        }
    }
}

/// `CC-BY 3.0`, `CC BY 3.0` and `cc by3.0` are one licence. A provenance test
/// that failed on a hyphen would be a spelling test, and the rule it is here
/// to enforce is about the licence being *named*.
fn mentions(haystack: &str, needle: &str) -> bool {
    fn flatten(text: &str) -> String {
        text.chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
            .map(|c| c.to_ascii_lowercase())
            .collect()
    }
    flatten(haystack).contains(&flatten(needle))
}

#[allow(dead_code)]
fn exists(path: &Path) -> bool {
    path.exists()
}
