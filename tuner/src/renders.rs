//! Content-keyed disk cache for **engine** renders (`data/cache/engine`).
//!
//! [`crate::cache`] holds the *reference* side — the Salamander recordings
//! played by [`crate::sampler`] — because that side does not move when the
//! engine does. This module is the other half, and it is the one the fits
//! spend their wall time in: every stage-2 search evaluates a candidate preset
//! by **rendering the instrument**, thirty recorded keys and six phrases at a
//! time, and a search that revisits a point re-renders it. `DECISIONS.md` 392
//! made the batch parallel and left the arithmetic alone; this makes the
//! arithmetic not happen twice.
//!
//! # The discipline is [`crate::cache`]'s, unchanged
//!
//! **The key is the whole of the input**, so a changed input misses rather than
//! returning something stale. There is no refresh flag and no timestamp. What
//! an engine render is a function of is exactly three things, and all three are
//! in the key:
//!
//! * **the preset**, as its own TOML bytes — `Preset::to_toml`, which is the
//!   only description of a preset the engine and the tuner agree on, and the
//!   same one `tuner/tests/calibration.rs` keys its corpus with. Any field of
//!   any section, anywhere, moves it;
//! * **the engine**, fingerprinted by *what it sounds like* rather than by a
//!   version number nobody remembers to bump — [`engine_fingerprint`] renders
//!   two short probes through this crate's own render path and hashes their
//!   samples. One probe is the pan-pot path and one carries a full
//!   `[voicing.mics]` with a mode-controlled band, so a change to **either**
//!   branch of `soundboard` misses the whole cache by construction. This is
//!   `tuner/tests/calibration.rs`'s `corpus_base` applied to audio instead of
//!   trajectories;
//! * **the material**: which note or phrase, at what velocity, for how long,
//!   with how much silence in front of it — [`NoteSpec`] and [`PhraseSpec`].
//!
//! A cache that can be wrong is worse than no cache, so nothing here is a
//! projection of the preset: the *whole* TOML is hashed even though most of a
//! preset cannot reach most of a render. A per-key slice was considered and
//! refused, and the reason is measurable rather than cautious — `Engine::new`
//! builds a `ResonanceBus` from the whole preset and the top octave's strings
//! have **no dampers**, so a single note's render is not a function of that
//! note's own entries alone.
//!
//! # What that leaves for an incremental fit
//!
//! Exactly what a content-keyed cache gives for free, which turns out to be the
//! useful half: a search round that moves *some* of its parameters and leaves
//! the rest re-renders only what moved. Two shapes of loop benefit and they are
//! the two the factory has.
//!
//! * **A candidate revisited.** Every compass round starts by scoring its own
//!   incumbent, the simplex re-evaluates its best point, the constrained pass
//!   starts where the relaxed pass stopped, and the report at the end scores the
//!   fitted preset again. On `piano-tuner mics --stage band` those repeats are a
//!   fifth of the renders, and they now cost a file read.
//! * **A key whose preset did not move.** A per-key bisection that builds one
//!   preset per step — base plus the one key it is moving — leaves every other
//!   key's preset bytes untouched, so every other key's render is a hit on the
//!   *next* round. That is the incremental refit `DECISIONS.md` 392 asked for,
//!   and it needs no per-key reasoning about what reaches what: the key already
//!   contains the whole preset, so it is right whether or not the coupling
//!   exists.
//!
//! # The format, and the proof it is the same render
//!
//! A 32-bit float WAV, as [`crate::cache::audio`] writes: the bytes that come
//! back are the `f32` values that went in, so a hit is **bit-identical** to a
//! fresh render rather than close to one. `tests/engine_cache.rs` renders one
//! note both ways and asserts exactly that, and asserts that touching any of the
//! three inputs misses.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use piano_emulator::preset::{MicVoicing, ModalBand, Preset};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, BLOCK};

use crate::audio::Audio;
use crate::cache::{self, Fingerprint};
use crate::SAMPLE_RATE;

/// Bump when the *layout* of a key changes — i.e. when this module starts
/// hashing a different set of fields. Everything an entry depends on is hashed
/// rather than declared, so this is the only hand-maintained number and it does
/// not move when the engine does.
const ENGINE_CACHE_VERSION: u32 = 1;

/// The probe [`engine_fingerprint`] renders the engine's own sounding path
/// with: mid-compass, three-strung, struck hard enough to open the hammer's
/// nonlinearity, and long enough to include the board's field but not so long
/// that every process pays for a second of DSP.
const PROBE: (u8, u16, f32) = (60, 90, 0.35);

/// One note, as the stereo boards render one: a strike at `preroll` samples
/// into a buffer of `preroll + duration`, with the preroll then dropped.
///
/// The preroll is part of the key because it is part of the render: an event
/// takes effect at the head of the block that contains it, so two prerolls that
/// are not both whole blocks put the strike at two different samples of the
/// window (`DECISIONS.md` 378).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteSpec {
    pub key: u8,
    pub velocity: u8,
    /// Seconds of note kept, after the preroll is dropped.
    pub duration_s: f64,
    /// Silence before the strike, in samples, dropped from what is returned.
    pub preroll: usize,
}

impl NoteSpec {
    pub fn new(key: u8, velocity: u8, duration_s: f64, preroll: usize) -> Self {
        Self {
            key,
            velocity,
            duration_s,
            preroll,
        }
    }

    fn preroll_s(&self) -> f64 {
        self.preroll as f64 / f64::from(SAMPLE_RATE)
    }
}

/// One phrase of the scoreboard's set, or any other named event list.
///
/// The events are hashed by value, so two phrases that share a name and differ
/// by one note are two entries.
pub struct PhraseSpec<'a> {
    pub name: &'a str,
    pub events: &'a [RenderEvent],
    pub duration_s: f64,
}

/// How many renders were answered from disk and how many were computed.
///
/// Global rather than per-instance because the interesting number is per
/// *process* — "this invocation of the fit rendered 412 sets and read 96" — and
/// because the fits build a cache handle per surface.
static HITS: AtomicUsize = AtomicUsize::new(0);
static MISSES: AtomicUsize = AtomicUsize::new(0);

/// `(hits, misses)` since the process started.
pub fn stats() -> (usize, usize) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// How much disk the engine cache may hold, in bytes.
///
/// Thirty-two gigabytes by default, which is about one coarse grid and two
/// searches, and `ENGINE_CACHE_MB` overrides it. It is a *budget* and not an
/// invalidation: see [`EngineRenders::prune`].
fn budget_bytes() -> u64 {
    std::env::var("ENGINE_CACHE_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32_768)
        .saturating_mul(1024 * 1024)
}

/// How many misses between two checks of [`budget_bytes`]. Two thousand entries
/// is under four gigabytes of overshoot and one directory scan per couple of
/// minutes of fitting.
const PRUNE_EVERY: usize = 2_000;

/// A line for a tool to print: how much of this invocation was already on disk.
pub fn stats_line() -> String {
    let (hits, misses) = stats();
    let total = hits + misses;
    format!(
        "engine renders: {total} asked for, {hits} from the cache ({:.0} %), {misses} rendered",
        if total == 0 {
            0.0
        } else {
            100.0 * hits as f64 / total as f64
        }
    )
}

/// **What the engine sounds like**, as a hash: the identity of every line of
/// code between a preset and a pair of channels.
///
/// Two probes, and both are needed. The first is the **pan-pot** path — no
/// `[voicing.mics]` at all, which is `DECISIONS.md` 103's neutrality contract
/// and a branch `Soundboard::with_mics` takes by itself. The second carries a
/// full microphone section *including* a mode-controlled band, so that the
/// stage this milestone rewrites is inside the fingerprint: a change to the
/// lobe with the pan-pot probe alone would leave every cached render of every
/// mic'd preset readable and wrong.
///
/// Computed once per process and memoised. It costs two renders of a third of a
/// second, which is under 20 ms on this machine and is paid whether or not
/// anything hits.
pub fn engine_fingerprint() -> Fingerprint {
    static BASE: OnceLock<Fingerprint> = OnceLock::new();
    *BASE.get_or_init(|| {
        let (key, vel, duration_s) = PROBE;
        let mut print = Fingerprint::new();
        print
            .str("cache/engine/probe")
            .u64(u64::from(ENGINE_CACHE_VERSION))
            .u64(u64::from(SAMPLE_RATE))
            .u64(BLOCK as u64);
        for mics in [None, Some(probe_mics())] {
            let mut preset = Preset::default();
            preset.voicing.mics = mics;
            let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel })];
            let (left, right) = render_to_buffer(&preset, &events, duration_s);
            print.samples(&left).samples(&right);
        }
        print
    })
}

/// The microphone section the second probe is rendered through. Not the shipped
/// one and deliberately not fitted: it exists to touch every branch of
/// `soundboard::Mics`, so it carries a band as well as a geometry.
fn probe_mics() -> MicVoicing {
    MicVoicing {
        spacing_m: 0.13,
        height_m: 0.12,
        span_m: 1.1,
        width: 1.3,
        diffuse_coherence: 2.0,
        // The lift is **0.99 and not 2.0** since item 418 railed it at one: a
        // fingerprint probe is a preset, and a preset the schema refuses is not
        // a description of what this engine renders. Changing it misses every
        // entry written before it, which is the same cost as bumping
        // `ENGINE_CACHE_VERSION` and is what the version is for — an entry is
        // the answer to exactly this question or it is not read.
        modal: Some(ModalBand {
            lo_hz: 220.0,
            hi_hz: 300.0,
            lift: 0.99,
        }),
    }
}

/// The cache, rooted at a directory.
#[derive(Clone, Debug)]
pub struct EngineRenders {
    dir: Option<PathBuf>,
}

impl EngineRenders {
    /// The cache under a data root — `<root>/cache/engine` for a library at
    /// `<root>/salamander`, beside the reference renders.
    pub fn at_data_root(data: &Path) -> Self {
        let handle = Self {
            dir: Some(cache::engine_dir(data)),
        };
        handle.prune();
        handle
    }

    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: Some(dir.into()),
        }
    }

    /// Keeps the directory under [`budget_bytes`], oldest first.
    ///
    /// **Eviction is always sound here and that is the point of the design.**
    /// An entry is a hit only if the whole of its input hashed to its name, so
    /// deleting one can cost a re-render and can never return a wrong answer —
    /// which is why a size cap is allowed where a `--refresh` flag is not. It
    /// is needed because these entries are large: three seconds of stereo
    /// `f32` is 1.9 MB, a coarse grid of 81 candidates over 36 items is 2916 of
    /// them, and one afternoon of fitting would otherwise fill a disk.
    ///
    /// Run once when a handle is opened, on the modification time, so the
    /// entries a run is *about* to ask for again are the ones that survive.
    fn prune(&self) {
        let Some(dir) = &self.dir else { return };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                if !meta.is_file() {
                    return None;
                }
                Some((meta.modified().ok()?, meta.len(), e.path()))
            })
            .collect();
        let budget = budget_bytes();
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        if total <= budget {
            return;
        }
        files.sort_by_key(|f| f.0);
        for (_, size, path) in files {
            if total <= budget {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(size);
            }
        }
    }

    /// A handle that renders and never reads or writes: the control the tests
    /// compare against, and what a caller with no writable data root gets.
    pub fn off() -> Self {
        Self { dir: None }
    }

    /// One note, from disk if this exact preset has been through this exact
    /// engine on this exact note before.
    pub fn note(&self, preset: &Preset, spec: NoteSpec) -> Audio {
        let mut print = self.base(preset);
        print
            .str("note")
            .u64(u64::from(spec.key))
            .u64(u64::from(spec.velocity))
            .f64(spec.duration_s)
            .u64(spec.preroll as u64);
        let name = format!(
            "note-k{:03}-v{:03}-{}.wav",
            spec.key,
            spec.velocity,
            print.hex()
        );
        self.fetch(&name, || render_note(preset, spec))
    }

    /// One phrase, or any other named list of events, rendered from zero.
    pub fn phrase(&self, preset: &Preset, spec: &PhraseSpec<'_>) -> Audio {
        let mut print = self.base(preset);
        print
            .str("phrase")
            .str(spec.name)
            .f64(spec.duration_s)
            .u64(spec.events.len() as u64);
        for event in spec.events {
            // The frame rather than the time, because the frame is what the
            // renderer acts on, and the `Debug` form of the event, which is
            // total and stable over its fields.
            print
                .u64(event.frame() as u64)
                .str(&format!("{:?}", event.event));
        }
        let name = format!("phrase-{}-{}.wav", sanitise(spec.name), print.hex());
        self.fetch(&name, || {
            let (left, right) = render_to_buffer(preset, spec.events, spec.duration_s as f32);
            Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
        })
    }

    /// Everything a render is a function of except the material.
    fn base(&self, preset: &Preset) -> Fingerprint {
        let mut print = engine_fingerprint();
        print.str("cache/engine/render").str(&preset.to_toml());
        print
    }

    fn fetch(&self, name: &str, render: impl FnOnce() -> Audio) -> Audio {
        let Some(dir) = &self.dir else {
            MISSES.fetch_add(1, Ordering::Relaxed);
            return render();
        };
        let path = dir.join(name);
        // Counting is done here rather than inside `cache::audio` so that the
        // number a tool prints is about *engine* renders and not about every
        // cached artefact in the process.
        if let Ok(hit) = crate::audio::load_wav(&path) {
            HITS.fetch_add(1, Ordering::Relaxed);
            return hit;
        }
        MISSES.fetch_add(1, Ordering::Relaxed);
        let fresh = cache::audio(&path, || Ok(render())).expect("an engine render cannot fail");
        // A long fit writes tens of thousands of these, so the budget cannot be
        // checked only when a handle opens: one `--stage band` run wrote 76 GB
        // before this line existed. Every `PRUNE_EVERY` misses is often enough
        // to bound the directory and rare enough that the directory scan is
        // noise against a render.
        if MISSES.load(Ordering::Relaxed) % PRUNE_EVERY == 0 {
            self.prune();
        }
        fresh
    }
}

/// The render itself, with no cache anywhere near it: what a hit has to equal.
pub fn render_note(preset: &Preset, spec: NoteSpec) -> Audio {
    let events = [RenderEvent::new(
        spec.preroll_s() as f32,
        Event::NoteOn {
            key: spec.key,
            vel: u16::from(spec.velocity),
        },
    )];
    debug_assert_eq!(
        events[0].frame(),
        spec.preroll,
        "the strike must land on the first sample of the window"
    );
    let (left, right) = render_to_buffer(
        preset,
        &events,
        (spec.preroll_s() + spec.duration_s) as f32,
    );
    Audio::new(
        SAMPLE_RATE,
        vec![
            left[spec.preroll..].to_vec(),
            right[spec.preroll..].to_vec(),
        ],
    )
    .expect("the engine renders stereo")
}

/// A phrase name as a file name: the set's names are already tame, but a cache
/// entry must not be able to escape its directory.
fn sanitise(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}
