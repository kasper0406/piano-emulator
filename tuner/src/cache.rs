//! Content-addressed disk caches for the offline tooling.
//!
//! Two of the batch subcommands spend most of their time re-deriving something
//! that did not change. `compass` and `bench` both render the *reference* — the Salamander recordings played by
//! [`crate::sampler`] — beside the engine's own render, and the reference side
//! only moves when the sampler's code, the SFZ, or the material asked of it
//! moves. Re-rendering it on every iteration of the engine is pure waste.
//!
//! # The discipline
//!
//! A cache that can be wrong is worse than no cache, because it makes a
//! measurement lie. So nothing here is invalidated by a timestamp or by a
//! `--refresh` flag a tired person forgets to pass: **the key is the whole of
//! the input**, hashed, and a changed input simply misses. A cache entry is
//! therefore never updated in place and never stale — it is either the answer
//! to exactly this question or it is not read at all.
//!
//! What has to go into a key is everything the render is a function of:
//!
//! * a **version constant** for the code that produces it, bumped by hand when
//!   that code changes in a way the other inputs cannot see —
//!   [`crate::sampler::SAMPLER_VERSION`] for the reference renders. The bump is
//!   the one manual step in the scheme and the comment on the constant says so;
//! * the **data** it reads, hashed by content: the SFZ file's bytes. (The FLAC
//!   payload beside it is pinned by `data/fetch_salamander.sh`, which is
//!   checksummed, so the SFZ text identifies the library.)
//! * the **material**: which phrase set version, or which key and velocity, and
//!   for how long, at what sample rate.
//!
//! # The format
//!
//! Cached audio is a 32-bit float WAV, which the crate already reads and writes
//! and which is lossless: the bytes that come back out of
//! [`crate::audio::load_wav`] are the same `f32` values that went into
//! [`Audio::write_wav`], so a cache hit is bit-identical to a fresh render
//! rather than merely close. `tests/reference_cache.rs` renders one phrase both
//! ways and asserts exactly that.
//!
//! Writes go to a temporary name in the same directory and are renamed into
//! place, so a run interrupted mid-write leaves no half-file for the next run
//! to read, and two processes racing on the same key both produce the same
//! bytes anyway.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::audio::{load_wav, Audio};
use crate::error::Result;

/// FNV-1a over 128 bits: the offset basis and prime of the reference
/// specification.
///
/// A hash here only has to make two different inputs land on two different
/// names — there is no adversary, and a miss costs a re-render rather than a
/// wrong answer. 128 bits makes an accidental collision across the few thousand
/// entries a working tree ever holds impossible in practice, and FNV is four
/// lines rather than a dependency.
const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// The accumulated identity of everything a cached artefact depends on.
///
/// Order matters and lengths are written for anything variable, so that
/// `("ab", "c")` and `("a", "bc")` cannot hash alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    state: u128,
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::new()
    }
}

impl Fingerprint {
    pub fn new() -> Self {
        Fingerprint { state: FNV_OFFSET }
    }

    pub fn bytes(&mut self, data: &[u8]) -> &mut Self {
        // The length first: a self-delimiting field cannot be confused with its
        // neighbour.
        let len = data.len() as u64;
        for &b in &len.to_le_bytes() {
            self.state = (self.state ^ u128::from(b)).wrapping_mul(FNV_PRIME);
        }
        for &b in data {
            self.state = (self.state ^ u128::from(b)).wrapping_mul(FNV_PRIME);
        }
        self
    }

    pub fn str(&mut self, value: &str) -> &mut Self {
        self.bytes(value.as_bytes())
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.bytes(&value.to_le_bytes())
    }

    /// The bit pattern, not the printed value: two floats that print alike and
    /// are not equal must not share a cache entry.
    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.bytes(&value.to_bits().to_le_bytes())
    }

    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.bytes(&value.to_bits().to_le_bytes())
    }

    /// The contents of a file, by value.
    pub fn file(&mut self, path: impl AsRef<Path>) -> Result<&mut Self> {
        let data = std::fs::read(path.as_ref())?;
        Ok(self.bytes(&data))
    }

    /// Every sample of a signal, by value — the way an engine's sounding path
    /// is fingerprinted by what it sounds like rather than by what it is
    /// built from.
    pub fn samples(&mut self, signal: &[f32]) -> &mut Self {
        self.u64(signal.len() as u64);
        for &v in signal {
            self.state = (self.state ^ u128::from(v.to_bits())).wrapping_mul(FNV_PRIME);
        }
        self
    }

    /// 32 lower-case hex digits, for use in a file name.
    pub fn hex(&self) -> String {
        format!("{:032x}", self.state)
    }

    pub fn value(&self) -> u128 {
        self.state
    }
}

/// A cached stereo (or mono) render: the file at `path` if it is readable, and
/// otherwise whatever `render` produces, written there for next time.
///
/// A failure to *read* is a miss, never an error — a truncated file from an
/// interrupted run, or a format the loader no longer understands, should cost a
/// re-render and nothing else. A failure to *write* is likewise swallowed: the
/// caller asked for a render and got one, and a read-only or full disk is not a
/// reason to fail a measurement.
pub fn audio<F>(path: &Path, render: F) -> Result<Audio>
where
    F: FnOnce() -> Result<Audio>,
{
    if let Ok(hit) = load_wav(path) {
        return Ok(hit);
    }
    let fresh = render()?;
    store(path, |temporary| fresh.write_wav(temporary));
    Ok(fresh)
}

/// Something that can be written to a cache entry and read back **as the same
/// value, bit for bit**.
///
/// Deliberately not `serde`. The one thing this cache holds besides audio is
/// the self-calibration gate's corpus of tracked trajectories, and there the
/// expensive step is not the render — measured on this machine, rendering A1
/// for 26 s costs 0.32 s and tracking its eighty partials costs 3.2 s. A JSON
/// corpus was tried first and is not worth having: at 4.9 MB of pretty-printed
/// numbers per note, `serde_json`'s exact (`float_roundtrip`) parser costs about
/// as much to read an entry as the tracker costs to recompute it, and the second
/// run of the gate came back 36.1 s against the first run's 38.3 s. A float
/// written as its bit pattern is exact by construction and reads at the speed of
/// a `memcpy`, so that is what an implementation of this trait writes.
///
/// `decode` returns `None` for anything it does not recognise — a truncated
/// file, an older layout — because a cache miss is always a legal answer.
pub trait Cacheable: Sized {
    fn encode(&self) -> Vec<u8>;
    fn decode(bytes: &[u8]) -> Option<Self>;
}

/// A cached value: the entry at `path` if it decodes, and otherwise whatever
/// `produce` returns, written there for next time.
pub fn stored<T, F>(path: &Path, produce: F) -> Result<T>
where
    T: Cacheable,
    F: FnOnce() -> Result<T>,
{
    if let Some(hit) = std::fs::read(path).ok().and_then(|b| T::decode(&b)) {
        return Ok(hit);
    }
    let fresh = produce()?;
    let encoded = fresh.encode();
    store(path, |temporary| Ok(std::fs::write(temporary, &encoded)?));
    Ok(fresh)
}

/// Little-endian readers for the hand-rolled encodings [`Cacheable`] asks for.
/// They advance the slice and give `None` at its end, so a decoder is a chain of
/// `?`s and a truncated file is a miss rather than a panic.
pub mod read {
    pub fn u8(bytes: &mut &[u8]) -> Option<u8> {
        let (head, tail) = bytes.split_first()?;
        *bytes = tail;
        Some(*head)
    }

    pub fn u32(bytes: &mut &[u8]) -> Option<u32> {
        let (head, tail) = bytes.split_at_checked(4)?;
        *bytes = tail;
        Some(u32::from_le_bytes(head.try_into().ok()?))
    }

    pub fn u64(bytes: &mut &[u8]) -> Option<u64> {
        let (head, tail) = bytes.split_at_checked(8)?;
        *bytes = tail;
        Some(u64::from_le_bytes(head.try_into().ok()?))
    }

    /// The bit pattern, so the value that comes back is the value that went in
    /// — including the sign of a zero and the payload of a NaN.
    pub fn f64(bytes: &mut &[u8]) -> Option<f64> {
        Some(f64::from_bits(u64(bytes)?))
    }

    pub fn string(bytes: &mut &[u8]) -> Option<String> {
        let len = usize::try_from(u64(bytes)?).ok()?;
        let (head, tail) = bytes.split_at_checked(len)?;
        *bytes = tail;
        String::from_utf8(head.to_vec()).ok()
    }

    /// `usize` from a length field, refusing anything a real file could not
    /// hold: a corrupt count must not make a decoder reserve a terabyte.
    pub fn len(bytes: &mut &[u8]) -> Option<usize> {
        let n = usize::try_from(u64(bytes)?).ok()?;
        (n <= bytes.len()).then_some(n)
    }
}

/// The matching writers.
pub mod write {
    pub fn f64(out: &mut Vec<u8>, value: f64) {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    pub fn u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn string(out: &mut Vec<u8>, value: &str) {
        u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }
}

/// Writes through a temporary name in the same directory, so that a reader
/// never sees a partial file and two processes racing on one key cannot
/// interleave. A failure to write is swallowed: the caller already has the
/// value it asked for, and a full or read-only disk is not a reason to fail a
/// measurement.
fn store(path: &Path, write: impl FnOnce(&Path) -> Result<()>) {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let temporary = dir.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("entry"),
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    if write(&temporary).is_ok() && std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
}

/// `<root>/cache/reference` for a data root of `<root>/salamander` — the
/// directory `README.md` documents the reference caches into.
pub fn reference_dir(data: &Path) -> PathBuf {
    data.parent()
        .unwrap_or_else(|| Path::new("data"))
        .join("cache")
        .join("reference")
}

/// `<repo>/data/cache/calibration` — the self-calibration gate's corpus of
/// tracked notes. Under `data/`, which is gitignored, like every other cache
/// here.
pub fn calibration_dir(repo: &Path) -> PathBuf {
    repo.join("data").join("cache").join("calibration")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fields_of_a_key_cannot_be_confused_with_each_other() {
        let mut a = Fingerprint::new();
        a.str("ab").str("c");
        let mut b = Fingerprint::new();
        b.str("a").str("bc");
        assert_ne!(a.value(), b.value());
    }

    #[test]
    fn floats_that_print_alike_hash_apart() {
        let mut a = Fingerprint::new();
        a.f64(0.0);
        let mut b = Fingerprint::new();
        b.f64(-0.0);
        assert_ne!(a.value(), b.value(), "0.0 and -0.0 are different renders");
    }

    #[test]
    fn a_fingerprint_is_a_function_of_its_input_alone() {
        let build = || {
            let mut f = Fingerprint::new();
            f.str("compass").u64(60).f64(3.6).samples(&[0.5, -0.25]);
            f.hex()
        };
        assert_eq!(build(), build());
        assert_eq!(build().len(), 32);
    }
}
