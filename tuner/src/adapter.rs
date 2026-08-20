//! The library-description layer: what a sample library *is*, as data.
//!
//! [`library`](crate::library) reads an SFZ and [`sampler`](crate::sampler)
//! plays one, and both are total on the opcodes they honour — neither has ever
//! contained the word "Salamander". What *was* Salamander-shaped is everything
//! upstream of them: the assumption that a library ships an SFZ at all, that
//! its keys are minor thirds, that it has sixteen velocity layers, that it is
//! already at 48 kHz, and that its key-off thumps and pedal actions are
//! indexed the way that one file happens to index them.
//!
//! This module is where those five facts live, one [`LibrarySpec`] per
//! library, and it answers them in the only place they can be answered without
//! touching a render: **it writes the SFZ**. A library that ships no
//! instrument definition gets one generated over its own tree; a library at
//! 44.1 kHz gets its recordings resampled once, offline, by the crate's own
//! band-limited sinc resampler, so that the boundary resampler is not inside
//! every subsequent measurement of that preset. Downstream of the generated
//! file nothing knows the difference, and nothing had to change.
//!
//! ### Why Salamander is here and is not adapted
//!
//! Salamander is the first instance, and its [`LibrarySpec`] describes a
//! library that **ships its own SFZ** ([`Source::Shipped`]). Nothing generates
//! anything for it and its render path is byte for byte the one it always had
//! — which is the whole point, since every board in this repository is barred
//! against renders made through it.
//!
//! What its spec is *for* is falsification. `adapter::tests::
//! the_salamander_description_agrees_with_the_shipped_sfz` reconstructs the
//! layout, the sixteen velocity bands, the mechanism census and the key spans
//! from the description alone and asserts they are what the shipped file says
//! — so a description layer that would have got Salamander wrong cannot claim
//! to have got a library nobody can check right. It also pins the one place
//! the shipped file is irregular: **C4's regions carry no `pitch_keycenter`**
//! and are a recorded key only through the midpoint of `lokey=59 hikey=61`.
//!
//! ### The five facts, and where each is used
//!
//! 1. **[`Layout`]** — which keys are *genuinely recorded*. The evaluation
//!    policy (`SHIPPING.md`) fits and scores against these and no others, and
//!    every drawn key is interpolated between them.
//! 2. **[`Bands`]** — how many velocity layers and where their bands sit.
//!    [`Sample::midi_velocity`](crate::library::Sample::midi_velocity) is the
//!    band midpoint and is the abscissa of every velocity fit, so choosing the
//!    bands *is* choosing what the hammer curve is fitted against
//!    (`DECISIONS.md` 519).
//! 3. **The rate** — [`LibrarySpec::published_rate_hz`] against
//!    [`LibrarySpec::delivered_rate_hz`]. Equal means the recordings reach the
//!    estimators untouched; unequal means [`resample_tree`] has been over them
//!    once and that is a stated systematic of that preset.
//! 4. **The key map** — [`FilePattern`], how a file names the key and layer it
//!    is a recording of, including libraries whose note names sit an octave
//!    below the standard spelling.
//! 5. **The mechanism** — [`MechanismFiles`], the key-off thumps, the pedal
//!    tray and the pitched release resonances, which are the whole input of
//!    the `noise` stage and half the input of `halo`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::library::MechanismKind;

/// Which keys of the compass a library actually recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Every key from `lo` to `hi`. The Iowa MIS piano is the only free
    /// candidate shaped this way.
    Chromatic { lo: u8, hi: u8 },
    /// Every `step` semitones from `lo`, and `extra` keys on top of that
    /// series — Salamander and bitKlavier are `step: 3` from 21, VCSL's Knight
    /// is `step: 2` from 21 with C8 (108) bolted on the end because a
    /// whole-tone series from A0 misses it by one.
    Interval { lo: u8, step: u8, extra: &'static [u8] },
    /// A list, for a library whose recorded set is not a series at all.
    Explicit(&'static [u8]),
}

impl Layout {
    /// The recorded keys, ascending, within the 88-key compass.
    pub fn keys(&self) -> Vec<u8> {
        let mut keys: Vec<u8> = match self {
            Layout::Chromatic { lo, hi } => (*lo..=*hi).collect(),
            Layout::Interval { lo, step, extra } => {
                let mut keys: Vec<u8> = (*lo..=108).step_by(usize::from(*step)).collect();
                keys.extend_from_slice(extra);
                keys
            }
            Layout::Explicit(keys) => keys.to_vec(),
        };
        keys.retain(|k| (21..=108).contains(k));
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// `lokey`/`hikey` for each recorded key: every key of the compass goes to
    /// the nearest recorded one, and a key equidistant from two goes to the
    /// **lower**.
    ///
    /// That tie-break is not a taste. It is what both shipped files this
    /// module is checked against already do — Salamander's minor thirds never
    /// tie, and VCSL's whole tones tie at every other key and its own
    /// generated SFZ sends each of them down (`A-1` covers 21-22, `B-1`
    /// covers 23-24) — so a description that broke the tie the other way would
    /// disagree with the one library that can arbitrate it.
    pub fn spans(&self) -> BTreeMap<u8, (u8, u8)> {
        let keys = self.keys();
        let mut spans: BTreeMap<u8, (u8, u8)> = keys.iter().map(|&k| (k, (k, k))).collect();
        if keys.is_empty() {
            return spans;
        }
        for key in 21..=108u8 {
            let owner = keys
                .iter()
                .copied()
                .min_by_key(|&k| (k.abs_diff(key), k))
                .expect("keys is not empty");
            let span = spans.get_mut(&owner).expect("owner is a recorded key");
            span.0 = span.0.min(key);
            span.1 = span.1.max(key);
        }
        spans
    }

    /// How this recorded set is spaced, in the words a generated report puts
    /// after "the library" — *"records one key every minor third"*.
    ///
    /// It exists because the reports used to say Salamander's minor thirds
    /// whatever library they were generated against (`DECISIONS.md` 521): a
    /// board that states another library's layout is a board that cannot be
    /// read, and the layout is already described here, once.
    pub fn spacing_phrase(&self) -> String {
        match self {
            Layout::Chromatic { .. } => "records every key of the compass".to_string(),
            Layout::Interval { step, extra, .. } => {
                let series = if *step <= 1 {
                    "records every key of the compass".to_string()
                } else {
                    format!("records one key every {}", interval_name(*step))
                };
                match extra.len() {
                    0 => series,
                    1 => format!("{series}, and one key more on top of that series"),
                    n => format!("{series}, and {n} keys more on top of that series"),
                }
            }
            Layout::Explicit(keys) => format!(
                "records {} keys, chosen rather than evenly spaced",
                keys.len()
            ),
        }
    }
}

/// The name a musician gives a distance of `semitones`, for reports that
/// describe how a library's recorded keys are spaced.
///
/// Only the intervals a sample library is actually laid out on are named;
/// anything else says its own size, which is worse prose and better than a
/// wrong name.
pub fn interval_name(semitones: u8) -> String {
    match semitones {
        0 => "no distance at all".to_string(),
        1 => "semitone".to_string(),
        2 => "whole tone".to_string(),
        3 => "minor third".to_string(),
        4 => "major third".to_string(),
        5 => "perfect fourth".to_string(),
        6 => "tritone".to_string(),
        7 => "perfect fifth".to_string(),
        12 => "octave".to_string(),
        n => format!("{n} semitones"),
    }
}

/// How a library's velocity layers divide the controller's 1-127.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bands {
    /// The bands the library's own instrument definition declares — the only
    /// honest choice when there is one, because the recordings were balanced
    /// against it.
    Explicit(&'static [(u8, u8)]),
    /// `n` bands of equal width over 1-127, which is what a library that says
    /// its layers are "evenly distributed across the dynamic range" and ships
    /// no map is asking for.
    Even(usize),
}

impl Bands {
    pub fn count(&self) -> usize {
        match self {
            Bands::Explicit(bands) => bands.len(),
            Bands::Even(n) => *n,
        }
    }

    /// The bands, softest first.
    pub fn bands(&self) -> Vec<(u8, u8)> {
        match self {
            Bands::Explicit(bands) => bands.to_vec(),
            Bands::Even(n) => {
                let n = (*n).max(1);
                (0..n)
                    .map(|i| {
                        let lo = 1 + (127 * i / n) as u8;
                        let hi = (127 * (i + 1) / n) as u8;
                        (lo, hi.max(lo))
                    })
                    .collect()
            }
        }
    }
}

/// How a file names the key and the layer it is a recording of.
#[derive(Clone, Debug)]
pub struct FilePattern {
    /// Directory under the library root, `""` for the root itself.
    pub dir: &'static str,
    /// `{note}` is the key in this library's own spelling, `{layer}` the
    /// 1-based layer index, `{key}` the MIDI number.
    pub template: &'static str,
    /// Octaves this library's note names sit **below** the standard spelling.
    /// VCSL calls MIDI 21 `A-1` where Salamander and bitKlavier call it `A0`,
    /// so it is 1 there and 0 for both of those.
    pub octave_offset: i32,
    /// Added to the MIDI key number to get `{n}`. Both libraries with a
    /// per-key key-off group number those files from the bottom of the
    /// keyboard rather than by MIDI number — `rel1` is MIDI 21 — so this is
    /// −20 there.
    pub key_offset: i32,
}

impl FilePattern {
    /// The relative path of one recording, as the generated SFZ will spell it.
    ///
    /// `{note}` is the key in this library's own spelling, `{layer}` the
    /// 1-based index, `{index}` the 0-based one, `{key}` the MIDI number and
    /// `{n}` that number plus [`FilePattern::key_offset`].
    pub fn path(&self, key: u8, layer: usize, extension: &str) -> String {
        let name = self
            .template
            .replace("{note}", &note_name(key, self.octave_offset))
            .replace("{layer}", &(layer + 1).to_string())
            .replace("{index}", &layer.to_string())
            .replace("{key}", &key.to_string())
            .replace("{n}", &(i32::from(key) + self.key_offset).to_string());
        if self.dir.is_empty() {
            format!("{name}.{extension}")
        } else {
            format!("{}/{name}.{extension}", self.dir)
        }
    }
}

/// One class of mechanism recording and where its files are.
#[derive(Clone, Debug)]
pub struct MechanismFiles {
    pub kind: MechanismKind,
    pub pattern: FilePattern,
    /// The keys this class has a recording for. A key-off group covers all 88
    /// even when the note groups cover 30; the pedal covers none.
    pub keys: Option<Layout>,
    /// Index range for a class that is numbered rather than keyed — the
    /// pedal's `pedalD1`/`pedalD2`.
    pub indices: &'static [usize],
    /// Velocity band, for a class recorded in tiers over the same key.
    pub band: (u8, u8),
    /// `rt_decay` the generated SFZ declares, in dB per second held. A release
    /// recording's level is only comparable with a strike's once the hold is
    /// named (`DECISIONS.md` 501); a generated map declares **zero** unless
    /// the library published a figure, because inventing one would put a
    /// number the estimators read into a file nobody measured.
    pub rt_decay: f64,
}

/// Where the instrument definition comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// The library ships one, at this path under its root. Nothing is
    /// generated and the render path is untouched.
    Shipped(&'static str),
    /// The library ships none, or ships one that cannot be used as a
    /// measurement input; [`LibrarySpec::emit_sfz`] writes one.
    Generated,
}

/// What a sample library is, as data.
#[derive(Clone, Debug)]
pub struct LibrarySpec {
    /// The name `piano-tuner adapt` is asked for.
    pub id: &'static str,
    /// The instrument actually recorded, for the generated file's header.
    pub instrument: &'static str,
    pub credit: &'static str,
    pub licence: &'static str,
    pub source_url: &'static str,
    /// The rate the library was published at.
    pub published_rate_hz: u32,
    /// The rate its files sit at once the fetch script has finished. Equal to
    /// `published_rate_hz` means nothing resampled them.
    pub delivered_rate_hz: u32,
    pub extension: &'static str,
    pub source: Source,
    pub layout: Layout,
    pub bands: Bands,
    pub notes: FilePattern,
    pub mechanism: &'static [MechanismFiles],
    /// `ampeg_release` the generated map declares for the note groups.
    pub ampeg_release_s: f64,
    /// Free text the generated file carries, and the caveats a successor must
    /// read before trusting a stage against this library.
    pub caveats: &'static [&'static str],
}

impl LibrarySpec {
    /// The spec of `id`, or `None`.
    pub fn find(id: &str) -> Option<&'static LibrarySpec> {
        SPECS.iter().find(|spec| spec.id == id)
    }

    pub fn all() -> &'static [LibrarySpec] {
        SPECS
    }

    /// True when the recordings reached the estimators at the rate they were
    /// published at.
    pub fn is_native_rate(&self) -> bool {
        self.published_rate_hz == self.delivered_rate_hz
    }

    /// Every file the description says this library contains, with the ones
    /// that are actually on disk under `root` marked.
    ///
    /// A description is a claim about a tree, and this is the claim checked. A
    /// missing file is never mapped: a generated SFZ that names a recording
    /// that is not there would make the survey fail one key at a time, deep
    /// inside a fit, instead of here.
    pub fn scan(&self, root: &Path) -> Scan {
        let mut notes = Vec::new();
        let bands = self.bands.bands();
        for key in self.layout.keys() {
            for (layer, &(lovel, hivel)) in bands.iter().enumerate() {
                let relative = self.notes.path(key, layer, self.extension);
                let present = root.join(&relative).is_file();
                notes.push(ScannedNote {
                    key,
                    layer,
                    lovel,
                    hivel,
                    relative,
                    present,
                });
            }
        }
        let mut mechanism = Vec::new();
        for class in self.mechanism {
            let keys: Vec<Option<u8>> = match &class.keys {
                Some(layout) => layout.keys().into_iter().map(Some).collect(),
                None => vec![None],
            };
            let indices: Vec<usize> = if class.indices.is_empty() {
                vec![0]
            } else {
                class.indices.to_vec()
            };
            for key in keys {
                for &index in &indices {
                    let relative = class.pattern.path(key.unwrap_or(60), index, self.extension);
                    let present = root.join(&relative).is_file();
                    mechanism.push(ScannedMechanism {
                        kind: class.kind,
                        key,
                        band: class.band,
                        rt_decay: class.rt_decay,
                        relative,
                        present,
                    });
                }
            }
        }
        Scan { notes, mechanism }
    }

    /// The instrument definition, in Salamander's idiom, over the files that
    /// are actually there.
    ///
    /// Three things it deliberately **asserts nothing about**, and each is a
    /// measurement decision rather than a formatting one:
    ///
    /// - `amp_veltrack=0`. Salamander's own file declares 73, and
    ///   `library.rs` has to undo it before two layers' levels can be
    ///   compared. A generated map declares no velocity law at all, so the
    ///   recordings arrive at the estimators at the level they were recorded
    ///   at, which is strictly better material than Salamander's.
    /// - `volume=0` on every group. A library that re-levels its own layers is
    ///   telling the `level` stage about its editor rather than about its
    ///   piano.
    /// - no `tune`, no `offset`. VCSL's generated map carries per-sample
    ///   `tune` of up to −47 cents from a bass pitch-detection failure, which
    ///   would corrupt the tuning-curve estimate outright, and `offset` is an
    ///   editorial trim of the attack the tracker finds for itself.
    pub fn emit_sfz(&self, root: &Path) -> Result<String> {
        if self.source != Source::Generated {
            return Err(Error::Config(format!(
                "{}: ships its own instrument definition; nothing to generate",
                self.id
            )));
        }
        let scan = self.scan(root);
        if scan.present_notes() == 0 {
            return Err(Error::Config(format!(
                "{}: no recordings found under {}",
                self.id,
                root.display()
            )));
        }
        let mut out = String::new();
        let _ = writeln!(out, "//=====================================");
        let _ = writeln!(out, "// {} — generated by `piano-tuner adapt {}`", self.instrument, self.id);
        let _ = writeln!(out, "//");
        let _ = writeln!(out, "// Author:  {}", self.credit);
        let _ = writeln!(out, "// Licence: {}", self.licence);
        let _ = writeln!(out, "// Source:  {}", self.source_url);
        let _ = writeln!(out, "//");
        let _ = writeln!(
            out,
            "// {} recorded keys x {} velocity layers; published at {} Hz, delivered at {} Hz{}.",
            self.layout.keys().len(),
            self.bands.count(),
            self.published_rate_hz,
            self.delivered_rate_hz,
            if self.is_native_rate() {
                " (untouched)"
            } else {
                " (resampled once, offline, by audio::resample)"
            }
        );
        let _ = writeln!(out, "//");
        let _ = writeln!(
            out,
            "// This map asserts no velocity law, no gain and no tuning: amp_veltrack=0,"
        );
        let _ = writeln!(
            out,
            "// volume absent, tune absent, offset absent. It is a measurement input, not"
        );
        let _ = writeln!(out, "// a performance instrument.");
        for caveat in self.caveats {
            let _ = writeln!(out, "//");
            for line in wrap(caveat, 72) {
                let _ = writeln!(out, "// {line}");
            }
        }
        let _ = writeln!(out, "//=====================================");
        let _ = writeln!(out);

        let spans = self.layout.spans();
        let _ = writeln!(out, "//Notes");
        let _ = writeln!(
            out,
            "<group> amp_veltrack=0 ampeg_release={}",
            trim_float(self.ampeg_release_s)
        );
        let _ = writeln!(out);
        for note in scan.notes.iter().filter(|n| n.present) {
            let (lokey, hikey) = spans[&note.key];
            let _ = writeln!(
                out,
                "<region> sample={} lokey={lokey} hikey={hikey} lovel={} hivel={} pitch_keycenter={}",
                note.relative, note.lovel, note.hivel, note.key
            );
        }

        for class in self.mechanism {
            let present: Vec<&ScannedMechanism> = scan
                .mechanism
                .iter()
                .filter(|m| m.present && m.kind == class.kind && m.band == class.band)
                .collect();
            if present.is_empty() {
                continue;
            }
            let _ = writeln!(out);
            let _ = writeln!(out, "//{}", mechanism_comment(class.kind));
            match class.kind {
                MechanismKind::KeyOff => {
                    let _ = writeln!(
                        out,
                        "<group> trigger=release pitch_keytrack=0 amp_veltrack=0 rt_decay={}",
                        trim_float(class.rt_decay)
                    );
                }
                MechanismKind::StringResonance => {
                    let _ = writeln!(
                        out,
                        "<group> trigger=release amp_veltrack=0 rt_decay={}",
                        trim_float(class.rt_decay)
                    );
                }
                MechanismKind::PedalDown => {
                    let _ = writeln!(
                        out,
                        "<group> group=1 lokey=-1 hikey=-1 on_locc64=126 on_hicc64=127 off_by=2"
                    );
                }
                MechanismKind::PedalUp => {
                    let _ = writeln!(out, "<group> group=2 lokey=-1 hikey=-1 on_locc64=0 on_hicc64=1");
                }
            }
            let count = present.len() as f64;
            for (i, entry) in present.iter().enumerate() {
                match (class.kind, entry.key) {
                    (MechanismKind::PedalDown | MechanismKind::PedalUp, _) => {
                        // Round robin over however many takes the library has,
                        // which is what Salamander's `lorand`/`hirand` pair is.
                        let lo = i as f64 / count;
                        let hi = (i + 1) as f64 / count;
                        let _ = writeln!(
                            out,
                            "<region> sample={} lorand={lo:.6} hirand={hi:.6}",
                            entry.relative
                        );
                    }
                    (_, Some(key)) => {
                        // A key-off recording answers its **own** key only —
                        // it is a recording of that damper — while a release
                        // resonance covers the span its note does.
                        let (lokey, hikey) = match class.kind {
                            MechanismKind::KeyOff => (key, key),
                            _ => *spans.get(&key).unwrap_or(&(key, key)),
                        };
                        let _ = write!(
                            out,
                            "<region> sample={} lokey={lokey} hikey={hikey}",
                            entry.relative
                        );
                        if class.kind == MechanismKind::StringResonance {
                            let _ = write!(out, " pitch_keycenter={key}");
                        }
                        if entry.band != (1, 127) {
                            let _ = write!(out, " lovel={} hivel={}", entry.band.0, entry.band.1);
                        }
                        let _ = writeln!(out);
                    }
                    (_, None) => {}
                }
            }
        }
        Ok(out)
    }
}

fn mechanism_comment(kind: MechanismKind) -> &'static str {
    match kind {
        MechanismKind::KeyOff => "HammerNoise — the key and its action returning to rest",
        MechanismKind::StringResonance => "Release string resonances",
        MechanismKind::PedalDown => "pedalAction — the tray going down",
        MechanismKind::PedalUp => "pedalAction — the tray coming up",
    }
}

/// One expected note recording and whether it is there.
#[derive(Clone, Debug)]
pub struct ScannedNote {
    pub key: u8,
    pub layer: usize,
    pub lovel: u8,
    pub hivel: u8,
    pub relative: String,
    pub present: bool,
}

/// One expected mechanism recording and whether it is there.
#[derive(Clone, Debug)]
pub struct ScannedMechanism {
    pub kind: MechanismKind,
    pub key: Option<u8>,
    pub band: (u8, u8),
    pub rt_decay: f64,
    pub relative: String,
    pub present: bool,
}

/// What a description found when it was checked against a tree.
#[derive(Clone, Debug)]
pub struct Scan {
    pub notes: Vec<ScannedNote>,
    pub mechanism: Vec<ScannedMechanism>,
}

impl Scan {
    pub fn present_notes(&self) -> usize {
        self.notes.iter().filter(|n| n.present).count()
    }

    pub fn missing_notes(&self) -> Vec<&ScannedNote> {
        self.notes.iter().filter(|n| !n.present).collect()
    }

    pub fn present_mechanism(&self) -> usize {
        self.mechanism.iter().filter(|m| m.present).count()
    }

    /// The keys with at least one recording on disk.
    pub fn recorded_keys(&self) -> Vec<u8> {
        let mut keys: Vec<u8> = self
            .notes
            .iter()
            .filter(|n| n.present)
            .map(|n| n.key)
            .collect();
        keys.dedup();
        keys
    }
}

/// A key's name in a library's own spelling: `octave_offset` octaves below the
/// standard one, where C4 is MIDI 60.
pub fn note_name(key: u8, octave_offset: i32) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = i32::from(key) / 12 - 1 - octave_offset;
    format!("{}{}", NAMES[usize::from(key % 12)], octave)
}

/// `1` rather than `1.0`, so a generated file reads like a hand-written one.
fn trim_float(x: f64) -> String {
    if (x - x.round()).abs() < 1e-9 {
        format!("{}", x.round() as i64)
    } else {
        format!("{x}")
    }
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// The measurement map of a fetched library tree.
///
/// Every board and every stage of the factory takes a **data directory** and
/// has to find the instrument definition inside it. Nine of them used to do
/// that by joining Salamander's own filename, which is the single most
/// Salamander-shaped line left in the repository — it is why `bench
/// data/bitklavier-piano-bar` used to say "no recordings here".
///
/// The rule, in order:
///
/// 1. A described library, resolved by the directory's own name: a library
///    that ships its map is read from the file it ships, and a generated one
///    from `<id>.sfz`. This is why a tree with two `.sfz` files in it — VCSL's
///    keeps the auto-generated map it ships, deliberately unused — is not
///    ambiguous.
/// 2. Otherwise `<basename>.sfz`.
/// 3. Otherwise the one `.sfz` in the directory, if there is exactly one.
///
/// An undescribed tree with two candidates is an error naming both, rather
/// than a coin toss.
pub fn instrument_path(data: &Path) -> Result<PathBuf> {
    let basename = data
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    if let Some(spec) = LibrarySpec::find(&basename) {
        let path = match spec.source {
            Source::Shipped(name) => data.join(name),
            Source::Generated => data.join(format!("{}.sfz", spec.id)),
        };
        if path.is_file() {
            return Ok(path);
        }
        return Err(Error::Config(format!(
            "{}: {} is described but its map is not there. Run \
             `data/fetch_*.sh`, or `piano-tuner adapt {}` if the tree is already down.",
            path.display(),
            spec.id,
            spec.id
        )));
    }
    let named = data.join(format!("{basename}.sfz"));
    if named.is_file() {
        return Ok(named);
    }
    let mut found: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(data) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("sfz") {
                found.push(path);
            }
        }
    }
    found.sort();
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(Error::Config(format!(
            "{}: no instrument definition here. `data/fetch_*.sh` puts one in place; \
             `piano-tuner adapt --list` names the libraries this repository describes.",
            data.display()
        ))),
        _ => Err(Error::Config(format!(
            "{}: {} instrument definitions and no description to choose between them: {}",
            data.display(),
            found.len(),
            found
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// **Scaffolding, and it is meant to be deleted.**
///
/// Three drivers still join Salamander's filename to their data directory:
/// `tools/melody.rs` and `tools/mics.rs` (the stereo-install track's files)
/// and `tools/tail.rs` (the halo track's). Each is a one-line adoption of
/// [`instrument_path`] that this track did not make, because a one-line change
/// inside another workstream's file is a merge conflict rather than a fix.
///
/// Until they adopt it, a generated library gets a symlink at the name those
/// three look for, so that `tail`, `mics` and `melody` can be run against it
/// at all. It is created only on request, it lives in the gitignored tree, and
/// **the moment those three call `instrument_path` this function and its
/// callers go**. `DECISIONS.md` 521.
pub fn write_legacy_alias(root: &Path, target: &Path) -> Result<PathBuf> {
    const LEGACY: &str = "SalamanderGrandPiano-V3+20200602.sfz";
    let alias = root.join(LEGACY);
    if alias.exists() {
        return Ok(alias);
    }
    let name = target
        .file_name()
        .ok_or_else(|| Error::Config("the map has no filename".into()))?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(name, &alias)
        .map_err(|e| Error::Config(format!("{}: {e}", alias.display())))?;
    #[cfg(not(unix))]
    std::fs::copy(target, &alias)?;
    Ok(alias)
}

/// Brings a whole tree onto the engine's clock, once, offline.
///
/// **The documented method.** Every file under `root` with `from_ext` is
/// decoded, resampled by [`audio::resample`](crate::audio::resample) — the
/// crate's own band-limited sinc resampler (`rubato`), the same one the
/// boundary resampler and the sampler's pitch shift use — and written back as
/// a 32-bit float WAV next to it. Float, not 24-bit integer: the resampler's
/// output is float and quantising it would need a dither decision, and a
/// dither is a noise floor written into the material the halo census reads.
///
/// Doing it here rather than at load time is what keeps the resampler out of
/// every subsequent measurement of that preset: after this runs, the tree is
/// a 48 kHz tree and `audio::load_at` passes it through.
///
/// Returns the number of files converted and the number already at `to_rate`.
pub fn resample_tree(
    root: &Path,
    from_ext: &str,
    to_rate: u32,
    mut progress: impl FnMut(&Path),
) -> Result<(usize, usize)> {
    let mut converted = 0;
    let mut skipped = 0;
    for path in walk(root, from_ext)? {
        let target = path.with_extension("wav");
        if target.is_file() && target != path {
            skipped += 1;
            continue;
        }
        progress(&path);
        let decoded = crate::audio::load(&path)?;
        if decoded.sample_rate == to_rate && path == target {
            skipped += 1;
            continue;
        }
        let channels = if decoded.sample_rate == to_rate {
            decoded.channels
        } else {
            crate::audio::resample(&decoded.channels, decoded.sample_rate, to_rate)?
        };
        crate::audio::Audio::new(to_rate, channels)?.write_wav(&target)?;
        converted += 1;
    }
    Ok((converted, skipped))
}

/// Every file under `root` with `extension`, in a stable order.
pub fn walk(root: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| Error::Config(format!("{}: {e}", dir.display())))?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some(extension) {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

// ---------------------------------------------------------------------------
// The instances

/// Salamander's own sixteen bands, transcribed from
/// `SalamanderGrandPiano-V3+20200602.sfz`. They are markedly uneven — the
/// third layer is two velocities wide and the first is twenty-six — which is
/// why they are [`Bands::Explicit`] and why a library that ships no map does
/// **not** inherit them.
const SALAMANDER_BANDS: &[(u8, u8)] = &[
    (1, 26),
    (27, 34),
    (35, 36),
    (37, 43),
    (44, 46),
    (47, 50),
    (51, 56),
    (57, 64),
    (65, 72),
    (73, 80),
    (81, 88),
    (89, 96),
    (97, 104),
    (105, 112),
    (113, 120),
    (121, 127),
];

/// The 88 keys a key-off group covers.
const ALL_KEYS: Layout = Layout::Chromatic { lo: 21, hi: 108 };

static SPECS: &[LibrarySpec] = &[
    LibrarySpec {
        id: "salamander",
        instrument: "Yamaha C5 grand (Salamander Grand Piano V3)",
        credit: "Alexander Holm; FLAC release assembled for FreePats by roberto@zenvoid.org",
        licence: "CC BY 3.0 — http://creativecommons.org/licenses/by/3.0/",
        source_url: "https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html",
        published_rate_hz: 48_000,
        delivered_rate_hz: 48_000,
        extension: "flac",
        source: Source::Shipped("SalamanderGrandPiano-V3+20200602.sfz"),
        layout: Layout::Interval { lo: 21, step: 3, extra: &[] },
        bands: Bands::Explicit(SALAMANDER_BANDS),
        notes: FilePattern { dir: "samples", template: "{note}v{layer}", octave_offset: 0, key_offset: 0 },
        mechanism: &[],
        ampeg_release_s: 1.0,
        caveats: &[
            "Ships its own SFZ and is played from it. This description exists to be \
             falsified against that file, never to replace it: every board in this \
             repository is barred against renders made through the shipped map.",
        ],
    },
    LibrarySpec {
        id: "bitklavier-piano-bar",
        instrument: "Steinway D concert grand, Taplin Auditorium, Princeton \
                     (bitKlavier Grand Sample Library, Piano Bar mic image)",
        credit: "Daniel Trueman, Princeton University Department of Music",
        licence: "CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/",
        source_url: "https://archive.org/details/bitKlavierGrand_PianoBar_48k24b",
        published_rate_hz: 48_000,
        delivered_rate_hz: 48_000,
        extension: "wav",
        source: Source::Generated,
        layout: Layout::Interval { lo: 21, step: 3, extra: &[] },
        // The author's own methodology says the sixteen layers were built
        // "relatively evenly distributed across the dynamic range", and the
        // library ships no velocity map at all, so even bands are the honest
        // reading and Salamander's uneven ones would be a borrowed fiction
        // (`DECISIONS.md` 519).
        bands: Bands::Even(16),
        notes: FilePattern { dir: "samples", template: "{note}v{layer}", octave_offset: 0, key_offset: 0 },
        mechanism: &[
            MechanismFiles {
                kind: MechanismKind::KeyOff,
                pattern: FilePattern { dir: "samples", template: "rel{n}", octave_offset: 0, key_offset: -20 },
                keys: Some(ALL_KEYS),
                indices: &[],
                band: (1, 127),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::StringResonance,
                pattern: FilePattern { dir: "samples", template: "harm{note}v1", octave_offset: 0, key_offset: 0 },
                keys: Some(Layout::Interval { lo: 21, step: 3, extra: &[] }),
                indices: &[],
                band: (1, 42),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::StringResonance,
                pattern: FilePattern { dir: "samples", template: "harm{note}v2", octave_offset: 0, key_offset: 0 },
                keys: Some(Layout::Interval { lo: 21, step: 3, extra: &[] }),
                indices: &[],
                band: (43, 84),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::StringResonance,
                pattern: FilePattern { dir: "samples", template: "harm{note}v3", octave_offset: 0, key_offset: 0 },
                keys: Some(Layout::Interval { lo: 21, step: 3, extra: &[] }),
                indices: &[],
                band: (85, 127),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::PedalDown,
                pattern: FilePattern { dir: "samples", template: "pedalD{layer}", octave_offset: 0, key_offset: 0 },
                keys: None,
                indices: &[0, 1],
                band: (1, 127),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::PedalUp,
                pattern: FilePattern { dir: "samples", template: "pedalU{layer}", octave_offset: 0, key_offset: 0 },
                keys: None,
                indices: &[0, 1],
                band: (1, 127),
                rt_decay: 0.0,
            },
        ],
        ampeg_release_s: 1.0,
        caveats: &[
            "PER-SAMPLE GAIN REBALANCING. The author's methodology states \
             \"small adjustments (usually <2 dB) to the gains of all the samples so \
             that they were evenly distributed, soft to loud, and so they matched as \
             well as possible across the keyboard\". That is exactly what the `level` \
             stage measures, so on this library `level` fits the EDITOR's balance, not \
             the piano's.",
            "RX8 SPECTRAL DENOISE was applied to every sample to remove room and mic \
             noise. That eats the low-level broadband floor the between-partial halo \
             census reads: the treble-halo shortfall is likely UNMEASURABLE here and \
             the halo track must not be re-baselined against this library.",
            "5 ms ATTACK FADE and Essentia-trimmed leading silence on every pitched \
             sample, and a 100 ms release fade at file end. Hammer-hardness and strike \
             estimates are biased soft; check before trusting `fit --stage hammer`.",
            "The mechanism samples carry far more hall than Salamander's tightly \
             edited ones — measured at -17.5 dB of late energy at 0.3 s against \
             Salamander's -41.6 — which is a real cost to the `noise` stage.",
        ],
    },
    LibrarySpec {
        id: "vcsl-knight-upright",
        instrument: "Knight upright piano (Versilian Community Sample Library)",
        credit: "Versilian Studios LLC; sampled by Simon Dalzell of Ivy Audio for VSCO 2 Pro",
        licence: "CC0 1.0 — https://creativecommons.org/publicdomain/zero/1.0/",
        source_url: "https://github.com/sgossner/VCSL",
        published_rate_hz: 44_100,
        delivered_rate_hz: 48_000,
        extension: "wav",
        source: Source::Generated,
        // Whole tones from A0, plus C8: a whole-tone series from 21 reaches
        // 107 and misses the instrument's top key by one, and the library
        // recorded it anyway (its own chart maps index 044 to key 108).
        layout: Layout::Interval { lo: 21, step: 2, extra: &[108] },
        // Two layers and no published map. Even halves put their midpoints at
        // 32 and 95; the library's own generated SFZ splits at 83/84, which is
        // the generator's default and not a statement about the recordings.
        bands: Bands::Even(2),
        notes: FilePattern {
            dir: "Sustains",
            template: "Player_vl{layer}_rr1_{note}",
            octave_offset: 1,
            key_offset: 0,
        },
        mechanism: &[
            MechanismFiles {
                kind: MechanismKind::StringResonance,
                pattern: FilePattern {
                    dir: "Releases",
                    template: "Player_rel_rr1_{note}",
                    octave_offset: 1,
                    key_offset: 0,
                },
                keys: Some(Layout::Interval { lo: 21, step: 2, extra: &[108] }),
                indices: &[],
                band: (1, 127),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::PedalDown,
                pattern: FilePattern {
                    dir: "Pedal/On",
                    template: "Player_PedOn_00{index}",
                    octave_offset: 0,
                    key_offset: 0,
                },
                keys: None,
                indices: &[0, 1, 2, 3],
                band: (1, 127),
                rt_decay: 0.0,
            },
            MechanismFiles {
                kind: MechanismKind::PedalUp,
                pattern: FilePattern {
                    dir: "Pedal/Off",
                    template: "Player_PedOff_00{index}",
                    octave_offset: 0,
                    key_offset: 0,
                },
                keys: None,
                indices: &[0, 1, 2, 3],
                band: (1, 127),
                rt_decay: 0.0,
            },
        ],
        ampeg_release_s: 0.4,
        caveats: &[
            "TWO VELOCITY LAYERS. The survey reduces a key to the median over its \
             layers — sixteen noisy measurements of one number on Salamander, two \
             here — and the hammer fit gets 2 x 45 = 90 (velocity, value) pairs \
             against Salamander's 16 x 30 = 480. Per-key brightness against velocity \
             is out of scope for this preset by construction.",
            "RESAMPLED ONCE, OFFLINE, 44100 -> 48000, by audio::resample. The tree \
             the estimators read is not the tree that was published; the method is one \
             pass of the crate's own band-limited sinc resampler, written to float WAV \
             so no dither decision enters the material.",
            "THE SHIPPED SFZ IS NOT USED AS A MEASUREMENT INPUT. It carries per-region \
             `volume` that compresses the piano's real 10.5 dB layer difference to \
             about 5, per-sample `tune` of up to -47 cents from a bass pitch-detection \
             failure, and `offset` trims of the attack. All three would be read as \
             the instrument.",
            "NO UNPITCHED KEY-OFF GROUP. The 45 release recordings are the STRINGS \
             still ringing (measured: the A2 release peaks at 220.0 Hz with 84 % of \
             its energy in the fundamental's band), so they are indexed as string \
             resonance. The `noise` stage's key-off term has no material on this \
             library; its pedal term has four takes each way.",
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::library::SampleLibrary;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the tuner sits in the workspace")
            .to_path_buf()
    }

    #[test]
    fn every_spec_has_a_licence_and_a_source() {
        for spec in LibrarySpec::all() {
            assert!(!spec.licence.is_empty(), "{}: no licence", spec.id);
            assert!(
                spec.licence.contains("http"),
                "{}: the licence has no URL, and the standing rule is that the \
                 licence and its URL are recorded before any parameter ships",
                spec.id
            );
            assert!(!spec.source_url.is_empty(), "{}: no source", spec.id);
            assert!(!spec.credit.is_empty(), "{}: no credit", spec.id);
        }
    }

    #[test]
    fn a_note_name_is_a_key_in_the_librarys_own_spelling() {
        assert_eq!(note_name(60, 0), "C4");
        assert_eq!(note_name(21, 0), "A0");
        assert_eq!(note_name(108, 0), "C8");
        assert_eq!(note_name(30, 0), "F#1");
        // VCSL's chart calls MIDI 21 `A-1` and MIDI 108 `C7`.
        assert_eq!(note_name(21, 1), "A-1");
        assert_eq!(note_name(57, 1), "A2");
        assert_eq!(note_name(108, 1), "C7");
    }

    #[test]
    fn even_bands_partition_the_controller() {
        let bands = Bands::Even(16).bands();
        assert_eq!(bands.len(), 16);
        assert_eq!(bands[0].0, 1);
        assert_eq!(bands[15].1, 127);
        for pair in bands.windows(2) {
            assert_eq!(pair[1].0, pair[0].1 + 1, "the bands leave a gap");
        }
        let two = Bands::Even(2).bands();
        assert_eq!(two, vec![(1, 63), (64, 127)]);
    }

    /// The falsification the whole description layer rests on: reconstruct
    /// Salamander from its [`LibrarySpec`] and check it against the file the
    /// library actually ships.
    ///
    /// Skips itself when the tree is not in this checkout — it is 707 MiB and
    /// gitignored — exactly as the other corpus tests do.
    #[test]
    fn the_salamander_description_agrees_with_the_shipped_sfz() {
        let root = repo().join("data/salamander");
        let Source::Shipped(name) = LibrarySpec::find("salamander").unwrap().source else {
            panic!("salamander ships its own map");
        };
        let sfz = root.join(name);
        if !sfz.is_file() {
            eprintln!("no data/salamander in this tree; skipping the description check");
            return;
        }
        let spec = LibrarySpec::find("salamander").unwrap();
        let shipped = SampleLibrary::from_sfz(&sfz).unwrap();

        // 1. The layout. Salamander records thirty keys at minor thirds.
        let described = spec.layout.keys();
        let actual: Vec<u8> = shipped.keys().collect();
        assert_eq!(described, actual, "the described layout is not the shipped one");
        assert_eq!(described.len(), 30);

        // 2. The bands, read off a key the shipped file spells in full.
        let layers = shipped.layers(21);
        let described_bands = spec.bands.bands();
        assert_eq!(layers.len(), described_bands.len());
        for (layer, &(lovel, hivel)) in layers.iter().zip(described_bands.iter()) {
            assert_eq!((layer.lovel, layer.hivel), (lovel, hivel));
        }

        // 3. The spans, including the tie-break. Minor thirds never tie, so
        //    every recorded key owns itself and its two neighbours except at
        //    the ends of the compass.
        let spans = spec.layout.spans();
        assert_eq!(spans[&21], (21, 22));
        assert_eq!(spans[&24], (23, 25));
        assert_eq!(spans[&60], (59, 61));
        assert_eq!(spans[&108], (107, 108));

        // 4. The one place the shipped file is irregular, pinned so that a
        //    reader of this module knows the description is not simply
        //    parroting it: C4's regions carry NO `pitch_keycenter` and are a
        //    recorded key only through the midpoint of `lokey=59 hikey=61`.
        // Read line by line rather than by substring: the shipped file is
        // CRLF, and an assertion that depended on that would be a line-ending
        // test wearing a measurement's clothes.
        let text = std::fs::read_to_string(&sfz).unwrap();
        let c4 = text
            .lines()
            .find(|line| line.contains("samples/C4v1.flac"))
            .expect("the shipped file maps C4's softest layer");
        assert!(c4.contains("lokey=59"), "C4's own region: {c4}");
        assert!(c4.contains("hikey=61"), "C4's own region: {c4}");
        assert!(
            !c4.contains("pitch_keycenter"),
            "C4 has grown a pitch_keycenter; the irregularity this test pins is \
             gone, and with it the reason the description is not simply \
             parroting the file: {c4}"
        );
        // Every other recorded key does declare one, which is what makes C4 the
        // exception rather than the rule.
        let a0 = text
            .lines()
            .find(|line| line.contains("samples/A0v1.flac"))
            .expect("the shipped file maps A0's softest layer");
        assert!(a0.contains("pitch_keycenter=21"), "A0's own region: {a0}");
        assert!(described.contains(&60), "C4 is a recorded key all the same");

        // 5. Nothing is generated for it.
        assert!(spec.emit_sfz(&root).is_err(), "salamander must not be generated");
    }

    #[test]
    fn a_generated_map_asserts_no_gain_no_velocity_law_and_no_tuning() {
        // Over a fixture tree rather than a corpus, so this runs everywhere.
        let dir = std::env::temp_dir().join("piano-tuner-adapter-emit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("Sustains")).unwrap();
        std::fs::create_dir_all(dir.join("Releases")).unwrap();
        std::fs::create_dir_all(dir.join("Pedal/On")).unwrap();
        let spec = LibrarySpec::find("vcsl-knight-upright").unwrap();
        for key in [21u8, 23, 57] {
            for layer in 0..2 {
                std::fs::write(dir.join(spec.notes.path(key, layer, "wav")), b"").unwrap();
            }
        }
        std::fs::write(dir.join("Releases/Player_rel_rr1_A2.wav"), b"").unwrap();
        std::fs::write(dir.join("Pedal/On/Player_PedOn_000.wav"), b"").unwrap();
        let sfz = spec.emit_sfz(&dir).unwrap();

        assert!(sfz.contains("amp_veltrack=0"));
        assert!(!sfz.contains("volume="), "a generated map states no gain");
        assert!(!sfz.contains("tune="), "a generated map states no tuning");
        assert!(!sfz.contains("offset="), "a generated map trims no attack");
        // Only the files that are there.
        assert!(sfz.contains("Player_vl1_rr1_A-1.wav"));
        assert!(!sfz.contains("Player_vl1_rr1_C#0.wav"));
        // The whole-tone tie-break, in the emitted spans.
        assert!(sfz.contains("lokey=21 hikey=22"));
        assert!(sfz.contains("lokey=23 hikey=24"));
        // And it parses back as the library it describes.
        let path = dir.join("generated.sfz");
        std::fs::write(&path, &sfz).unwrap();
        let library = SampleLibrary::from_sfz(&path).unwrap();
        assert_eq!(library.keys().collect::<Vec<_>>(), vec![21, 23, 57]);
        assert_eq!(library.layers(21).len(), 2);
        assert_eq!(library.layers(21)[0].midi_velocity(), 32);
        assert_eq!(library.layers(21)[1].midi_velocity(), 95);
        assert_eq!(library.layers(21)[0].volume_db, 0.0);
        let resonance = library.mechanism_of(MechanismKind::StringResonance);
        assert_eq!(resonance.len(), 1);
        assert_eq!(resonance[0].key, Some(57));
        assert_eq!(library.mechanism_of(MechanismKind::PedalDown).len(), 1);
    }

    #[test]
    fn a_scan_reports_what_is_missing_rather_than_mapping_it() {
        let dir = std::env::temp_dir().join("piano-tuner-adapter-scan");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("samples")).unwrap();
        let spec = LibrarySpec::find("bitklavier-piano-bar").unwrap();
        std::fs::write(dir.join("samples/C4v1.wav"), b"").unwrap();
        let scan = spec.scan(&dir);
        assert_eq!(scan.present_notes(), 1);
        assert_eq!(scan.notes.len(), 30 * 16);
        assert_eq!(scan.missing_notes().len(), 30 * 16 - 1);
        assert_eq!(scan.recorded_keys(), vec![60]);
    }

    #[test]
    fn the_rate_a_preset_was_measured_through_is_part_of_its_description() {
        assert!(LibrarySpec::find("salamander").unwrap().is_native_rate());
        assert!(LibrarySpec::find("bitklavier-piano-bar").unwrap().is_native_rate());
        // The one preset whose material passed through a resampler, and it
        // says so rather than hiding it.
        let knight = LibrarySpec::find("vcsl-knight-upright").unwrap();
        assert!(!knight.is_native_rate());
        assert_eq!(knight.published_rate_hz, 44_100);
        assert_eq!(knight.delivered_rate_hz, crate::SAMPLE_RATE);
    }
}
