//! What a sampled piano ships as: one recording per note per velocity layer,
//! indexed by an SFZ file.
//!
//! The estimators want two things from a library that a directory listing does
//! not give them — which key a file is a recording of, and where in the
//! dynamic range it sits — and both are in the SFZ. Reading the instrument
//! definition rather than the filenames also draws the line in the right place:
//! a file is analysed because a region maps it to a key, so release samples,
//! hammer noises and pedal actions drop out of the survey by construction
//! instead of by a pattern match on somebody's naming scheme.
//!
//! Only the opcodes that answer those two questions are understood
//! (`sample`, `pitch_keycenter`/`key`/`lokey`/`hikey`, `lovel`/`hivel`,
//! `trigger`); everything else in the file is skipped. This is not an SFZ
//! player.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// One recording: a key of the instrument, struck at one velocity layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    pub path: PathBuf,
    /// MIDI key number, 21 (A0) … 108 (C8).
    pub key: u8,
    /// Index of the layer within this key, 0 for the softest.
    pub layer: u8,
    /// MIDI velocity band the layer covers, inclusive.
    pub lovel: u8,
    pub hivel: u8,
}

impl Sample {
    /// The MIDI velocity this layer is the recording *of*: the middle of its
    /// band, which is the velocity that would most often trigger it and the
    /// abscissa the velocity map is fitted against.
    pub fn midi_velocity(&self) -> u8 {
        ((u16::from(self.lovel) + u16::from(self.hivel)) / 2) as u8
    }
}

/// Every struck-note recording an SFZ instrument maps, grouped by key.
#[derive(Clone, Debug, Default)]
pub struct SampleLibrary {
    /// Key → layers, softest first.
    notes: BTreeMap<u8, Vec<Sample>>,
}

impl SampleLibrary {
    /// Reads an SFZ file and indexes the struck-note regions it maps.
    ///
    /// Sample paths are resolved relative to the SFZ file's own directory, as
    /// an SFZ player resolves them.
    pub fn from_sfz(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Self::parse_sfz(&std::fs::read_to_string(path)?, &root)
    }

    fn parse_sfz(text: &str, root: &Path) -> Result<Self> {
        // An SFZ region inherits the opcodes of the headers above it. Only
        // <global>, <master> and <group> nest around regions; a new header of a
        // given rank clears the ones below it.
        let mut global = Opcodes::default();
        let mut master = Opcodes::default();
        let mut group = Opcodes::default();
        let mut region = Opcodes::default();
        let mut current = Level::Global;
        let mut regions: Vec<Opcodes> = Vec::new();

        for token in tokens(text) {
            match token {
                Token::Header(name) => {
                    if current == Level::Region {
                        regions.push(std::mem::take(&mut region));
                    }
                    match name {
                        "global" => {
                            global = Opcodes::default();
                            master = Opcodes::default();
                            group = Opcodes::default();
                            current = Level::Global;
                        }
                        "master" => {
                            master = Opcodes::default();
                            group = Opcodes::default();
                            current = Level::Master;
                        }
                        "group" => {
                            group = Opcodes::default();
                            current = Level::Group;
                        }
                        "region" => {
                            region = global.merged(&master).merged(&group);
                            current = Level::Region;
                        }
                        // <control>, <curve>, <effect>: nothing here reads them,
                        // and none of them may contain a region.
                        _ => current = Level::Other,
                    }
                }
                Token::Opcode(key, value) => {
                    let target = match current {
                        Level::Global => &mut global,
                        Level::Master => &mut master,
                        Level::Group => &mut group,
                        Level::Region => &mut region,
                        Level::Other => continue,
                    };
                    target.set(key, value);
                }
            }
        }
        if current == Level::Region {
            regions.push(region);
        }

        let mut notes: BTreeMap<u8, Vec<Sample>> = BTreeMap::new();
        for region in regions {
            // A release sample is the sound of the key coming *up*: the damper
            // landing, or the string resonance it leaves behind. Neither is a
            // struck note and neither has partials to fit.
            if region.trigger.as_deref().is_some_and(|t| t != "attack") {
                continue;
            }
            let Some(sample) = region.sample.as_deref() else {
                continue;
            };
            let Some(key) = region.key() else { continue };
            notes.entry(key).or_default().push(Sample {
                path: root.join(sample),
                key,
                layer: 0,
                lovel: region.lovel.unwrap_or(1),
                hivel: region.hivel.unwrap_or(127),
            });
        }
        for (&key, layers) in notes.iter_mut() {
            if layers.len() > usize::from(u8::MAX) {
                return Err(Error::Config(format!("key {key} has {} layers", layers.len())));
            }
            layers.sort_by_key(|s| (s.lovel, s.hivel));
            for (i, sample) in layers.iter_mut().enumerate() {
                sample.layer = i as u8;
            }
        }
        Ok(Self { notes })
    }

    /// The sampled keys, ascending.
    pub fn keys(&self) -> impl Iterator<Item = u8> + '_ {
        self.notes.keys().copied()
    }

    pub fn key_count(&self) -> usize {
        self.notes.len()
    }

    pub fn sample_count(&self) -> usize {
        self.notes.values().map(Vec::len).sum()
    }

    /// The layers recorded for one key, softest first.
    pub fn layers(&self, key: u8) -> &[Sample] {
        self.notes.get(&key).map_or(&[], Vec::as_slice)
    }

    /// Every recording in the library, key by key and layer by layer.
    pub fn samples(&self) -> impl Iterator<Item = &Sample> {
        self.notes.values().flatten()
    }

    /// The same library with only `keys` in it — how a pilot run over three
    /// notes is asked for.
    pub fn restricted_to(&self, keys: &[u8]) -> Self {
        Self {
            notes: self
                .notes
                .iter()
                .filter(|(key, _)| keys.contains(key))
                .map(|(key, layers)| (*key, layers.clone()))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Global,
    Master,
    Group,
    Region,
    Other,
}

/// The handful of opcodes this module understands, at one level of the
/// hierarchy.
#[derive(Clone, Debug, Default)]
struct Opcodes {
    sample: Option<String>,
    pitch_keycenter: Option<i32>,
    key: Option<i32>,
    lokey: Option<i32>,
    hikey: Option<i32>,
    lovel: Option<u8>,
    hivel: Option<u8>,
    trigger: Option<String>,
}

impl Opcodes {
    /// `other`'s settings laid over this level's.
    fn merged(&self, other: &Opcodes) -> Opcodes {
        Opcodes {
            sample: other.sample.clone().or_else(|| self.sample.clone()),
            pitch_keycenter: other.pitch_keycenter.or(self.pitch_keycenter),
            key: other.key.or(self.key),
            lokey: other.lokey.or(self.lokey),
            hikey: other.hikey.or(self.hikey),
            lovel: other.lovel.or(self.lovel),
            hivel: other.hivel.or(self.hivel),
            trigger: other.trigger.clone().or_else(|| self.trigger.clone()),
        }
    }

    fn set(&mut self, name: &str, value: &str) {
        match name {
            "sample" => self.sample = Some(value.replace('\\', "/")),
            "pitch_keycenter" => self.pitch_keycenter = note_number(value),
            "key" => self.key = note_number(value),
            "lokey" => self.lokey = note_number(value),
            "hikey" => self.hikey = note_number(value),
            "lovel" => self.lovel = value.parse().ok(),
            "hivel" => self.hivel = value.parse().ok(),
            "trigger" => self.trigger = Some(value.to_ascii_lowercase()),
            _ => {}
        }
    }

    /// Which key this region is a recording of.
    ///
    /// `pitch_keycenter` is the pitch the sample was recorded at, which is the
    /// question; `key` sets it and the range together. Failing both, a region
    /// mapped across a range is taken to be a recording of the middle of it —
    /// true of every library that samples a subset of the compass and stretches
    /// each recording over its neighbours. A range that is not on the keyboard
    /// (SFZ spells "never plays on a key" as `lokey=-1`) is not a note.
    fn key(&self) -> Option<u8> {
        let center = self.pitch_keycenter.or(self.key).or_else(|| {
            let (lo, hi) = (self.lokey?, self.hikey.unwrap_or(self.lokey?));
            Some((lo + hi) / 2)
        })?;
        u8::try_from(center).ok().filter(|k| (21..=108).contains(k))
    }
}

/// An SFZ header `<name>` or an opcode `name=value`.
enum Token<'a> {
    Header(&'a str),
    Opcode(&'a str, &'a str),
}

/// Splits an SFZ file into headers and opcodes.
///
/// SFZ separates opcodes by whitespace but allows the *value* of the last one
/// on a line to contain spaces (sample paths do, in the wild). The rule that
/// makes both work is that a value runs to the next token containing an `=`,
/// which is where the next opcode must start.
fn tokens(text: &str) -> Vec<Token<'_>> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        let mut rest = line.trim();
        while !rest.is_empty() {
            if let Some(stripped) = rest.strip_prefix('<') {
                let Some(end) = stripped.find('>') else { break };
                out.push(Token::Header(&stripped[..end]));
                rest = stripped[end + 1..].trim_start();
                continue;
            }
            let Some(eq) = rest.find('=') else { break };
            let name = rest[..eq].trim();
            let after = &rest[eq + 1..];
            // The value ends where the next opcode starts: the last whitespace
            // before the next `=`.
            let end = match after.find('=') {
                Some(next) => after[..next].rfind(char::is_whitespace).unwrap_or(after.len()),
                None => after.len(),
            };
            out.push(Token::Opcode(name, after[..end].trim()));
            rest = after[end..].trim_start();
        }
    }
    out
}

/// An SFZ key: either a MIDI number or a note name like `c#4` (C4 = 60, as in
/// the rest of this crate).
fn note_number(text: &str) -> Option<i32> {
    let text = text.trim();
    if let Ok(number) = text.parse::<i32>() {
        return Some(number);
    }
    let mut chars = text.chars();
    let step = match chars.next()?.to_ascii_lowercase() {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };
    let rest = chars.as_str();
    let (accidental, rest) = match rest.chars().next() {
        Some('#') => (1, &rest[1..]),
        Some('b') => (-1, &rest[1..]),
        _ => (0, rest),
    };
    let octave: i32 = rest.parse().ok()?;
    Some((octave + 1) * 12 + step + accidental)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SFZ: &str = "\
// a comment
<group> amp_veltrack=73 ampeg_release=1
<region> sample=samples/C4v2.flac lokey=59 hikey=61 lovel=27 hivel=34 pitch_keycenter=60
<region> sample=samples/C4v1.flac lokey=59 hikey=61 lovel=1 hivel=26 pitch_keycenter=60
<region> sample=samples/C4v3.flac lokey=59 hikey=61 lovel=35 pitch_keycenter=60
<group> trigger=release volume=-4
<region> sample=samples/harmC4.flac lokey=59 hikey=61
<group> group=1 hikey=-1 lokey=-1 on_locc64=126
<region> sample=samples/pedalD1.flac
";

    fn library() -> SampleLibrary {
        SampleLibrary::parse_sfz(SFZ, Path::new("/lib")).unwrap()
    }

    #[test]
    fn regions_are_indexed_by_key_and_ordered_by_velocity() {
        let library = library();
        assert_eq!(library.key_count(), 1);
        assert_eq!(library.sample_count(), 3);
        let layers = library.layers(60);
        assert_eq!(
            layers.iter().map(|s| s.layer).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(layers[0].path, Path::new("/lib/samples/C4v1.flac"));
        assert_eq!(layers[0].midi_velocity(), 13);
        assert_eq!(layers[1].midi_velocity(), 30);
        // An unset hivel is the top of the range, not zero.
        assert_eq!(layers[2].hivel, 127);
    }

    #[test]
    fn release_samples_and_unmapped_regions_are_not_notes() {
        // The release group and the pedal group are both in the fixture; one is
        // excluded by its trigger and the other by having no key at all.
        assert!(library().samples().all(|s| s.key == 60));
    }

    #[test]
    fn a_region_inherits_the_group_and_the_group_does_not_leak_backwards() {
        let sfz = "<group> lovel=1 hivel=64 pitch_keycenter=60\n\
                   <region> sample=a.flac\n\
                   <group> pitch_keycenter=72\n\
                   <region> sample=b.flac lovel=65 hivel=127\n";
        let library = SampleLibrary::parse_sfz(sfz, Path::new(".")).unwrap();
        assert_eq!(library.layers(60)[0].hivel, 64);
        // The second group replaced the first, so b.flac is not velocity-split.
        assert_eq!(library.layers(72)[0].lovel, 65);
        assert_eq!(library.layers(72).len(), 1);
    }

    #[test]
    fn keys_may_be_named_and_a_range_stands_in_for_a_missing_centre() {
        let sfz = "<region> sample=a.flac key=c4\n\
                   <region> sample=b.flac lokey=71 hikey=73\n\
                   <region> sample=c.flac key=f#2\n";
        let library = SampleLibrary::parse_sfz(sfz, Path::new(".")).unwrap();
        assert_eq!(library.keys().collect::<Vec<_>>(), vec![42, 60, 72]);
    }

    #[test]
    fn a_sample_path_may_contain_spaces() {
        let sfz = "<region> key=60 sample=my samples/C4 v1.flac\n";
        let library = SampleLibrary::parse_sfz(sfz, Path::new("/lib")).unwrap();
        assert_eq!(
            library.layers(60)[0].path,
            Path::new("/lib/my samples/C4 v1.flac")
        );
    }
}
