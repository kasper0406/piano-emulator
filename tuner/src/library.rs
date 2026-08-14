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
//! The same file also maps what the instrument does when it is *not* being
//! struck — the key-off thumps and the pedal action — and those are the
//! parameter set for the engine's mechanism noises
//! ([`estimate::noise`](crate::estimate::noise)). They are indexed separately,
//! by [`SampleLibrary::mechanism`], and again by what the SFZ says rather than
//! by what the files are called: a release region with `pitch_keytrack=0` is a
//! recording of the action (an unpitched sample the player must not transpose),
//! one without it is the string still ringing, and a region triggered by CC 64
//! crossing rather than by a key is the pedal.
//!
//! Only the opcodes that answer those questions are understood (`sample`,
//! `pitch_keycenter`/`key`/`lokey`/`hikey`, `lovel`/`hivel`, `trigger`,
//! `volume`, `amp_veltrack`, `pitch_keytrack`, `on_locc64`/`on_hicc64`);
//! everything else in the file is skipped. This is not an SFZ player.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// One recording: a key of the instrument, struck at one velocity layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub path: PathBuf,
    /// MIDI key number, 21 (A0) … 108 (C8).
    pub key: u8,
    /// Index of the layer within this key, 0 for the softest.
    pub layer: u8,
    /// MIDI velocity band the layer covers, inclusive.
    pub lovel: u8,
    pub hivel: u8,
    /// The `volume` the instrument plays this region at, in dB. Levels are only
    /// comparable between groups once it is applied: Salamander attenuates its
    /// key-off group by 37 dB and its pedal groups by 19–20, and comparing the
    /// raw files would say a damper landing is as loud as the note.
    pub volume_db: f64,
}

impl Sample {
    /// The MIDI velocity this layer is the recording *of*: the middle of its
    /// band, which is the velocity that would most often trigger it and the
    /// abscissa the velocity map is fitted against.
    pub fn midi_velocity(&self) -> u8 {
        ((u16::from(self.lovel) + u16::from(self.hivel)) / 2) as u8
    }
}

/// Which part of the action a recording is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MechanismKind {
    /// The key and its action returning to rest, with the damper landing: one
    /// per key, `rel1`…`rel88` in Salamander.
    KeyOff,
    /// The sustain pedal's tray going down, triggered by CC 64 crossing high.
    PedalDown,
    /// The same tray coming up.
    PedalUp,
}

/// One recording of the mechanism rather than of a string.
#[derive(Clone, Debug, PartialEq)]
pub struct MechanismSample {
    pub path: PathBuf,
    pub kind: MechanismKind,
    /// The key this recording belongs to, where it has one. The pedal is
    /// global and has none.
    pub key: Option<u8>,
    /// `volume`, in dB, as for [`Sample::volume_db`].
    pub volume_db: f64,
    /// `amp_veltrack`, as a percentage, where the group sets one. The SFZ law
    /// is `40 log10(v / 127)` dB scaled by this over a hundred.
    pub amp_veltrack: Option<f64>,
}

/// Every recording an SFZ instrument maps: the struck notes grouped by key, and
/// the mechanism.
#[derive(Clone, Debug, Default)]
pub struct SampleLibrary {
    /// Key → layers, softest first.
    notes: BTreeMap<u8, Vec<Sample>>,
    mechanism: Vec<MechanismSample>,
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
        let mut mechanism: Vec<MechanismSample> = Vec::new();
        for region in regions {
            let Some(sample) = region.sample.as_deref() else {
                continue;
            };
            let path = root.join(sample);
            if let Some(kind) = region.mechanism() {
                mechanism.push(MechanismSample {
                    path,
                    kind,
                    key: region.key(),
                    volume_db: region.volume.unwrap_or(0.0),
                    amp_veltrack: region.amp_veltrack,
                });
                continue;
            }
            // A release sample is the sound of the key coming *up*: the damper
            // landing, or the string resonance it leaves behind. Neither is a
            // struck note and neither has partials to fit.
            if region.trigger.as_deref().is_some_and(|t| t != "attack") {
                continue;
            }
            let Some(key) = region.key() else { continue };
            notes.entry(key).or_default().push(Sample {
                path,
                key,
                layer: 0,
                lovel: region.lovel.unwrap_or(1),
                hivel: region.hivel.unwrap_or(127),
                volume_db: region.volume.unwrap_or(0.0),
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
        Ok(Self { notes, mechanism })
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

    /// Every mechanism recording the instrument maps, in file order.
    pub fn mechanism(&self) -> &[MechanismSample] {
        &self.mechanism
    }

    /// The mechanism recordings of one kind, ascending by key.
    pub fn mechanism_of(&self, kind: MechanismKind) -> Vec<&MechanismSample> {
        let mut samples: Vec<&MechanismSample> =
            self.mechanism.iter().filter(|s| s.kind == kind).collect();
        samples.sort_by_key(|s| s.key);
        samples
    }

    /// The layer of `key` a strike at `velocity` would trigger, or the same
    /// layer of the nearest key that has one.
    ///
    /// A library samples a subset of the compass — Salamander records 30 keys
    /// of 88 — while it ships a key-off recording for every key, so a level
    /// quoted against "a strike of the same key" has to fall back to the
    /// nearest key that was struck at all. Both quantities move smoothly across
    /// a couple of semitones; which key it settled for is returned so a caller
    /// can say so.
    pub fn nearest_layer(&self, key: u8, velocity: u8) -> Option<&Sample> {
        self.notes
            .iter()
            .filter_map(|(&sampled, layers)| {
                let layer = layers
                    .iter()
                    .find(|s| (s.lovel..=s.hivel).contains(&velocity))?;
                Some((sampled.abs_diff(key), layer))
            })
            .min_by_key(|&(distance, _)| distance)
            .map(|(_, layer)| layer)
    }

    /// The MIDI velocities the library's own layers are centred on, lowest and
    /// highest — the dynamic range it was recorded across.
    pub fn velocity_span(&self) -> Option<(u8, u8)> {
        let mut velocities = self.samples().map(Sample::midi_velocity);
        let first = velocities.next()?;
        Some(velocities.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v))))
    }

    /// The same library with only `keys` in it — how a pilot run over three
    /// notes is asked for. The mechanism is carried over whole: it is not
    /// per-key material and a pilot run over three notes still wants it.
    pub fn restricted_to(&self, keys: &[u8]) -> Self {
        Self {
            notes: self
                .notes
                .iter()
                .filter(|(key, _)| keys.contains(key))
                .map(|(key, layers)| (*key, layers.clone()))
                .collect(),
            mechanism: self.mechanism.clone(),
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
    volume: Option<f64>,
    amp_veltrack: Option<f64>,
    pitch_keytrack: Option<i32>,
    on_locc64: Option<i32>,
    on_hicc64: Option<i32>,
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
            volume: other.volume.or(self.volume),
            amp_veltrack: other.amp_veltrack.or(self.amp_veltrack),
            pitch_keytrack: other.pitch_keytrack.or(self.pitch_keytrack),
            on_locc64: other.on_locc64.or(self.on_locc64),
            on_hicc64: other.on_hicc64.or(self.on_hicc64),
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
            "volume" => self.volume = value.parse().ok(),
            "amp_veltrack" => self.amp_veltrack = value.parse().ok(),
            "pitch_keytrack" => self.pitch_keytrack = value.parse().ok(),
            "on_locc64" => self.on_locc64 = value.parse().ok(),
            "on_hicc64" => self.on_hicc64 = value.parse().ok(),
            _ => {}
        }
    }

    /// Which part of the mechanism this region records, if any.
    ///
    /// Two markers, both of them statements the SFZ makes about how the sample
    /// must be *played* rather than about what it is called. A release region
    /// with `pitch_keytrack=0` is a sample the player is told not to transpose,
    /// which is what an unpitched noise is; a release region without it is the
    /// string, and has a pitch to track. A region gated on CC 64 rather than on
    /// a key is the pedal, and which way the gate opens says which direction
    /// the tray moved.
    fn mechanism(&self) -> Option<MechanismKind> {
        if self.on_locc64.is_some() || self.on_hicc64.is_some() {
            let low = self.on_locc64.unwrap_or(0);
            let high = self.on_hicc64.unwrap_or(127);
            return Some(if low >= 64 {
                MechanismKind::PedalDown
            } else if high <= 63 {
                MechanismKind::PedalUp
            } else {
                // A gate that spans the whole controller is not a crossing.
                return None;
            });
        }
        let released = self.trigger.as_deref().is_some_and(|t| t == "release");
        (released && self.pitch_keytrack == Some(0)).then_some(MechanismKind::KeyOff)
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
<group> trigger=release volume=-4 amp_veltrack=94 rt_decay=6
<region> sample=samples/harmC4.flac lokey=59 hikey=61
//HammerNoise
<group> trigger=release pitch_keytrack=0 volume=-37 amp_veltrack=82 rt_decay=2
<region> sample=samples/rel40.flac lokey=60 hikey=60
<region> sample=samples/rel1.flac lokey=21 hikey=21
<group> group=1 hikey=-1 lokey=-1 on_locc64=126 on_hicc64=127 off_by=2 volume=-20
<region> sample=samples/pedalD1.flac
<group> group=2 hikey=-1 lokey=-1 on_locc64=0 on_hicc64=1 volume=-19
<region> sample=samples/pedalU1.flac
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
    fn the_action_is_indexed_by_what_the_sfz_says_it_is() {
        let library = library();
        let key_off = library.mechanism_of(MechanismKind::KeyOff);
        // Both hammer-noise regions, ordered by key — and *not* the string
        // resonance in the release group above them, which has a pitch to
        // track and is a recording of the string.
        assert_eq!(
            key_off.iter().map(|s| s.key).collect::<Vec<_>>(),
            vec![Some(21), Some(60)]
        );
        assert_eq!(key_off[0].path, Path::new("/lib/samples/rel1.flac"));
        assert_eq!(key_off[0].volume_db, -37.0);
        assert_eq!(key_off[0].amp_veltrack, Some(82.0));
        assert!(library
            .mechanism()
            .iter()
            .all(|s| !s.path.ends_with("harmC4.flac")));

        // The pedal, told apart by which way its gate on CC 64 opens.
        let down = library.mechanism_of(MechanismKind::PedalDown);
        let up = library.mechanism_of(MechanismKind::PedalUp);
        assert_eq!(down.len(), 1);
        assert_eq!(down[0].volume_db, -20.0);
        assert_eq!(down[0].key, None);
        assert_eq!(up.len(), 1);
        assert_eq!(up[0].path, Path::new("/lib/samples/pedalU1.flac"));

        // A key-off recording of a key the library never struck still has a
        // strike to be measured against.
        assert_eq!(library.nearest_layer(21, 90).map(|s| s.key), Some(60));
        assert_eq!(library.nearest_layer(60, 90).map(|s| s.layer), Some(2));
        assert_eq!(library.nearest_layer(60, 200), None);
        assert_eq!(library.velocity_span(), Some((13, 81)));
        // A pilot run over three notes still carries the whole mechanism.
        assert_eq!(library.restricted_to(&[]).mechanism().len(), 4);
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
