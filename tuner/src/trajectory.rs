//! The data the analysis front end hands to the estimators: for one recorded
//! note, the measured frequency and amplitude of each partial as a function of
//! time.
//!
//! Everything here is `serde`-serializable so a run of the tracker — which is
//! minutes of FFTs over a whole sample library — can be cached to disk and the
//! estimators iterated against it.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// The stiff-string partial layout,
///
/// ```text
/// f_k = k f0 sqrt(1 + B k^2 + B4 k^4)
/// ```
///
/// which is both what the engine synthesizes (`SPEC.md`, "String") and what
/// seeds the tracker's search for partial `k`.
///
/// `B4` is signed and is zero unless a fit put something there — at which point
/// the law is exactly the two-parameter one, term for term.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InharmonicModel {
    pub f0_hz: f64,
    /// Inharmonicity coefficient `B`, dimensionless; ~1e-4 in the bass to
    /// ~1e-2 at the top of the compass.
    pub b: f64,
    /// Fourth-order coefficient `B4`, **signed** and normally zero. A wound
    /// bass string's `B` falls 25–37 % along its own series and a short wound
    /// tenor string's rises 24–45 % (`TUNING_REPORT.md` §1); one `k^4` term is
    /// how much of that shape the engine can be told about.
    ///
    /// `#[serde(default)]`: trajectory caches written before the term existed
    /// reload as the two-parameter model they were.
    #[serde(default)]
    pub b4: f64,
}

impl InharmonicModel {
    pub fn new(f0_hz: f64, b: f64) -> Self {
        Self { f0_hz, b, b4: 0.0 }
    }

    /// The same layout with a fourth-order term.
    pub fn with_b4(f0_hz: f64, b: f64, b4: f64) -> Self {
        Self { f0_hz, b, b4 }
    }

    /// A perfectly harmonic series — the right seed when nothing is known
    /// about the string's stiffness yet.
    pub fn harmonic(f0_hz: f64) -> Self {
        Self {
            f0_hz,
            b: 0.0,
            b4: 0.0,
        }
    }

    /// Frequency of partial `k` (1-based).
    pub fn partial(&self, k: u32) -> f64 {
        let k = f64::from(k);
        let k2 = k * k;
        k * self.f0_hz * (1.0 + self.b * k2 + self.b4 * k2 * k2).max(0.0).sqrt()
    }

    /// The highest partial index at or below `limit_hz`, capped at `max_k`.
    /// `f_k` is strictly increasing in `k` for the coefficients any fit here
    /// produces, so this is a simple walk rather than a search.
    pub fn partials_below(&self, limit_hz: f64, max_k: u32) -> u32 {
        let mut k = 0;
        while k < max_k && self.partial(k + 1) <= limit_hz {
            k += 1;
        }
        k
    }

    /// Deviation of `measured` from partial `k` of this model, in cents.
    pub fn cents_from_partial(&self, k: u32, measured_hz: f64) -> f64 {
        cents(self.partial(k), measured_hz)
    }
}

/// Interval between two frequencies in cents. Positive when `b` is the higher.
pub fn cents(a_hz: f64, b_hz: f64) -> f64 {
    if a_hz <= 0.0 || b_hz <= 0.0 {
        return f64::NAN;
    }
    1200.0 * (b_hz / a_hz).log2()
}

/// Which recorded note a set of trajectories came from. Both fields are
/// optional information the analysis itself never needs; they exist so cached
/// trajectory files are self-describing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteId {
    /// MIDI key number, 21 (A0) … 108 (C8).
    pub key: u8,
    /// Index of the velocity layer in the source library, if it has layers.
    pub velocity_layer: Option<u8>,
}

impl NoteId {
    pub fn new(key: u8) -> Self {
        Self {
            key,
            velocity_layer: None,
        }
    }

    pub fn layer(key: u8, velocity_layer: u8) -> Self {
        Self {
            key,
            velocity_layer: Some(velocity_layer),
        }
    }
}

/// One measurement of one partial: its frequency and amplitude at one instant.
///
/// `amplitude` is in the units of the input signal — the amplitude of the
/// sinusoid, not its RMS and not a power — so that a partial rendered at
/// full scale reads 1.0. `time_s` is the centre of the analysis window the
/// measurement came from, measured from the start of the analysed signal.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackPoint {
    pub time_s: f64,
    pub frequency_hz: f64,
    pub amplitude: f64,
}

/// The trajectory of a single partial: `(k, f_k(t), a_k(t))`.
///
/// Points are in ascending time order but need not be contiguous — a partial
/// that dips into the noise floor and comes back leaves a gap rather than two
/// tracks, because it is still the same mode of the same string.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PartialTrack {
    /// Partial index, 1-based; `k = 1` is the fundamental.
    pub k: u32,
    pub points: Vec<TrackPoint>,
}

impl PartialTrack {
    pub fn new(k: u32) -> Self {
        Self { k, points: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn start_s(&self) -> Option<f64> {
        self.points.first().map(|p| p.time_s)
    }

    pub fn end_s(&self) -> Option<f64> {
        self.points.last().map(|p| p.time_s)
    }

    /// The loudest measurement on the track — the natural reference point for
    /// a decay fit, and for a partial's contribution to the strike spectrum.
    pub fn peak(&self) -> Option<TrackPoint> {
        self.points
            .iter()
            .copied()
            .max_by(|a, b| a.amplitude.total_cmp(&b.amplitude))
    }

    /// Median measured frequency. Robust against the frames where a partial
    /// has decayed into the noise, which is what the mean is not.
    pub fn median_frequency(&self) -> Option<f64> {
        median(self.points.iter().map(|p| p.frequency_hz))
    }

    /// Amplitude-weighted mean frequency — the estimate to feed an `f0`/`B`
    /// fit, since the loud frames are the ones whose frequency is trustworthy.
    pub fn weighted_frequency(&self) -> Option<f64> {
        let (num, den) = self
            .points
            .iter()
            .fold((0.0, 0.0), |(n, d), p| (n + p.frequency_hz * p.amplitude, d + p.amplitude));
        if den > 0.0 {
            Some(num / den)
        } else {
            None
        }
    }

    /// Linear interpolation of the amplitude envelope at `t`; `None` outside
    /// the track's own time span (never extrapolates — the estimators must see
    /// missing data as missing).
    pub fn amplitude_at(&self, t: f64) -> Option<f64> {
        self.interpolate(t, |p| p.amplitude)
    }

    /// Linear interpolation of the measured frequency at `t`.
    pub fn frequency_at(&self, t: f64) -> Option<f64> {
        self.interpolate(t, |p| p.frequency_hz)
    }

    fn interpolate(&self, t: f64, field: impl Fn(&TrackPoint) -> f64) -> Option<f64> {
        let first = self.points.first()?;
        let last = self.points.last()?;
        if t < first.time_s || t > last.time_s {
            return None;
        }
        let idx = self
            .points
            .partition_point(|p| p.time_s <= t)
            .clamp(1, self.points.len() - 1);
        let (a, b) = (&self.points[idx - 1], &self.points[idx]);
        let span = b.time_s - a.time_s;
        if span <= 0.0 {
            return Some(field(a));
        }
        let u = (t - a.time_s) / span;
        Some(field(a) * (1.0 - u) + field(b) * u)
    }
}

/// Every partial trajectory extracted from one recording, plus the analysis
/// settings that produced them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoteTrajectories {
    /// Where the audio came from; free-form provenance for cached files.
    pub source: String,
    pub note: Option<NoteId>,
    pub sample_rate: f64,
    /// Analysis window length in seconds. The estimators need it: it is the
    /// time resolution of every envelope in `tracks`, and no feature shorter
    /// than it survived the transform.
    pub window_s: f64,
    pub hop_s: f64,
    /// The partial layout used to seed track association. Measured
    /// frequencies are close to, but deliberately not constrained to, it.
    pub seed: InharmonicModel,
    /// Estimated strike time, in seconds from the start of the analysed
    /// signal. Amplitudes are reported on the recording's clock, so the
    /// estimators subtract this to get time since the strike.
    pub onset_s: f64,
    /// One entry per partial that was found, in ascending `k`.
    pub tracks: Vec<PartialTrack>,
}

impl NoteTrajectories {
    pub fn track(&self, k: u32) -> Option<&PartialTrack> {
        self.tracks.iter().find(|t| t.k == k)
    }

    pub fn max_k(&self) -> u32 {
        self.tracks.iter().map(|t| t.k).max().unwrap_or(0)
    }

    /// Total number of measurements, across all partials.
    pub fn point_count(&self) -> usize {
        self.tracks.iter().map(|t| t.len()).sum()
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn with_note(mut self, note: NoteId) -> Self {
        self.note = Some(note);
        self
    }

    /// Cache to disk. Pretty-printed: these files get read by humans when an
    /// estimator misbehaves, and they compress away to nothing anyway.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)?;
        Ok(())
    }

    pub fn read_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        Ok(serde_json::from_reader(BufReader::new(file))?)
    }
}

/// The exact, compact form the self-calibration gate keeps its corpus in.
///
/// The JSON above is the *human's* copy — a survey writes it, and somebody reads
/// it when an estimator misbehaves. This is the machine's: every float as its
/// own bit pattern, so a decode is exactly the value that was encoded, and a
/// note that costs 3.2 s to track reads back in milliseconds instead of the
/// couple of seconds `serde_json`'s exact float parser wants for five megabytes
/// of digits ([`crate::cache::Cacheable`], `DECISIONS.md` 284).
///
/// **The encoder destructures the struct exhaustively on purpose.** Adding a
/// field to [`NoteTrajectories`] or to anything it holds is then a compile
/// error here rather than a field silently dropped from every cached note.
impl crate::cache::Cacheable for NoteTrajectories {
    fn encode(&self) -> Vec<u8> {
        use crate::cache::write;
        let NoteTrajectories {
            source,
            note,
            sample_rate,
            window_s,
            hop_s,
            seed,
            onset_s,
            tracks,
        } = self;
        let mut out = Vec::with_capacity(64 + 24 * self.point_count());
        out.extend_from_slice(CORPUS_MAGIC);
        write::string(&mut out, source);
        match note {
            None => out.push(0),
            Some(NoteId { key, velocity_layer }) => {
                out.push(1);
                out.push(*key);
                match velocity_layer {
                    None => out.push(0),
                    Some(layer) => {
                        out.push(1);
                        out.push(*layer);
                    }
                }
            }
        }
        write::f64(&mut out, *sample_rate);
        write::f64(&mut out, *window_s);
        write::f64(&mut out, *hop_s);
        let InharmonicModel { f0_hz, b, b4 } = seed;
        write::f64(&mut out, *f0_hz);
        write::f64(&mut out, *b);
        write::f64(&mut out, *b4);
        write::f64(&mut out, *onset_s);
        write::u64(&mut out, tracks.len() as u64);
        for PartialTrack { k, points } in tracks {
            write::u32(&mut out, *k);
            write::u64(&mut out, points.len() as u64);
            for TrackPoint {
                time_s,
                frequency_hz,
                amplitude,
            } in points
            {
                write::f64(&mut out, *time_s);
                write::f64(&mut out, *frequency_hz);
                write::f64(&mut out, *amplitude);
            }
        }
        out
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        use crate::cache::read;
        let mut rest = bytes.strip_prefix(CORPUS_MAGIC)?;
        let bytes = &mut rest;
        let source = read::string(bytes)?;
        let note = match read::u8(bytes)? {
            0 => None,
            1 => {
                let key = read::u8(bytes)?;
                let velocity_layer = match read::u8(bytes)? {
                    0 => None,
                    1 => Some(read::u8(bytes)?),
                    _ => return None,
                };
                Some(NoteId { key, velocity_layer })
            }
            _ => return None,
        };
        let sample_rate = read::f64(bytes)?;
        let window_s = read::f64(bytes)?;
        let hop_s = read::f64(bytes)?;
        let seed = InharmonicModel {
            f0_hz: read::f64(bytes)?,
            b: read::f64(bytes)?,
            b4: read::f64(bytes)?,
        };
        let onset_s = read::f64(bytes)?;
        // Every count is checked against what is left in the buffer before it is
        // reserved, so a corrupt length is a miss and not an allocation.
        let track_count = read::len(bytes)?;
        let mut tracks = Vec::with_capacity(track_count);
        for _ in 0..track_count {
            let k = read::u32(bytes)?;
            let point_count = read::len(bytes)?;
            let mut points = Vec::with_capacity(point_count);
            for _ in 0..point_count {
                points.push(TrackPoint {
                    time_s: read::f64(bytes)?,
                    frequency_hz: read::f64(bytes)?,
                    amplitude: read::f64(bytes)?,
                });
            }
            tracks.push(PartialTrack { k, points });
        }
        // Anything left over is a file this decoder does not understand.
        bytes.is_empty().then_some(NoteTrajectories {
            source,
            note,
            sample_rate,
            window_s,
            hop_s,
            seed,
            onset_s,
            tracks,
        })
    }
}

/// Identifies the compact encoding, and versions it: a changed layout gets a
/// changed magic and every old entry misses.
const CORPUS_MAGIC: &[u8] = b"PTNT0001";

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_partial_layout_matches_the_engines_formula() {
        let m = InharmonicModel::new(261.6, 4e-4);
        assert!((m.partial(1) - 261.6 * (1.0 + 4e-4f64).sqrt()).abs() < 1e-9);
        // k=8 of a B=4e-4 string is 51 cents sharp of the eighth harmonic,
        // which is the deviation SPEC.md's tuning test relies on.
        let sharpness = cents(8.0 * m.f0_hz, m.partial(8));
        assert!((sharpness - 21.9).abs() < 0.5, "{sharpness} cents");
    }

    #[test]
    fn partials_below_stops_at_the_limit_and_at_the_cap() {
        let m = InharmonicModel::harmonic(100.0);
        assert_eq!(m.partials_below(1050.0, 80), 10);
        assert_eq!(m.partials_below(1e9, 12), 12);
    }

    #[test]
    fn envelope_interpolation_never_extrapolates() {
        let track = PartialTrack {
            k: 1,
            points: vec![
                TrackPoint { time_s: 0.0, frequency_hz: 100.0, amplitude: 1.0 },
                TrackPoint { time_s: 1.0, frequency_hz: 101.0, amplitude: 0.5 },
            ],
        };
        assert_eq!(track.amplitude_at(0.5), Some(0.75));
        assert_eq!(track.frequency_at(0.0), Some(100.0));
        assert_eq!(track.amplitude_at(1.0), Some(0.5));
        assert_eq!(track.amplitude_at(-0.1), None);
        assert_eq!(track.amplitude_at(1.1), None);
    }

    #[test]
    fn trajectories_round_trip_through_json() {
        let traj = NoteTrajectories {
            source: "synthetic".into(),
            note: Some(NoteId::layer(60, 7)),
            sample_rate: 48_000.0,
            window_s: 0.5,
            hop_s: 0.01,
            seed: InharmonicModel::new(261.6, 4e-4),
            onset_s: 0.002,
            tracks: vec![PartialTrack {
                k: 1,
                points: vec![TrackPoint { time_s: 0.25, frequency_hz: 261.7, amplitude: 0.4 }],
            }],
        };
        let json = serde_json::to_string(&traj).unwrap();
        let back: NoteTrajectories = serde_json::from_str(&json).unwrap();
        assert_eq!(back.note, traj.note);
        assert_eq!(back.seed, traj.seed);
        assert_eq!(back.tracks[0].points[0], traj.tracks[0].points[0]);
    }
}
