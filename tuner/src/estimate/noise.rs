//! The mechanism's own sounds, from a library's release and pedal recordings.
//!
//! `docs/history/TUNING_REPORT.md` §5 ranked this the cheapest item on its backlog because
//! nothing here needs fitting in the sense the rest of `estimate` does: the
//! recordings *are* the parameter set. Salamander ships one key-off sample per
//! key (`rel1`…`rel88`) and four pedal samples, the SFZ says what level they
//! play at, and three numbers come straight out of each recording —
//!
//! * **how loud**, as a peak relative to a velocity-90 strike of the same key,
//! * **how long**, as the time to fall 40 dB,
//! * **what colour**, as the power-weighted centroid of its first 100 ms —
//!
//! which are exactly the three the engine's `[noise]` section holds. All three
//! come from [`residual::transient_metrics`](crate::residual::transient_metrics),
//! the same measurement the report tabulated with.
//!
//! What this module does that reading the table off §5 does not: it applies the
//! SFZ's own `volume` to both sides of the comparison, so a library that
//! attenuates its key-off group by 37 dB is measured as it plays rather than as
//! it is stored; it reduces 88 per-key levels to the handful of compass anchors
//! the preset holds; and it says which of the four events were measured at all,
//! so the rest can be inherited instead of invented.
//!
//! # What is inherited rather than measured
//!
//! **The damper lift.** No library records one, because a damper lifting under
//! a hammer blow is inaudible — the sound only exists under a key pressed too
//! slowly to reach escapement (`PHYSICS.md` §6), where it is the whole sound.
//! It is derived from the measured key-off by the same offsets the engine's
//! default uses: 6 dB quieter, a third as long, and brighter, because felt
//! leaving a string is a lighter event than a key arriving at its rest.
//!
//! **The pedal's velocity law.** The SFZ gives the pedal groups no
//! `amp_veltrack`, and there is no pedal drive in the recordings to fit one
//! against; the base preset's figure stands.
//!
//! # What is refused rather than written
//!
//! Everything above assumes the recordings *are* the piano's mechanism. On two
//! of the three libraries in this repository they are not: they are the
//! editor's gain and the hall's reverb, and reading them as a mechanism writes
//! a key-off thump as loud as the note it belongs to. [`MAX_MECHANISM_LEVEL_DB`]
//! is the plausibility gate that refuses them, and a group that fails it is
//! **inherited from the base preset rather than written** — the same convention
//! `[noise.strike]` already had (`engine::preset`, "Fields a preset may leave
//! out"): a mechanism table at the base's own value is one nobody measured on
//! this piano. `DECISIONS.md` 531.

use crate::preset::{EventNoise, NoiseAnchor, NoiseTables, HIGHEST_KEY, LOWEST_KEY};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseConfig {
    /// The strike a mechanism level is quoted against — the layer a mezzo-forte
    /// blow would trigger, which is what these noises are heard against in
    /// playing, and the reference §5's table uses.
    pub reference_velocity: u8,
    /// Keys between compass anchors. The preset interpolates between them, and
    /// the measured levels are not smooth — neighbouring octaves differ by
    /// 10 dB — so each anchor is the median of the keys around it rather than
    /// one key's own measurement.
    pub anchor_step: u8,
    /// Ceiling on the velocity law written into the preset, in dB.
    ///
    /// The SFZ's `amp_veltrack` tracks the velocity of the *note-on* that
    /// started the release sample; the engine's `velocity_db` tracks the
    /// *release* velocity. They are different gestures, and a law measured on
    /// one is a bound on the other rather than a value for it — Salamander's
    /// 82 % over its own recorded dynamic range comes to nearly 40 dB, which no
    /// key release spans. The ceiling is a judgement, and it is the only one in
    /// this module.
    pub max_velocity_db: f64,
    /// Level of the damper lift relative to the key-off of the same key, dB.
    pub damper_lift_db: f64,
    /// Its decay as a fraction of the key-off's, and its centroid as a multiple.
    pub damper_lift_decay: f64,
    pub damper_lift_centroid: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            reference_velocity: 90,
            anchor_step: 12,
            max_velocity_db: 24.0,
            damper_lift_db: -6.0,
            damper_lift_decay: 1.0 / 3.0,
            damper_lift_centroid: 1.6,
        }
    }
}

/// Engine bounds a written event has to be inside — `engine::preset`'s own,
/// mirrored in `crate::preset`.
const MIN_DECAY_S: f64 = 0.01;
const MAX_DECAY_S: f64 = 10.0;
const MIN_CENTROID_HZ: f64 = 1.0;
const MAX_CENTROID_HZ: f64 = 0.45 * 48_000.0;

/// The hottest a mechanism recording may sit against the note it belongs to and
/// still be a measurement of a **piano** rather than of an editor, in dB.
///
/// Every reading here is a peak relative to a velocity-90 strike of the same
/// key with both sides at the level the instrument plays them, so the number is
/// directly comparable across libraries and directly comparable to the
/// literature. Two anchors bound it, and they agree:
///
/// * **Measured, in this repository.** The only mechanism group here that is
///   demonstrably the piano's own is Salamander's `//HammerNoise` (`volume=-37`
///   in its own SFZ, `PHYSICS.md` §5). Run through
///   [`measure_mechanism`](crate::survey::measure_mechanism) its **88** key-off
///   recordings read **−39.0 dB at the quietest key and −24.64 dB at the
///   hottest**, median **−32.9**; the anchors that reduction writes into
///   `presets/salamander-c5.toml` run −37.08 to −30.06. Its two pedal-down
///   takes read −35.8 and −40.8, its two pedal-up takes −42.4 and −48.8.
/// * **Measured, in the literature.** Askenfelt removed C4's strings and
///   replaced them with a dummy mass to record the structure-borne path alone
///   (STL-QPSR 34(4) 15, 1993; `PHYSICS.md` §5): the mechanism sits **~40 dB
///   below the string partials**, and the touch precursor 30–40 dB below the
///   first transversal wave. `PHYSICS.md`'s own build note says the same thing
///   as a specification — "band-limited to ~2 kHz and ~40 dB under the
///   partials".
///
/// So a genuine mechanism noise lives in roughly −39 … −25 dB and neither
/// source comes within twenty decibels of its own note. The gate is the hottest
/// genuine reading measured here plus **three take-to-take sigmas** of the
/// library it was measured on (1.40 dB, `DECISIONS.md` 457):
/// −24.64 + 3 × 1.40 = **−20.44**, taken to the whole decibel on the measured
/// side. Three sigmas rather than a margin chosen for comfort, so that
/// re-measuring Salamander on a different take cannot rail its own table; the
/// whole decibel on the measured side rather than the loose one, so that
/// rounding can only refuse more.
///
/// It is a **floor under an absurdity**, not a fit target: nothing between −39
/// and −21 dB is judged by it at all, and every reading Salamander offers
/// clears it by 3.6 dB or more. What it catches is the class `DECISIONS.md` 528
/// named on the bitKlavier grand and 531 found still written into the preset —
/// a key-off group published at the editor's own gain with the hall on it,
/// which reads **−14.9 to +10.87 dB** against its own note, median −3.25. Not
/// one of its 88 recordings is under −21, and **21 of them are louder than the
/// note they belong to**: a damper landing louder than the chord.
pub const MAX_MECHANISM_LEVEL_DB: f64 = -21.0;

/// The share of a group's readings that has to survive [`MAX_MECHANISM_LEVEL_DB`]
/// for the group to be written at all.
///
/// A **strict majority**, and the reason is not statistical taste: the
/// per-anchor reduction is already a median, so it survives a few broken files
/// on its own (`eighty_eight_measured_keys_become_a_handful_of_anchors` puts one
/// key 20 dB out and the anchor does not move). What a median cannot survive is
/// a group that is hot *as a group*. Once half or more of a library's mechanism
/// recordings are implausible, the ones that pass are that group's quiet tail —
/// the takes whose own note happened to be loud — and a table drawn from them
/// is a measurement of the tail, not of the mechanism.
///
/// Strict, because the case that decides it is a two-take pedal group: the
/// bitKlavier grand's two pedal-down takes read −33.4 and −20.2 dB, and there is
/// no majority in one of two. A median of one surviving take is a coin toss
/// dressed as a measurement.
pub const MIN_PLAUSIBLE_SHARE: f64 = 0.5;

/// What the plausibility gate found in one mechanism group.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Screening {
    /// Recordings the library offered and that decoded.
    pub read: usize,
    /// How many of them sit at or under [`MAX_MECHANISM_LEVEL_DB`].
    pub kept: usize,
    /// The hottest reading in the group, dB against its own note. `NaN` when
    /// the group is empty.
    pub hottest_db: f64,
}

impl Screening {
    /// Whether the library recorded this event at all.
    pub fn recorded(&self) -> bool {
        self.read > 0
    }

    /// Whether the group may be written into a preset.
    pub fn accepted(&self) -> bool {
        self.read > 0 && self.kept as f64 > MIN_PLAUSIBLE_SHARE * self.read as f64
    }

    /// Why a group was refused, in one line, or `None` when it was not.
    pub fn refusal(&self) -> Option<String> {
        (self.recorded() && !self.accepted()).then(|| {
            format!(
                "{} of {} readings sit at or under {MAX_MECHANISM_LEVEL_DB:.1} dB against \
                 their own note (hottest {:+.2} dB)",
                self.kept, self.read, self.hottest_db
            )
        })
    }
}

/// The gate's verdict on a whole library. `damper_lift` is not here because it
/// is derived from the key-off recordings and carries their verdict.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoiseScreening {
    pub key_off: Screening,
    pub pedal_down: Screening,
    pub pedal_up: Screening,
}

impl NoiseScreening {
    /// The events by the name they carry in the preset, key-off's verdict
    /// standing for the damper lift it is derived from.
    pub fn events(&self) -> [(&'static str, Screening); 4] {
        [
            ("key_off", self.key_off),
            ("damper_lift", self.key_off),
            ("pedal_down", self.pedal_down),
            ("pedal_up", self.pedal_up),
        ]
    }

    /// The events that were recorded, read, and refused.
    pub fn refused(&self) -> Vec<&'static str> {
        self.events()
            .into_iter()
            .filter(|(_, s)| s.recorded() && !s.accepted())
            .map(|(name, _)| name)
            .collect()
    }

    /// A preset's `description` with this verdict on it, replacing any earlier
    /// one.
    ///
    /// Idempotent, because the mechanism stage is re-entrant and a description
    /// that grew a clause per run would be a log rather than a statement. The
    /// clause is always last and always starts [`PROVENANCE_MARKER`], so
    /// removing the old one is removing a suffix.
    pub fn describe(&self, description: &str) -> String {
        let kept = match description.find(PROVENANCE_MARKER) {
            Some(at) => &description[..at],
            None => description,
        };
        match self.provenance() {
            Some(clause) => format!("{kept}{PROVENANCE_MARKER}{clause}"),
            None => kept.to_string(),
        }
    }

    /// The sentence a preset's `description` carries when a table was refused,
    /// which is where this repository records that a value is not a
    /// measurement of the piano the preset names.
    pub fn provenance(&self) -> Option<String> {
        let refused = self.refused();
        if refused.is_empty() {
            return None;
        }
        let named: Vec<String> = refused.iter().map(|n| format!("[noise.{n}]")).collect();
        let list = match named.split_last() {
            Some((last, [])) => last.clone(),
            Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
            None => unreachable!("refused is not empty"),
        };
        Some(format!(
            "{list} {} — its mechanism recordings read hotter than \
             {MAX_MECHANISM_LEVEL_DB:.1} dB against their own notes, which is the \
             library's editorial gain and its room rather than the action \
             (DECISIONS.md 531)",
            if named.len() == 1 {
                "is INHERITED from presets/default.toml and is not a measurement of this piano"
            } else {
                "are INHERITED from presets/default.toml and are not measurements of this piano"
            },
        ))
    }
}

/// What [`NoiseScreening::describe`] writes its clause behind, and finds an
/// earlier one by. It has to be a string nothing else in a description can
/// produce, because removing the old clause is removing everything after it.
pub const PROVENANCE_MARKER: &str = "; mechanism noise: ";

/// One group, screened: the readings that are a mechanism, and the verdict.
///
/// A reading is refused when it does not decode to a finite level, or when it
/// sits hotter than [`MAX_MECHANISM_LEVEL_DB`] against its own note — which
/// includes the two ways the old code hid the same defect, a reading that
/// **inverts** (louder than the note, a positive level) and one that **rails**
/// (clamped to 0 dB on the way into the preset by [`clamp_db`]).
pub fn screen(metrics: &[EventMetrics]) -> (Vec<EventMetrics>, Screening) {
    let read: Vec<EventMetrics> = metrics
        .iter()
        .copied()
        .filter(|m| m.level_db.is_finite())
        .collect();
    let kept: Vec<EventMetrics> = read
        .iter()
        .copied()
        .filter(|m| m.level_db <= MAX_MECHANISM_LEVEL_DB)
        .collect();
    let hottest_db = read
        .iter()
        .map(|m| m.level_db)
        .fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a });
    let screening = Screening {
        read: read.len(),
        kept: kept.len(),
        hottest_db,
    };
    (
        if screening.accepted() {
            kept
        } else {
            Vec::new()
        },
        screening,
    )
}

/// One mechanism recording, measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventMetrics {
    /// The key this recording belongs to, where it has one.
    pub key: Option<u8>,
    /// Peak level relative to a strike of the reference velocity, in dB, with
    /// both sides at the level the instrument plays them.
    pub level_db: f64,
    /// Time to fall 40 dB, seconds.
    pub decay_s: f64,
    /// Power-weighted centroid of the first 100 ms, Hz.
    pub centroid_hz: f64,
    /// The key whose strike the level was measured against. Not always the
    /// recording's own: a library samples a subset of the compass and ships a
    /// key-off for every key.
    pub reference_key: u8,
}

/// Everything a library said about its mechanism.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MechanismMeasurements {
    pub key_off: Vec<EventMetrics>,
    pub pedal_down: Vec<EventMetrics>,
    pub pedal_up: Vec<EventMetrics>,
    /// The key-off group's `amp_veltrack`, as a percentage.
    pub key_off_veltrack: Option<f64>,
    /// The MIDI velocities the library's layers span, which is the range the
    /// tracking law is read over.
    pub velocity_span: Option<(u8, u8)>,
}

impl MechanismMeasurements {
    pub fn is_empty(&self) -> bool {
        self.key_off.is_empty() && self.pedal_down.is_empty() && self.pedal_up.is_empty()
    }
}

/// The `[noise]` section these measurements describe, over a base preset's own.
///
/// Every event the library did not record keeps `base`'s: a library with no
/// pedal samples in it leaves the pedal alone rather than writing silence, and
/// one with nothing at all returns `base` unchanged.
pub fn fit_noise(
    measurements: &MechanismMeasurements,
    base: &NoiseTables,
    config: &NoiseConfig,
) -> NoiseTables {
    fit_noise_screened(measurements, base, config).0
}

/// [`fit_noise`], with the plausibility gate's verdict beside the tables.
///
/// The verdict is what a caller writes into the preset's `description` and
/// prints: a table this returns unchanged from `base` is one the library did
/// not record **or** one it recorded implausibly, and those are different
/// facts about the same piano.
pub fn fit_noise_screened(
    measurements: &MechanismMeasurements,
    base: &NoiseTables,
    config: &NoiseConfig,
) -> (NoiseTables, NoiseScreening) {
    let velocity_db = key_off_velocity_db(measurements, config)
        .unwrap_or(f64::from(base.key_off.velocity_db))
        .min(config.max_velocity_db);
    // Screened once per *group*, not once per table: the damper lift is
    // derived from the key-off recordings, so refusing them refuses it.
    let (key_off_metrics, key_off_screen) = screen(&measurements.key_off);
    let (pedal_down_metrics, pedal_down_screen) = screen(&measurements.pedal_down);
    let (pedal_up_metrics, pedal_up_screen) = screen(&measurements.pedal_up);
    let key_off = fit_event(
        &key_off_metrics,
        &base.key_off,
        velocity_db,
        0.0,
        1.0,
        1.0,
        config,
    );
    // The lift is the fall, lighter and shorter — see the module header.
    let damper_lift = fit_event(
        &key_off_metrics,
        &base.damper_lift,
        velocity_db,
        config.damper_lift_db,
        config.damper_lift_decay,
        config.damper_lift_centroid,
        config,
    );
    let pedal_down = fit_event(
        &pedal_down_metrics,
        &base.pedal_down,
        f64::from(base.pedal_down.velocity_db),
        0.0,
        1.0,
        1.0,
        config,
    );
    let pedal_up = fit_event(
        &pedal_up_metrics,
        &base.pedal_up,
        f64::from(base.pedal_up.velocity_db),
        0.0,
        1.0,
        1.0,
        config,
    );
    let tables = NoiseTables {
        key_off,
        damper_lift,
        pedal_down,
        pedal_up,
        // The hammer's own noise is not a mechanism recording: no library
        // isolates a blow, so nothing here can measure it and `base`'s value —
        // silence, unless something already fitted one — stands.
        // `estimate::attack` is what fills it, from the struck notes themselves.
        strike: base.strike.clone(),
    };
    (
        tables,
        NoiseScreening {
            key_off: key_off_screen,
            pedal_down: pedal_down_screen,
            pedal_up: pedal_up_screen,
        },
    )
}

/// The dB the SFZ's own tracking law puts between the softest and loudest
/// velocity the library was recorded at.
///
/// SFZ's `amp_veltrack` is a percentage of the full `40 log10(v / 127)` law, so
/// the span between two velocities is that fraction of `40 log10(hi / lo)`.
pub fn key_off_velocity_db(
    measurements: &MechanismMeasurements,
    config: &NoiseConfig,
) -> Option<f64> {
    let veltrack = measurements.key_off_veltrack?;
    let (lo, hi) = measurements.velocity_span?;
    if lo == 0 || hi <= lo {
        return None;
    }
    let span = 40.0 * (f64::from(hi) / f64::from(lo)).log10();
    Some((veltrack / 100.0 * span).clamp(0.0, config.max_velocity_db))
}

/// One event: the medians of what was measured, or the base event where nothing
/// was.
fn fit_event(
    metrics: &[EventMetrics],
    base: &EventNoise,
    velocity_db: f64,
    level_offset_db: f64,
    decay_scale: f64,
    centroid_scale: f64,
    config: &NoiseConfig,
) -> EventNoise {
    if metrics.is_empty() {
        return base.clone();
    }
    let decay_s = median(metrics.iter().map(|m| m.decay_s))
        .map(|d| d * decay_scale)
        .filter(|d| d.is_finite())
        .unwrap_or(f64::from(base.decay_s))
        .clamp(MIN_DECAY_S, MAX_DECAY_S);
    let centroid_hz = median(metrics.iter().map(|m| m.centroid_hz))
        .map(|c| c * centroid_scale)
        .filter(|c| c.is_finite())
        .unwrap_or(f64::from(base.centroid_hz))
        .clamp(MIN_CENTROID_HZ, MAX_CENTROID_HZ);
    let levels: Vec<(Option<u8>, f64)> = metrics.iter().map(|m| (m.key, m.level_db)).collect();
    let level_db = compass_anchors(&levels, level_offset_db, config)
        .unwrap_or_else(|| base.level_db.clone());
    EventNoise {
        centroid_hz: centroid_hz as f32,
        decay_s: decay_s as f32,
        velocity_db: velocity_db as f32,
        level_db,
    }
}

/// The measured levels reduced to compass anchors.
///
/// A global event — one with no key on any of its recordings — is one anchor at
/// the bottom of the compass, which is how the preset spells "the same
/// everywhere". A per-key event is one anchor every
/// [`NoiseConfig::anchor_step`] keys, each the median of the keys nearest it,
/// with the first and last measured keys always anchored so that the
/// interpolation is never an extrapolation.
///
/// `levels` is `(key, dB)` with `None` for a global event. Public because the
/// hammer's own noise ([`estimate::attack`](crate::estimate::attack)) is
/// measured from the struck notes rather than from a mechanism recording, and
/// reduces its thirty measured keys to anchors by exactly this rule.
pub fn compass_anchors(
    levels: &[(Option<u8>, f64)],
    offset_db: f64,
    config: &NoiseConfig,
) -> Option<Vec<NoiseAnchor>> {
    let keyed: Vec<(u8, f64)> = levels
        .iter()
        .filter_map(|&(key, db)| Some((key?, db)))
        .filter(|(_, db)| db.is_finite())
        .collect();
    if keyed.is_empty() {
        let level = median(levels.iter().map(|&(_, db)| db).filter(|db| db.is_finite()))?;
        return Some(vec![NoiseAnchor {
            key: LOWEST_KEY,
            db: clamp_db(level + offset_db),
        }]);
    }
    let step = i32::from(config.anchor_step.max(1));
    let lowest = keyed.iter().map(|&(key, _)| key).min()?;
    let highest = keyed.iter().map(|&(key, _)| key).max()?;
    let mut centres: Vec<u8> = Vec::new();
    let mut key = i32::from(lowest);
    while key <= i32::from(highest) {
        centres.push(key as u8);
        key += step;
    }
    if centres.last() != Some(&highest) {
        centres.push(highest);
    }
    let half = (step + 1) / 2;
    let mut anchors: Vec<NoiseAnchor> = Vec::with_capacity(centres.len());
    for centre in centres {
        let near = keyed
            .iter()
            .filter(|&&(key, _)| i32::from(key.abs_diff(centre)) <= half)
            .map(|&(_, db)| db);
        let Some(level) = median(near) else { continue };
        anchors.push(NoiseAnchor {
            key: centre.clamp(LOWEST_KEY, HIGHEST_KEY),
            db: clamp_db(level + offset_db),
        });
    }
    // The highest measured key is always anchored, so it is in the list twice
    // whenever the step happens to land on it; anchors must ascend strictly.
    anchors.dedup_by_key(|anchor| anchor.key);
    (!anchors.is_empty()).then_some(anchors)
}

/// A mechanism event louder than the note it belongs to is a broken
/// measurement, and the engine refuses one.
///
/// **This clamp is a backstop and was never a gate.** Before `DECISIONS.md` 531
/// it was the only thing standing between an implausible reading and the
/// preset, and what it did with one was *round it to the rail* — which is how
/// `presets/concert-grand-d.toml` came to ship a key-off at exactly 0.0 dB, a
/// damper landing as loud as the note. [`screen`] refuses the reading now; this
/// still holds the arithmetic legal for the derived damper lift, whose −6 dB
/// offset is applied after the anchor is taken.
fn clamp_db(db: f64) -> f32 {
    if db.is_finite() {
        db.min(0.0) as f32
    } else {
        0.0
    }
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
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

    fn metrics(key: Option<u8>, level_db: f64, decay_s: f64, centroid_hz: f64) -> EventMetrics {
        EventMetrics {
            key,
            level_db,
            decay_s,
            centroid_hz,
            reference_key: key.unwrap_or(60),
        }
    }

    /// The §5 table itself: five key-off recordings and the two pedal ones.
    fn measured() -> MechanismMeasurements {
        MechanismMeasurements {
            key_off: vec![
                metrics(Some(21), -37.3, 0.165, 166.0),
                metrics(Some(57), -30.2, 0.245, 187.0),
                metrics(Some(60), -35.4, 0.265, 192.0),
                metrics(Some(72), -25.4, 0.210, 143.0),
                metrics(Some(96), -33.5, 0.285, 255.0),
            ],
            pedal_down: vec![metrics(None, -35.8, 5.76, 77.0)],
            pedal_up: vec![metrics(None, -42.4, 0.320, 187.0)],
            key_off_veltrack: Some(82.0),
            velocity_span: Some((8, 120)),
        }
    }

    #[test]
    fn the_measured_table_comes_back_as_the_preset_the_report_wrote_by_hand() {
        let base = NoiseTables::default();
        let fitted = fit_noise(&measured(), &base, &NoiseConfig::default());
        // The colour and the length are the medians of the same recordings the
        // hand-written default was rounded from.
        assert!((fitted.key_off.centroid_hz - 187.0).abs() < 1.0, "{fitted:?}");
        assert!((fitted.key_off.decay_s - 0.245).abs() < 1e-3, "{fitted:?}");
        assert!((fitted.pedal_down.decay_s - 5.76).abs() < 1e-3);
        assert!((fitted.pedal_up.centroid_hz - 187.0).abs() < 1e-3);
        // The pedal is global: one anchor, and it is the level of the file.
        assert_eq!(fitted.pedal_down.level_db.len(), 1);
        assert!((fitted.pedal_down.level_db[0].db + 35.8).abs() < 1e-3);
        // The key-off's anchors span the measured keys and stay inside them.
        let anchors = &fitted.key_off.level_db;
        assert_eq!(anchors.first().map(|a| a.key), Some(21));
        assert_eq!(anchors.last().map(|a| a.key), Some(96));
        assert!(anchors.windows(2).all(|w| w[0].key < w[1].key));
        assert!(anchors.iter().all(|a| a.db <= 0.0));
        // The lift is the fall, offset — it was never recorded.
        assert!(
            (fitted.damper_lift.level_db[0].db - (fitted.key_off.level_db[0].db - 6.0)).abs()
                < 1e-3
        );
        assert!((fitted.damper_lift.decay_s * 3.0 - fitted.key_off.decay_s).abs() < 1e-3);
        assert!(fitted.damper_lift.centroid_hz > fitted.key_off.centroid_hz);
    }

    #[test]
    fn the_velocity_law_is_the_sfzs_own_under_the_ceiling() {
        let config = NoiseConfig::default();
        let mut measurements = measured();
        // 82 % of 40 log10(120/8) is 38.6 dB, which is the note-on law and not
        // the release one: what is written is the ceiling.
        let raw = 0.82 * 40.0 * (120.0f64 / 8.0).log10();
        assert!(raw > config.max_velocity_db);
        assert_eq!(
            key_off_velocity_db(&measurements, &config),
            Some(config.max_velocity_db)
        );
        // A gentler library writes what it says.
        measurements.key_off_veltrack = Some(20.0);
        let expected = 0.2 * 40.0 * (120.0f64 / 8.0).log10();
        let fitted = key_off_velocity_db(&measurements, &config).unwrap();
        assert!((fitted - expected).abs() < 1e-9, "{fitted}");
        // And a library that says nothing leaves the base preset's figure.
        measurements.key_off_veltrack = None;
        assert_eq!(key_off_velocity_db(&measurements, &config), None);
        let base = NoiseTables::default();
        let table = fit_noise(&measurements, &base, &config);
        assert_eq!(table.key_off.velocity_db, base.key_off.velocity_db);
    }

    #[test]
    fn an_event_the_library_never_recorded_is_inherited_and_not_invented() {
        let base = NoiseTables::default();
        let measurements = MechanismMeasurements {
            key_off: measured().key_off,
            ..MechanismMeasurements::default()
        };
        let fitted = fit_noise(&measurements, &base, &NoiseConfig::default());
        assert_eq!(fitted.pedal_down, base.pedal_down);
        assert_eq!(fitted.pedal_up, base.pedal_up);
        assert_ne!(fitted.key_off, base.key_off);
        // Nothing at all is the base table, exactly.
        assert_eq!(
            fit_noise(
                &MechanismMeasurements::default(),
                &base,
                &NoiseConfig::default()
            ),
            base
        );
    }

    /// The gate's own arithmetic, at its two edges and on the group that
    /// derives from another (`DECISIONS.md` 531).
    #[test]
    fn a_reading_hotter_than_its_own_note_is_not_a_mechanism() {
        // The boundary is inclusive on the plausible side, so a library
        // measured at exactly the constant is written.
        let at = metrics(Some(60), MAX_MECHANISM_LEVEL_DB, 0.24, 190.0);
        let over = metrics(Some(60), MAX_MECHANISM_LEVEL_DB + 0.01, 0.24, 190.0);
        assert_eq!(screen(&[at]).1.kept, 1);
        assert_eq!(screen(&[over]).1.kept, 0);
        // Not finite is not read at all — it is neither kept nor a refusal.
        assert_eq!(screen(&[metrics(Some(60), f64::NAN, 0.24, 190.0)]).1.read, 0);

        // Salamander's own table passes untouched, and says nothing.
        let base = NoiseTables::default();
        let config = NoiseConfig::default();
        let (fitted, screening) = fit_noise_screened(&measured(), &base, &config);
        assert!(screening.refused().is_empty(), "{screening:?}");
        assert_eq!(screening.provenance(), None);
        assert_eq!(fitted, fit_noise(&measured(), &base, &config));
        assert!((screening.key_off.hottest_db + 25.4).abs() < 1e-9, "{screening:?}");

        // A key-off group hot enough to be refused takes the damper lift with
        // it, because the lift is derived from those same recordings — and it
        // leaves the pedals alone, because they are a different group.
        let mut hot = measured();
        for m in &mut hot.key_off {
            m.level_db += 30.0;
        }
        let (fitted, screening) = fit_noise_screened(&hot, &base, &config);
        assert_eq!(fitted.key_off, base.key_off);
        assert_eq!(fitted.damper_lift, base.damper_lift);
        // (§5's pedal reading *is* the default's, so "accepted" is the
        // assertion here and not "different from the base".)
        assert!(screening.pedal_down.accepted(), "{screening:?}");
        assert!(screening.pedal_up.accepted(), "{screening:?}");
        assert_eq!(screening.refused(), vec!["key_off", "damper_lift"]);
        let described = screening.describe("from somewhere");
        assert!(described.starts_with("from somewhere; mechanism noise: "), "{described}");
        assert_eq!(screening.describe(&described), described);
        // And a later run that finds nothing wrong takes the clause back off.
        let (_, clean) = fit_noise_screened(&measured(), &base, &config);
        assert_eq!(clean.describe(&described), "from somewhere");
    }

    #[test]
    fn eighty_eight_measured_keys_become_a_handful_of_anchors() {
        let config = NoiseConfig::default();
        // A level that falls smoothly across the compass, with one key 20 dB
        // out — a recording that clipped, or a file that is not what it says.
        let key_off: Vec<EventMetrics> = (21..=108u8)
            .map(|key| {
                let mut level = -40.0 + f64::from(key - 21) * 0.15;
                if key == 64 {
                    level -= 20.0;
                }
                metrics(Some(key), level, 0.24, 190.0)
            })
            .collect();
        let fitted = fit_noise(
            &MechanismMeasurements {
                key_off,
                ..MechanismMeasurements::default()
            },
            &NoiseTables::default(),
            &config,
        );
        let anchors = &fitted.key_off.level_db;
        assert!(anchors.len() <= 9, "{} anchors", anchors.len());
        assert!(anchors.windows(2).all(|w| w[0].key < w[1].key));
        assert_eq!(anchors.first().map(|a| a.key), Some(21));
        assert_eq!(anchors.last().map(|a| a.key), Some(108));
        // The outlier is a median away from the anchor it sits under.
        let near = anchors
            .iter()
            .find(|a| a.key == 69)
            .expect("an anchor at 69");
        assert!(
            (f64::from(near.db) - (-40.0 + 48.0 * 0.15)).abs() < 1.0,
            "{near:?}"
        );
        assert!(anchors.iter().all(|a| a.db <= 0.0));
    }
}
