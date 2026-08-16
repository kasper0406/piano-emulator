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
    let velocity_db = key_off_velocity_db(measurements, config)
        .unwrap_or(f64::from(base.key_off.velocity_db))
        .min(config.max_velocity_db);
    let key_off = fit_event(
        &measurements.key_off,
        &base.key_off,
        velocity_db,
        0.0,
        1.0,
        1.0,
        config,
    );
    // The lift is the fall, lighter and shorter — see the module header.
    let damper_lift = fit_event(
        &measurements.key_off,
        &base.damper_lift,
        velocity_db,
        config.damper_lift_db,
        config.damper_lift_decay,
        config.damper_lift_centroid,
        config,
    );
    let pedal_down = fit_event(
        &measurements.pedal_down,
        &base.pedal_down,
        f64::from(base.pedal_down.velocity_db),
        0.0,
        1.0,
        1.0,
        config,
    );
    let pedal_up = fit_event(
        &measurements.pedal_up,
        &base.pedal_up,
        f64::from(base.pedal_up.velocity_db),
        0.0,
        1.0,
        1.0,
        config,
    );
    NoiseTables {
        key_off,
        damper_lift,
        pedal_down,
        pedal_up,
        // The hammer's own noise is not a mechanism recording: no library
        // isolates a blow, so nothing here can measure it and `base`'s value —
        // silence, unless something already fitted one — stands.
        // `estimate::attack` is what fills it, from the struck notes themselves.
        strike: base.strike.clone(),
    }
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
