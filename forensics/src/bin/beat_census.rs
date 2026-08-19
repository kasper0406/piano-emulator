//! **Which cells share a beat rate**, by key and partial, on any two presets —
//! the census `engine/tests/partials.rs`'s
//! `no_beat_rate_is_shared_across_the_measured_presets_compass` takes, printed
//! rather than reduced to one number.
//!
//! That test asserts a *construction* property: no eigenmode beat rate may be
//! shared by more than one cell in fifty. It reads `presets/salamander-c5.toml`
//! and therefore moves whenever `notes.partial_sigma_scale` does, because the
//! damping is what splits a partial's modes. This is the instrument that says
//! *which* cells moved and why, which the assertion cannot.
//!
//! ```sh
//! cargo run --release -p forensics --bin beat_census -- \
//!     presets/salamander-c5.toml [other.toml]
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};

use piano_emulator::preset::Preset;
use piano_emulator::string::PianoString;

/// One millihertz is a beat period of a quarter of an hour: the test's own bin.
const SAME_HZ: f64 = 1.0e-3;
const RANGE: (f64, f64) = (0.05, 5.0);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let paths = if paths.is_empty() {
        vec!["presets/salamander-c5.toml".to_string()]
    } else {
        paths
    };
    let mut worst_sets: Vec<HashSet<(u8, usize)>> = Vec::new();
    for path in &paths {
        let preset = Preset::load(std::path::Path::new(path))?;
        let mut cells: Vec<(u8, usize, Vec<f64>)> = Vec::new();
        for key in 21..=108u8 {
            let string = PianoString::new(
                preset.string_params(key),
                &preset.voicing,
                preset.partial_shaping(key),
            );
            for k in 1..=string.partial_count() {
                let modes = string.partial_modes(k);
                let mut rates = Vec::new();
                for (i, a) in modes.iter().enumerate() {
                    for b in &modes[i + 1..] {
                        rates.push(f64::from((a.hz - b.hz).abs()));
                    }
                }
                cells.push((key, k, rates));
            }
        }
        let mut bins: HashMap<i64, HashSet<(u8, usize)>> = HashMap::new();
        for (key, k, rates) in &cells {
            for r in rates {
                if !(RANGE.0..RANGE.1).contains(r) {
                    continue;
                }
                for bin in [(r / SAME_HZ).floor() as i64, (r / SAME_HZ).ceil() as i64] {
                    bins.entry(bin).or_default().insert((*key, *k));
                }
            }
        }
        let (bin, set) = bins
            .iter()
            .max_by_key(|(_, c)| c.len())
            .map(|(b, c)| (*b, c.clone()))
            .unwrap_or((0, HashSet::new()));
        // Where they are: how the worst bin's cells fall by key and by partial.
        let mut by_key: BTreeMap<u8, usize> = BTreeMap::new();
        let mut by_k: BTreeMap<usize, usize> = BTreeMap::new();
        for &(key, k) in &set {
            *by_key.entry(key).or_default() += 1;
            *by_k.entry(k).or_default() += 1;
        }
        println!(
            "\n{path}\n  worst bin {:.3} Hz: {} of {} cells ({:.2} %), bar {} \
             (one in fifty)\n  by key: {}\n  by partial: {}",
            bin as f64 * SAME_HZ,
            set.len(),
            cells.len(),
            100.0 * set.len() as f64 / cells.len() as f64,
            cells.len() / 50,
            by_key
                .iter()
                .map(|(k, n)| format!("{k}:{n}"))
                .collect::<Vec<_>>()
                .join(" "),
            by_k
                .iter()
                .map(|(k, n)| format!("k{k}:{n}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        // How many cells sit near-degenerate at all, which is what feeds the
        // lowest bins: the count with any pairwise split under the bottom of
        // the counted range and just above it.
        let under = cells
            .iter()
            .filter(|(_, _, r)| r.iter().any(|x| *x < RANGE.0))
            .count();
        let just_over = cells
            .iter()
            .filter(|(_, _, r)| r.iter().any(|x| (RANGE.0..0.10).contains(x)))
            .count();
        println!("  cells with a split under {:.2} Hz: {under}; between {:.2} and 0.10 Hz: {just_over}",
            RANGE.0, RANGE.0);
        worst_sets.push(set);
    }
    if worst_sets.len() == 2 {
        let gained: Vec<(u8, usize)> = worst_sets[1]
            .difference(&worst_sets[0])
            .copied()
            .collect::<Vec<_>>();
        let mut gained = gained;
        gained.sort();
        println!("\ncells the second preset adds to the worst bin: {gained:?}");
    }
    Ok(())
}
