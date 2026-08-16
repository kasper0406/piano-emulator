//! `piano-tuner fit` — stage 2's per-note fits, selected by `--stage`.
//!
//! Two drivers live under one command because they fit **the same fields off
//! the same material** and one supersedes the other:
//! [`partials`] is the original per-partial half (`notes.comb_floor`,
//! `notes.partial_gains`, `notes.partial_sigma_scale`, `[noise.strike]`,
//! `notes.damper_sigma`) and [`motion`] re-fits `notes.partial_gains` as the
//! full measured ratio against a probe whose own row is cleared first, which is
//! what makes it re-entrant where `partials` is not (`DECISIONS.md` 231, 237,
//! 243). Putting them behind one `--stage` list is the only place a person can
//! see that ordering.
//!
//! ```text
//! piano-tuner fit <instrument.sfz> --preset <base.toml> [--out <file>]
//! ```
//!
//! With no `--stage`, the five **motion** stages run in order, which is the
//! re-entrant path and the one that builds `presets/salamander-c5.toml`:
//!
//! | stage | field |
//! |---|---|
//! | `false_beat` | `notes.false_beat` |
//! | `strike_direction` | `[voicing.strike_direction]` |
//! | `detune` | `notes.detune_cents`, where a beat still identifies it |
//! | `partial_gains` | `notes.partial_gains` |
//! | `texture` | both, **drawn** for the keys the library never sampled |
//!
//! `--stage partials` is the sixth and is **opt-in and alone**: it is not
//! re-entrant, its `--preset` and `--out` may not be the same file, and it must
//! be fitted from the survey base rather than from a preset that already
//! carries its answers (see [`partials`]'s own header). Asking for it in the
//! same invocation as a motion stage is refused rather than ordered, because
//! the two want different base presets.
//!
//! The other two stage-2 fits are separate commands, because they loop over
//! different material rather than over the sampled notes: `piano-tuner
//! sympathetic` (duplex, halo, stereo spread) and `piano-tuner tail` (the
//! upper partials' decay).

pub mod motion;
pub mod partials;

/// Splits the `--stage` list off and hands the rest to the driver that owns
/// the named stages. Every other flag is passed through untouched, so each
/// driver still parses exactly the flags its own header documents.
pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut stages: Vec<String> = Vec::new();
    let mut rest: Vec<String> = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--stage" {
            stages.push(args.next().ok_or("--stage needs a name")?);
        } else {
            rest.push(arg);
        }
    }

    const MOTION_STAGES: [&str; 5] = [
        "false_beat",
        "strike_direction",
        "detune",
        "partial_gains",
        "texture",
    ];
    if let Some(unknown) = stages
        .iter()
        .find(|s| s.as_str() != "partials" && !MOTION_STAGES.contains(&s.as_str()))
    {
        return Err(format!(
            "unknown stage {unknown:?}; stages are partials, {}",
            MOTION_STAGES.join(", ")
        )
        .into());
    }

    let wants_partials = stages.iter().any(|s| s == "partials");
    let motion_stages: Vec<&String> = stages.iter().filter(|s| s.as_str() != "partials").collect();
    if wants_partials && !motion_stages.is_empty() {
        return Err("`--stage partials` runs alone: it is not re-entrant and is fitted \
                    from the survey base, where the motion stages are fitted from the \
                    preset they are written into"
            .into());
    }
    if wants_partials {
        partials::run(rest)?;
        return Ok(());
    }

    let mut forwarded = rest;
    for stage in motion_stages {
        forwarded.push("--stage".to_string());
        forwarded.push(stage.clone());
    }
    motion::run(forwarded)
}
