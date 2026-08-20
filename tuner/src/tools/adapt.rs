//! `piano-tuner adapt` — turn a downloaded sample library into a measurement
//! input.
//!
//! Two jobs, both of them once per library rather than once per fit:
//!
//! 1. **Write the instrument definition** the library does not ship, from its
//!    [`LibrarySpec`](piano_tuner::adapter::LibrarySpec), over the files that
//!    are actually on disk. Everything downstream — `survey`, every `fit`
//!    stage, `sympathetic`, `tail`, `noise`, `mics`, `level` and all five
//!    boards — takes an SFZ path and does not care that this one was
//!    generated.
//! 2. **Bring the tree onto the engine's clock**, where the library was
//!    published at another rate. One offline pass of the crate's own
//!    band-limited sinc resampler, written to float WAV; after it the tree is
//!    a 48 kHz tree and the boundary resampler is not inside any subsequent
//!    measurement.
//!
//! Neither job can touch a library that ships a usable map: `adapt salamander`
//! refuses, by construction, because the shipped file is what every bar in
//! this repository was measured through.
//!
//! ```text
//! piano-tuner adapt <library-id> --root <dir> [--out <file.sfz>] [--resample]
//! piano-tuner adapt --list
//! ```

use std::path::{Path, PathBuf};

use piano_tuner::adapter::{resample_tree, write_legacy_alias, LibrarySpec, Source};

type Exit = std::result::Result<(), Box<dyn std::error::Error>>;

pub fn run(args: Vec<String>) -> Exit {
    let mut id: Option<String> = None;
    let mut root: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut resample = false;
    let mut legacy_alias = false;
    let mut list = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => list = true,
            "--resample" => resample = true,
            "--legacy-alias" => legacy_alias = true,
            "--root" => {
                i += 1;
                root = Some(PathBuf::from(args.get(i).ok_or("--root needs a directory")?));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(args.get(i).ok_or("--out needs a path")?));
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}").into()),
            other => id = Some(other.to_string()),
        }
        i += 1;
    }

    if list || id.is_none() {
        print_catalogue();
        return Ok(());
    }
    let id = id.expect("checked");
    let spec = LibrarySpec::find(&id)
        .ok_or_else(|| format!("no library called {id:?}; `adapt --list` names them"))?;
    let root = root.unwrap_or_else(|| PathBuf::from("data").join(&id));
    if !root.is_dir() {
        return Err(format!(
            "{}: not a directory — run the library's own data/fetch_*.sh first",
            root.display()
        )
        .into());
    }

    println!("{}: {}", spec.id, spec.instrument);
    println!("  credit  {}", spec.credit);
    println!("  licence {}", spec.licence);
    println!("  source  {}", spec.source_url);
    println!("  root    {}", root.display());

    if let Source::Shipped(name) = spec.source {
        println!(
            "\nthis library ships its own instrument definition ({name}) and is played from it.\n\
             Nothing is generated: every bar in this repository was measured through that file,\n\
             and a generated stand-in would silently re-bar them."
        );
        return Ok(());
    }

    if resample {
        if spec.is_native_rate() {
            println!(
                "\n{} Hz already; nothing to resample.",
                spec.delivered_rate_hz
            );
        } else {
            println!(
                "\nresampling {} Hz -> {} Hz (audio::resample, band-limited sinc, float WAV out)",
                spec.published_rate_hz, spec.delivered_rate_hz
            );
            let mut done = 0usize;
            let (converted, skipped) = resample_tree(
                &root,
                source_extension(spec),
                spec.delivered_rate_hz,
                |path| {
                    done += 1;
                    if done % 25 == 0 {
                        println!("  {done:4}  {}", short(path, &root));
                    }
                },
            )?;
            println!("  {converted} converted, {skipped} already in place");
        }
    }

    let scan = spec.scan(&root);
    let keys = scan.recorded_keys();
    println!(
        "\nscanned: {} of {} expected note recordings present, over {} keys x {} layers",
        scan.present_notes(),
        scan.notes.len(),
        keys.len(),
        spec.bands.count()
    );
    if scan.present_notes() < scan.notes.len() {
        let missing = scan.missing_notes();
        println!("  {} missing, first five:", missing.len());
        for note in missing.iter().take(5) {
            println!("    {}", note.relative);
        }
        if !spec.is_native_rate() && !resample {
            println!(
                "  (the description expects {} files; pass --resample to build them)",
                spec.extension
            );
        }
    }
    println!(
        "  mechanism: {} of {} present",
        scan.present_mechanism(),
        scan.mechanism.len()
    );

    let text = spec.emit_sfz(&root)?;
    let out = out.unwrap_or_else(|| root.join(format!("{}.sfz", spec.id)));
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, &text)?;
    println!(
        "\nwrote {} ({} regions, {} bytes)",
        out.display(),
        text.matches("<region>").count(),
        text.len()
    );

    if legacy_alias {
        // Scaffolding; see `adapter::write_legacy_alias`. Three drivers still
        // join Salamander's own filename to their data directory and are
        // another workstream's files, so until they adopt `instrument_path`
        // this is how `tail`, `mics` and `melody` can be pointed at a library
        // that is not Salamander at all.
        let alias = write_legacy_alias(&root, &out)?;
        println!(
            "\nlegacy alias: {} -> {}\n  SCAFFOLDING. tools/tail.rs, tools/mics.rs and \
             tools/melody.rs still join Salamander's filename to their data directory.\n  \
             Delete this the moment those three call adapter::instrument_path (DECISIONS.md 521).",
            alias.display(),
            out.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    if !spec.caveats.is_empty() {
        println!("\ncaveats carried into the generated file — read them before trusting a stage:");
        for (i, caveat) in spec.caveats.iter().enumerate() {
            println!("  {}. {}", i + 1, first_sentence(caveat));
        }
    }
    Ok(())
}

/// The extension the recordings arrive in, before any resampling. A library
/// delivered at a different rate from the one it was published at is one whose
/// files this tool rewrites, and it rewrites them to `wav`.
fn source_extension(spec: &LibrarySpec) -> &'static str {
    if spec.is_native_rate() {
        spec.extension
    } else {
        "flac"
    }
}

fn short(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn first_sentence(text: &str) -> String {
    let mut out = String::new();
    for word in text.split_whitespace() {
        out.push_str(word);
        if word.ends_with('.') && out.len() > 40 {
            break;
        }
        out.push(' ');
    }
    out.trim_end().to_string()
}

fn print_catalogue() {
    println!("libraries this repository has a description for:\n");
    for spec in LibrarySpec::all() {
        let source = match spec.source {
            Source::Shipped(name) => format!("ships {name}"),
            Source::Generated => "generated here".to_string(),
        };
        println!("  {}", spec.id);
        println!("    {}", spec.instrument);
        println!("    {}", spec.licence);
        println!(
            "    {} keys x {} layers, {} Hz{}, map: {source}",
            spec.layout.keys().len(),
            spec.bands.count(),
            spec.delivered_rate_hz,
            if spec.is_native_rate() {
                " native"
            } else {
                " (resampled at fetch)"
            }
        );
    }
    println!(
        "\nusage: piano-tuner adapt <id> --root <dir> [--out <file.sfz>] [--resample] [--legacy-alias]"
    );
}
