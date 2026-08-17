//! The header and the profile: two things that are only wrong at *link* time
//! or at *ship* time, which is far too late to find out.
//!
//! `include/piano_emulator.h` is committed so that the Swift of M2/M3 and the C
//! harness build without a Rust step, and so that an ABI change is a diff in
//! review. A committed generated file drifts unless something watches it, and
//! this is that something.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn source() -> String {
    std::fs::read_to_string(manifest_dir().join("src/lib.rs")).expect("src/lib.rs")
}

fn header() -> String {
    std::fs::read_to_string(manifest_dir().join("include/piano_emulator.h"))
        .expect("the committed header")
}

/// Every `extern "C"` function the crate exports, in source order.
fn exported_functions(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line
                .strip_prefix("pub extern \"C\" fn ")
                .or_else(|| line.strip_prefix("pub unsafe extern \"C\" fn "))?;
            Some(rest.split('(').next()?.to_string())
        })
        .collect()
}

/// The tool-free half: every exported symbol is declared, and every declaration
/// is an exported symbol. This runs everywhere, including on a machine with no
/// `cbindgen`, and it is the check that actually catches the mistake — a new
/// entry point added to the Rust and not regenerated into the header.
#[test]
fn the_header_declares_exactly_what_the_crate_exports() {
    let src = source();
    let header = header();
    let exports = exported_functions(&src);
    assert!(
        exports.len() >= 12,
        "only {} exported functions found — the parser has stopped working",
        exports.len()
    );
    for name in &exports {
        assert!(
            header.contains(&format!("{name}(")),
            "`{name}` is exported from src/lib.rs but is not in the header; \
             run ffi/generate-header.sh"
        );
    }
    // ... and nothing in the header that the library does not define, which is
    // the failure that only shows up as a link error in Xcode.
    for line in header.lines() {
        let Some(start) = line.find("pe_") else {
            continue;
        };
        if !line.trim_end().ends_with(';') || line.trim_start().starts_with('*') {
            continue;
        }
        let name: String = line[start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !line[start..].starts_with(&format!("{name}(")) {
            continue;
        }
        assert!(
            exports.contains(&name),
            "the header declares `{name}` but the crate does not export it"
        );
    }
    // The one struct the ABI passes by value, at the size the header states.
    assert_eq!(std::mem::size_of::<piano_emulator_ffi::pe_event_t>(), 16);
    assert_eq!(std::mem::align_of::<piano_emulator_ffi::pe_event_t>(), 4);
}

/// The strict half, when the tool that wrote the header is available: byte for
/// byte, or the committed file is stale.
///
/// Skipped rather than failed when `cbindgen` is missing or is a different
/// version, because its output is not stable across versions and a developer
/// without it should still be able to run the suite. The check above is the one
/// that never skips.
#[test]
fn the_committed_header_is_what_cbindgen_writes_today() {
    let version = Command::new("cbindgen").arg("--version").output();
    let Ok(version) = version else {
        println!("cbindgen not installed — skipping the byte-for-byte check");
        return;
    };
    let version = String::from_utf8_lossy(&version.stdout).trim().to_string();
    if version != "cbindgen 0.28.0" {
        println!("{version} is not the version the header was written with — skipping");
        return;
    }

    let out = std::env::temp_dir().join(format!("pe-header-{}.h", std::process::id()));
    let status = Command::new("cbindgen")
        .arg("--config")
        .arg(manifest_dir().join("cbindgen.toml"))
        .arg("--output")
        .arg(&out)
        .arg(manifest_dir().join("src/lib.rs"))
        .status()
        .expect("cbindgen runs");
    assert!(status.success(), "cbindgen failed");
    let generated = std::fs::read_to_string(&out).expect("the generated header");
    let _ = std::fs::remove_file(&out);
    let committed = header();
    if generated != committed {
        // Printing two 400-line headers helps nobody; print where they part.
        let at = generated
            .lines()
            .zip(committed.lines())
            .position(|(a, b)| a != b);
        let detail = match at {
            Some(i) => format!(
                "line {}:\n  generated: {}\n  committed: {}",
                i + 1,
                generated.lines().nth(i).unwrap_or(""),
                committed.lines().nth(i).unwrap_or("")
            ),
            None => format!(
                "the committed header is {} lines, cbindgen writes {}",
                committed.lines().count(),
                generated.lines().count()
            ),
        };
        panic!("include/piano_emulator.h is stale — run ffi/generate-header.sh\n{detail}");
    }
}

/// A Rust panic that unwound across the C ABI would be undefined behaviour, so
/// the shipped library must not be able to unwind at all. Cargo has no
/// per-crate-type panic setting, and putting `panic = "abort"` in
/// `[profile.release]` would change what the engine's own acceptance numbers are
/// measured under, so the shipped library gets its own profile.
///
/// This test is the only thing that would notice the profile being deleted,
/// because nothing in a normal build uses it.
#[test]
fn the_shipping_profile_aborts_on_panic() {
    let root = manifest_dir()
        .parent()
        .expect("workspace root")
        .to_path_buf();
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("workspace manifest");
    let dist = manifest
        .split("[profile.dist]")
        .nth(1)
        .expect("the workspace has no [profile.dist] — see DECISIONS.md 380");
    let dist = dist.split("\n[").next().expect("a section body");
    assert!(
        dist.contains("panic = \"abort\""),
        "[profile.dist] does not abort on panic:\n{dist}"
    );
    assert!(
        dist.contains("inherits = \"release\""),
        "[profile.dist] is not a release build:\n{dist}"
    );
    assert!(
        manifest.contains("members = [\"engine\", \"ffi\""),
        "the ffi crate is not a workspace member"
    );
    assert!(
        Path::new(&root).join("ffi/harness/render.c").exists(),
        "the C harness is missing"
    );
}
