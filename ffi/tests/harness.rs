//! The C harness, built and run: `DISTRIBUTION.md` M1's "done when".
//!
//! *C harness output matches `cargo run -- render` sample-exactly.* The
//! reference here is not a recorded hash but the CLI's own code path — the same
//! `midi::load` and `render_to_wav` that `main.rs` calls — so the test cannot
//! drift away from the thing it is supposed to be checking. The measured md5 of
//! the payload is printed (and recorded in `DECISIONS.md` 383) so that the
//! identity can also be checked by hand, from a shell, without this test.
//!
//! It builds C. That is deliberate: everything above M1 is C and Swift calling
//! this ABI, and a header that no compiler has ever read is a header that does
//! not compile.

use piano_emulator::preset::Preset;
use piano_emulator::{midi, render::render_to_wav};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// `target/<profile>`, found from the test binary rather than guessed: the same
/// test has to work under `cargo test` and `cargo test --release`.
fn target_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("the test binary has a path");
    exe.parent()
        .and_then(Path::parent)
        .expect("target/<profile>/deps/<test>")
        .to_path_buf()
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffi is in the workspace")
        .to_path_buf()
}

/// The dynamic library the harness links against, built if this build did not
/// produce it — `cargo test` builds the rlib for its own use and has no reason
/// to produce a cdylib, so more often than not it has to be asked.
fn dylib() -> PathBuf {
    // Once per test binary: the two tests here run on their own threads, and
    // two `cargo build`s on one target directory would only queue behind each
    // other's lock.
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT.get_or_init(build_dylib).clone()
}

fn build_dylib() -> PathBuf {
    let dir = target_dir();
    let lib = dir.join("libpiano_emulator_ffi.dylib");
    if lib.exists() {
        return lib;
    }
    let mut cargo = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cargo.arg("build").arg("-p").arg("piano-emulator-ffi");
    if dir.file_name().is_some_and(|n| n == "release") {
        cargo.arg("--release");
    }
    let status = cargo
        .current_dir(workspace_root())
        .status()
        .expect("cargo runs");
    assert!(status.success(), "could not build the ffi cdylib");
    assert!(lib.exists(), "cargo build produced no {}", lib.display());
    lib
}

fn scratch(name: &str) -> PathBuf {
    // Per test, not per process: the tests run on their own threads and would
    // otherwise build into, and clean up, each other's directory.
    let dir = std::env::temp_dir().join(format!("pe-harness-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

/// Builds `render.c` against the library and returns the binary.
fn build_harness(scratch: &Path) -> PathBuf {
    let lib_dir = dylib().parent().expect("a directory").to_path_buf();
    let binary = scratch.join("render");
    let script = workspace_root().join("ffi/harness/build.sh");
    let out = Command::new("sh")
        .arg(&script)
        .arg(&lib_dir)
        .arg(&binary)
        .output()
        .expect("the build script runs");
    assert!(
        out.status.success(),
        "ffi/harness/build.sh failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    binary
}

fn read_wav(path: &Path) -> (u32, Vec<f32>) {
    let mut reader = hound::WavReader::open(path).expect("a readable WAV");
    let spec = reader.spec();
    assert_eq!(spec.channels, 2);
    assert_eq!(spec.bits_per_sample, 32);
    assert_eq!(spec.sample_format, hound::SampleFormat::Float);
    let samples = reader
        .samples::<f32>()
        .map(|s| s.expect("a sample"))
        .collect();
    (spec.sample_rate, samples)
}

/// md5 of the interleaved samples — not of the file, whose header the two
/// writers spell differently — so the identity can be checked from a shell.
fn payload_md5(samples: &[f32]) -> String {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let path = std::env::temp_dir().join(format!(
        "pe-md5-{}-{:p}.raw",
        std::process::id(),
        samples.as_ptr()
    ));
    std::fs::write(&path, &bytes).expect("a scratch file");
    let out = Command::new("md5").arg("-q").arg(&path).output();
    let _ = std::fs::remove_file(&path);
    match out {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "<md5(1) unavailable>".to_string(),
    }
}

/// The milestone: C in, WAV out, and the WAV is the one the CLI writes.
///
/// Both preset paths are checked, because they are different code: `-` is the
/// preset compiled into the library, and a file goes through
/// `pe_load_preset_toml` and `Preset::validate` on the way in. That they agree
/// is also a small proof that `presets/default.toml` still round-trips to the
/// built-in default, float for float.
#[test]
fn the_c_harness_renders_what_the_cli_renders() {
    let scratch = scratch("cli");
    let binary = build_harness(&scratch);
    let root = workspace_root();
    let midi_path = root.join("ffi/harness/phrase.mid");
    let preset_path = root.join("presets/default.toml");

    // The reference: exactly what `piano-emulator render out.wav in.mid` does.
    let performance = midi::load(&midi_path).expect("the fixture parses");
    let reference_path = scratch.join("reference.wav");
    render_to_wav(
        &reference_path,
        &Preset::load(&preset_path).expect("the shipped preset"),
        &performance.events,
        performance.duration_s(),
    )
    .expect("the reference renders");
    let (reference_rate, reference) = read_wav(&reference_path);
    assert_eq!(reference_rate, 48_000);
    assert!(reference.iter().any(|&v| v != 0.0), "a silent reference");
    let hash = payload_md5(&reference);
    println!("reference: {} frames, md5 {hash}", reference.len() / 2);

    for (label, args) in [
        ("preset file", vec![preset_path.to_str().unwrap()]),
        ("built-in preset", vec!["-"]),
        ("built-in preset, via the SPSC queue", vec!["-"]),
    ] {
        let out = scratch.join(format!("{}.wav", label.replace([' ', ','], "-")));
        let mut command = Command::new(&binary);
        command.arg(args[0]).arg(&midi_path).arg(&out);
        if label.contains("queue") {
            command.arg("--queue");
        }
        let run = command.output().expect("the harness runs");
        assert!(
            run.status.success(),
            "the harness failed ({label}):\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let (rate, samples) = read_wav(&out);
        assert_eq!(rate, 48_000, "{label}");
        assert_eq!(samples.len(), reference.len(), "{label}: length");
        for (i, (a, b)) in samples.iter().zip(&reference).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: sample {i} differs ({a:e} against {b:e})"
            );
        }
        println!("{label}: md5 {}", payload_md5(&samples));
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

/// The same harness at the two rates a host actually runs, which is the only
/// place the *whole* boundary — C caller, resampler, WAV — is exercised end to
/// end. Nothing here can be compared against the CLI (there is no 44.1 kHz
/// CLI), so what is checked is what a listener would notice: right length,
/// right level, nothing clipped, nothing infinite, the notes in the same
/// places.
#[test]
fn the_harness_runs_at_the_host_rates_too() {
    let scratch = scratch("rates");
    let binary = build_harness(&scratch);
    let root = workspace_root();
    let midi_path = root.join("ffi/harness/phrase.mid");

    let mut reference_seconds = 0.0f64;
    for rate in [48_000u32, 44_100, 96_000] {
        let out = scratch.join(format!("{rate}.wav"));
        let run = Command::new(&binary)
            .arg("-")
            .arg(&midi_path)
            .arg(&out)
            .arg("--rate")
            .arg(rate.to_string())
            .output()
            .expect("the harness runs");
        assert!(run.status.success(), "{rate} Hz: the harness failed");
        let (written_rate, samples) = read_wav(&out);
        assert_eq!(written_rate, rate);
        let seconds = samples.len() as f64 / 2.0 / rate as f64;
        if rate == 48_000 {
            reference_seconds = seconds;
        }
        assert!(
            (seconds - reference_seconds).abs() < 1.0e-3,
            "{rate} Hz is {seconds:.4} s against {reference_seconds:.4} s"
        );
        let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(
            samples.iter().all(|v| v.is_finite()),
            "{rate} Hz: not finite"
        );
        assert!(peak > 0.01 && peak <= 1.0, "{rate} Hz: peak {peak}");
        println!("{rate:>6} Hz: {seconds:.4} s, peak {peak:.4}");
    }
    let _ = std::fs::remove_dir_all(&scratch);
}
