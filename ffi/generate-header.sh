#!/bin/sh
# Regenerates include/piano_emulator.h from src/lib.rs.
#
# The header is committed so that the Swift side (DISTRIBUTION.md M2/M3) and the
# C harness build without a Rust toolchain step, and so that a change to the ABI
# shows up as a diff in review rather than as a surprise at link time.
# `ffi/tests/header.rs` regenerates it into a temporary file and fails if the
# committed one differs, so this script is the only way to move it.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v cbindgen >/dev/null 2>&1; then
    echo "cbindgen not found: cargo install cbindgen --version 0.28.0" >&2
    exit 1
fi

# Single-file mode (`src/lib.rs`) rather than crate mode (`--crate`): crate mode
# shells out to `cargo metadata`, which resolves the whole workspace for every
# platform and so fails on a Mac that has never downloaded the Linux-only audio
# backends. Everything the header exports is declared in `src/lib.rs` and
# cbindgen follows the `mod` declarations from there, so the two modes produce
# the same header when both can run.
cbindgen --config "$here/cbindgen.toml" \
    --output "$here/include/piano_emulator.h" "$here/src/lib.rs"
echo "wrote $here/include/piano_emulator.h"
