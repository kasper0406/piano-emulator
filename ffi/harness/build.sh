#!/bin/sh
# Builds the C harness against the ffi crate's dynamic library.
#
#   ./build.sh [lib-dir] [out-binary]
#
# `lib-dir` defaults to the workspace's `target/debug`, which is where
# `cargo build -p piano-emulator-ffi` puts `libpiano_emulator_ffi.dylib`; pass
# `target/release` for the release library. The dylib and not the staticlib
# because the static one drags every system framework `cpal` touches onto the
# link line, and the harness is a demonstration of the ABI rather than of
# macOS's linker.
#
# `ffi/tests/harness.rs` runs this script and then the binary, so a change that
# stops the C from compiling is a red test and not a surprise at M2.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)

lib_dir=${1:-$root/target/debug}
out=${2:-$here/render}

if [ ! -e "$lib_dir/libpiano_emulator_ffi.dylib" ] &&
    [ ! -e "$lib_dir/libpiano_emulator_ffi.so" ]; then
    echo "no libpiano_emulator_ffi in $lib_dir — run:" >&2
    echo "    cargo build -p piano-emulator-ffi" >&2
    exit 1
fi

cc=${CC:-cc}
"$cc" -std=c11 -O2 -Wall -Wextra -Wpedantic \
    -I "$here/../include" \
    -o "$out" "$here/render.c" \
    -L "$lib_dir" -lpiano_emulator_ffi \
    -Wl,-rpath,"$lib_dir" \
    -lm
echo "built $out against $lib_dir"
