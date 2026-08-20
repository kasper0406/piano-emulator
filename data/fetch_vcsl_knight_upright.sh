#!/bin/sh
# Fetch the Knight upright from the Versilian Community Sample Library — the
# upright of the shipped range, and the source of presets/upright-parlour.toml.
#
#   VCSL, "Upright Piano, Knight": a Knight upright (Alfred Knight Ltd,
#   London), recorded mid-close and acoustically neutral per VCSL's own
#   contribution standard, at 44.1 kHz / 24 bit, stereo.
#
#   45 keys across the full 88-key compass: whole tones from A0 (MIDI 21) up
#   to MIDI 107, plus C8 (MIDI 108), which the whole-tone series misses by one
#   and which the library recorded anyway. Two velocity layers per key (vl1,
#   vl2), un-normalized: measured 10.5 dB peak / 9.3 dB RMS apart at A3, and
#   unlike VCSL's Steinway B 1895 folder there is no NORMALIZED.txt here. 45
#   release recordings, 8 pedal-mechanism recordings (PedOn/PedOff x4).
#
#   Author:  Versilian Studios LLC (Sam Gossner); the Knight upright was
#            sampled by Simon Dalzell of Ivy Audio for the VSCO 2 Pro set.
#   Licence: Creative Commons Zero 1.0 Universal (CC0 1.0)
#            https://creativecommons.org/publicdomain/zero/1.0/
#            The repository's LICENSE file is the CC0 1.0 legal code verbatim
#            (https://raw.githubusercontent.com/sgossner/VCSL/master/LICENSE)
#            and its README states: "This collection is under a Creative
#            Commons 0 license. Essentially it's Public Domain- you can do
#            whatever you want with these sounds (even make commercial
#            software), no royalties, no credit, no special terms."
#   Source:  https://github.com/sgossner/VCSL
#   Bundle:  https://versilian-studios.com/vcsl-keys/
#   File:    https://versilian-studios.com/Distro/VCSL_Keys.zip
#   size:    655370475 bytes
#   sha256:  2e91c9aa7b16d936f035963149df1fe4cbd65911116fb3e3ea6daca52e92024b
#            NOT published upstream; pinned here at this repository's own first
#            fetch, the way fetch_salamander.sh pins its own. If it changes,
#            Versilian have re-cut the bundle and the parameters estimated from
#            the old one need re-deriving, not silently accepting.
#
# WHAT SHIPS AND WHAT DOES NOT. The recordings are never redistributed by this
# repository. This script downloads them into the gitignored data/; what ships
# is the parameters estimated from them, under MIT. CC0 asks for nothing, but
# the attribution is recorded in ATTRIBUTION.md and in the preset's own header
# anyway: knowing which piano a preset is a measurement of is a property of the
# measurement, not a licence obligation.
#
# TWO THINGS THIS SCRIPT DOES BEYOND UNPACKING, both of them once:
#
#  1. RESAMPLES 44100 -> 48000, offline. The engine's clock is 48 kHz. Left
#     alone, every single measurement of this preset would have the boundary
#     resampler inside it; done here, once, the tree the estimators read is a
#     48 kHz tree. The method is one pass of the crate's own band-limited sinc
#     resampler (`audio::resample`, rubato — the same one the boundary
#     resampler and the sampler's pitch shift use), written to 32-bit float
#     WAV. Float rather than 24-bit integer on purpose: quantising the
#     resampler's output would need a dither decision, and a dither is a noise
#     floor written into the material the halo census reads.
#
#  2. GENERATES THE INSTRUMENT DEFINITION. VCSL does ship one
#     (`Upright Piano, Knight.sfz`) and it is kept, as
#     `shipped-VCSL-generated.sfz`, for reference — but it is NOT a measurement
#     input. It is auto-generated and it carries three things that would be
#     read as the instrument: per-region `volume` that re-levels the layers
#     (13.73 dB on vl1 against 8.13 on vl2, compressing the piano's real
#     10.5 dB to about 5) where library.rs deliberately applies volume_db when
#     comparing levels; per-sample `tune` of up to -47 cents from the
#     generator's bass pitch-detection failures, which would corrupt the
#     tuning-curve estimate outright; and `offset` trims of the attack the
#     tracker finds for itself.
#
# Both are `piano-tuner adapt vcsl-knight-upright --resample`, which is called
# at the end when a build is available and printed when it is not.

set -eu

url='https://versilian-studios.com/Distro/VCSL_Keys.zip'
archive='VCSL_Keys.zip'
sha256='2e91c9aa7b16d936f035963149df1fe4cbd65911116fb3e3ea6daca52e92024b'
tree='vcsl-knight-upright'
inner='Upright Piano, Knight'
marker="$tree/$tree.sfz"

cd "$(dirname "$0")"

if [ ! -f "$archive" ]; then
    echo "fetching $archive (625 MiB)"
    curl -fSL --progress-bar --retry 8 --retry-delay 5 --retry-all-errors \
         -C - -o "$archive.part" "$url"
    mv "$archive.part" "$archive"
fi

echo "verifying $archive"
if command -v shasum >/dev/null 2>&1; then
    echo "$sha256  $archive" | shasum -a 256 -c -
else
    echo "$sha256  $archive" | sha256sum -c -
fi

if [ ! -d "$tree/Sustains" ]; then
    echo "unpacking $inner/ into $tree/"
    # The bundle carries nine keyboard instruments; only one is wanted, and
    # unpacking the other eight would cost 500 MiB for nothing.
    rm -rf "$tree.part"
    unzip -q "$archive" "$inner/*" "$inner.sfz" -d "$tree.part"
    rm -rf "$tree"
    mv "$tree.part/$inner" "$tree"
    # Kept, clearly named, and never used as a measurement input — see above.
    mv "$tree.part/$inner.sfz" "$tree/shipped-VCSL-generated.sfz"
    rmdir "$tree.part" 2>/dev/null || rm -rf "$tree.part"
    find "$tree" -name '.DS_Store' -delete
fi

if [ ! -f "$marker" ]; then
    echo "resampling to 48 kHz and writing the measurement map"
    if command -v cargo >/dev/null 2>&1 && [ -f ../Cargo.toml ]; then
        (cd .. && cargo run --release --quiet -p piano-tuner -- \
            adapt vcsl-knight-upright --root "data/$tree" --resample \
            --out "data/$marker")
    else
        echo "  no cargo here; run this from the repo root when you have one:" >&2
        echo "    cargo run --release -p piano-tuner -- adapt vcsl-knight-upright \\" >&2
        echo "      --root data/$tree --resample --out data/$marker" >&2
    fi
fi

echo "ready: $(pwd)/$tree"
echo "  $(find "$tree/Sustains" -name '*.wav' | wc -l | tr -d ' ') struck recordings at 48 kHz,"
echo "  licence CC0 1.0, Versilian Studios LLC / Simon Dalzell."
