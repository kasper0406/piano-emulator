#!/bin/sh
# Fetch the bitKlavier Grand Sample Library — the concert grand of the shipped
# range, and the source of presets/concert-grand-d.toml.
#
#   bitKlavier Grand Sample Library, "Piano Bar" mic image, 48 kHz / 24 bit.
#   A Steinway D concert grand recorded in Taplin Auditorium, Princeton
#   University, between 13 and 20 January 2021. The Piano Bar image is a pair
#   of Earthworks omnis on a bar laid across the harp near the hammers — a
#   close image without room or lid interaction, i.e. the same geometry class
#   as Salamander's AKG C414 AB pair.
#
#   30 keys at exact minor thirds A0..C8 (MIDI 21..108), 16 velocity layers per
#   key, 90 release resonances (30 keys x 3 tiers), 88 chromatic key-off
#   samples (rel1..rel88), and four pedal-mechanism samples
#   (pedalD1/D2, pedalU1/U2). Deliberately the same shape as Salamander: the
#   author adopted Salamander's naming convention.
#
#   Author:  Daniel Trueman <dtrueman [at] princeton [dot] edu>, Princeton
#            University Department of Music
#   Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
#            https://creativecommons.org/licenses/by/4.0/
#            The licence URL is carried in the archive item's own metadata
#            (field `licenseurl` of
#            https://archive.org/metadata/bitKlavierGrand_PianoBar_48k24b),
#            not merely asserted by a third party.
#   Source:  https://bitklavier.com/the-bitklavier-grand/
#   Item:    https://archive.org/details/bitKlavierGrand_PianoBar_48k24b
#   File:    https://archive.org/download/bitKlavierGrand_PianoBar_48k24b/bitKlavierGrand_PianoBar_48k24b.zip
#   size:    2778583990 bytes
#   md5:     5e6f6f84696f9ec01c97081490cdb7de   (published by the archive item)
#   sha1:    c24d72c60b03d8b5243d25d2d604505df411bce1  (published)
#   sha256:  2efa04c28d09a07ee1bf7eacff767615ee632932ef37dd82b668d0fe718df152
#            NOT published upstream; computed at this repository's own first
#            fetch and pinned below, the way fetch_salamander.sh pins its own.
#            All three digests are checked, so a mismatch names which moved.
#
# WHAT SHIPS AND WHAT DOES NOT. The recordings are never redistributed by this
# repository. This script downloads them into the gitignored data/ for anyone
# re-running the estimation pipeline; what this repository ships is the
# parameters estimated from them, under MIT, with the attribution above
# repeated in ATTRIBUTION.md and in the preset's own header and `description`
# field so it travels with the data rather than with this script.
#
# NO SFZ SHIPS WITH THIS LIBRARY. The tuner's survey and reference sampler are
# both driven by an SFZ. `piano-tuner adapt bitklavier-piano-bar` writes one
# over this tree, in Salamander's idiom; this script calls it at the end when a
# build of the tuner is available, and prints the command when it is not.
#
# archive.org's CDN nodes intermittently answer 5xx. Every request below is
# retried; plain retries succeed.

set -eu

url='https://archive.org/download/bitKlavierGrand_PianoBar_48k24b/bitKlavierGrand_PianoBar_48k24b.zip'
archive='bitKlavierGrand_PianoBar_48k24b.zip'
md5='5e6f6f84696f9ec01c97081490cdb7de'
sha1='c24d72c60b03d8b5243d25d2d604505df411bce1'
sha256='2efa04c28d09a07ee1bf7eacff767615ee632932ef37dd82b668d0fe718df152'
tree='bitklavier-piano-bar'
# The one file whose presence means the tree is unpacked and adapted.
marker="$tree/bitklavier-piano-bar.sfz"

cd "$(dirname "$0")"

if [ ! -f "$archive" ]; then
    echo "fetching $archive (2.59 GiB)"
    # To a temporary name first, and resumably: an interrupted download must
    # not be mistaken for a complete one, and 2.6 GiB is too much to restart.
    attempt=1
    until curl -fSL --progress-bar --retry 8 --retry-delay 5 --retry-all-errors \
               -C - -o "$archive.part" "$url"; do
        attempt=$((attempt + 1))
        if [ "$attempt" -gt 10 ]; then
            echo "giving up after 10 attempts; archive.org is answering 5xx" >&2
            exit 1
        fi
        echo "attempt $attempt (archive.org node returned an error; retrying)" >&2
        sleep 10
    done
    mv "$archive.part" "$archive"
fi

echo "verifying $archive"
if command -v shasum >/dev/null 2>&1; then
    echo "$sha1  $archive" | shasum -a 1 -c -
    echo "$sha256  $archive" | shasum -a 256 -c -
else
    echo "$sha1  $archive" | sha1sum -c -
    echo "$sha256  $archive" | sha256sum -c -
fi
if command -v md5sum >/dev/null 2>&1; then
    echo "$md5  $archive" | md5sum -c -
elif command -v md5 >/dev/null 2>&1; then
    got=$(md5 -q "$archive")
    [ "$got" = "$md5" ] || { echo "md5 mismatch: $got != $md5" >&2; exit 1; }
    echo "$archive: MD5 OK"
fi

if [ ! -d "$tree/samples" ]; then
    echo "unpacking into $tree/"
    rm -rf "$tree.part"
    mkdir -p "$tree.part"
    unzip -q "$archive" -d "$tree.part"
    # The zip carries a macOS sidecar tree and Finder droppings; neither is
    # part of the library and both would be walked by the adapter.
    rm -rf "$tree.part/__MACOSX"
    find "$tree.part" -name '.DS_Store' -delete
    # One top-level directory inside, as with Salamander: flatten it so the
    # tree sits at a fixed path whatever the release is named.
    inner=$(find "$tree.part" -mindepth 1 -maxdepth 1 -type d | head -1)
    count=$(find "$tree.part" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
    rm -rf "$tree"
    if [ "$count" = "1" ] && [ -n "$inner" ]; then
        mv "$inner" "$tree"
        rm -rf "$tree.part"
    else
        mv "$tree.part" "$tree"
    fi
    # The adapter expects the recordings under samples/; some releases put
    # them at the top level.
    if [ ! -d "$tree/samples" ]; then
        mkdir -p "$tree/samples"
        find "$tree" -maxdepth 1 -name '*.wav' -exec mv {} "$tree/samples/" \;
    fi
fi

echo "adapting: writing the SFZ this library does not ship"
if [ ! -f "$marker" ]; then
    if command -v cargo >/dev/null 2>&1 && [ -f ../Cargo.toml ]; then
        (cd .. && cargo run --release --quiet -p piano-tuner -- \
            adapt bitklavier-piano-bar --root "data/$tree" --out "data/$marker")
    else
        echo "  no cargo here; run this from the repo root when you have one:" >&2
        echo "    cargo run --release -p piano-tuner -- adapt bitklavier-piano-bar \\" >&2
        echo "      --root data/$tree --out data/$marker" >&2
    fi
fi

echo "ready: $(pwd)/$tree"
echo "  $(find "$tree/samples" -name '*.wav' | wc -l | tr -d ' ') recordings, 48 kHz/24 bit,"
echo "  licence CC BY 4.0, Daniel Trueman / Princeton University."
