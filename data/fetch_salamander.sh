#!/bin/sh
# Fetch the Salamander Grand Piano, the target instrument of TUNING.md's stage 1.
#
#   Salamander Grand Piano V3 (FLAC), a Yamaha C5 recorded at 48 kHz / 24 bit
#   with two AKG C414 in AB about 12 cm above the strings: 16 velocity layers,
#   notes sampled in minor thirds from A0 to C8, plus release and pedal noises.
#
#   Author:  Alexander Holm <axeldenstore [at] gmail [dot] com>
#   FLAC release assembled for FreePats by roberto@zenvoid.org
#   Licence: Creative Commons Attribution 3.0 (CC-BY 3.0)
#            http://creativecommons.org/licenses/by/3.0/
#   Source:  https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
#   File:    https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SFZ+FLAC-V3+20200602.tar.gz
#   sha256:  b7760e168494cf095344e217b0af013fc449ad033abbbdf1c65211cf11dc038b
#   size:    741757374 bytes
#
# Anything derived from these recordings — presets/salamander-c5.toml and every
# render made with it — carries the attribution above; it is repeated in the
# preset file's header and in its `description` field so it travels with the
# data rather than with this script.
#
# The archive and the unpacked tree land in data/, which is gitignored. Both
# steps are skipped when their output is already in place, so re-running this
# after a partial run costs one checksum.

set -eu

url='https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPiano-SFZ+FLAC-V3+20200602.tar.gz'
archive='SalamanderGrandPiano-SFZ+FLAC-V3+20200602.tar.gz'
sha256='b7760e168494cf095344e217b0af013fc449ad033abbbdf1c65211cf11dc038b'
# The one file whose presence means the tree is unpacked.
marker='salamander/SalamanderGrandPiano-V3+20200602.sfz'

cd "$(dirname "$0")"

if [ ! -f "$archive" ]; then
    echo "fetching $archive (707 MiB)"
    # To a temporary name first: an interrupted download must not be mistaken
    # for a complete one on the next run.
    curl -fSL --progress-bar -o "$archive.part" "$url"
    mv "$archive.part" "$archive"
fi

echo "verifying $archive"
if command -v shasum >/dev/null 2>&1; then
    echo "$sha256  $archive" | shasum -a 256 -c -
else
    echo "$sha256  $archive" | sha256sum -c -
fi

if [ ! -f "$marker" ]; then
    echo "unpacking into salamander/"
    mkdir -p salamander
    # The archive has one top-level directory; strip it so the tree is at a
    # fixed path whatever the release is named.
    tar xzf "$archive" -C salamander --strip-components=1
fi

echo "ready: $(pwd)/salamander"
echo "  $(ls salamander/samples | grep -c '\.flac$') samples, licence CC-BY 3.0, Alexander Holm"
