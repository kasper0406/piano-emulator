#!/bin/sh
# build.sh — the whole Swift side, from a terminal, with no Xcode window open.
#
#   ./app/build.sh                 build everything and run the offline harness
#   ./app/build.sh --no-harness    build only
#   ./app/build.sh --register      also copy to /Applications and launch once,
#                                  which is what registers the AUv3 with PluginKit
#   ./app/build.sh --auval         also run `auval -v aumu Pemu KsNi`
#   ./app/build.sh --clean         throw away the generated project and the build
#
# What it produces, in `app/build/`:
#
#   Piano Emulator.app             the container app, with
#     Contents/PlugIns/PianoEmulatorAU.appex   the AUv3 inside it
#   parity-harness                 the offline render harness
#
# ## Signing
#
# The build is **ad-hoc signed** (`CODE_SIGN_IDENTITY = -`), which is a real
# signature as far as the sandbox and PluginKit are concerned and no signature
# at all as far as Gatekeeper is. That is deliberate: it is the weakest thing
# that lets the app actually launch and the appex actually register on this
# machine, and it needs no Developer Program membership.
#
# Shipping is a **separate step**, and it is `DISTRIBUTION.md` M5, not this
# milestone. It is not a flag on this script because it is not a build; it is a
# different signature over the same bytes:
#
#   codesign --force --options runtime --timestamp \
#       --sign "Developer ID Application: NAME (TEAMID)" \
#       --entitlements app/AUv3/PianoEmulatorAU.entitlements \
#       "build/Piano Emulator.app/Contents/PlugIns/PianoEmulatorAU.appex"
#   codesign --force --options runtime --timestamp \
#       --sign "Developer ID Application: NAME (TEAMID)" \
#       --entitlements app/App/PianoEmulator.entitlements \
#       "build/Piano Emulator.app"
#   xcrun notarytool submit ... && xcrun stapler staple "build/Piano Emulator.app"
#
# Inside out, one component at a time, and *not* `codesign --deep`, which has
# been deprecated for signing since macOS 13 whatever the plugin CI guides say.
# The App Store build differs again (`3rd Party Mac Developer Application`, a
# provisioning profile, no Developer ID) and is M7.
#
# SPDX-License-Identifier: MIT

set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/.." && pwd)

run_harness=1
register=0
run_auval=0
clean=0
configuration=Release

for argument in "$@"; do
    case "$argument" in
    --no-harness) run_harness=0 ;;
    --register) register=1 ;;
    --auval) run_auval=1 ;;
    --clean) clean=1 ;;
    --debug) configuration=Debug ;;
    *)
        echo "build.sh: unknown option $argument" >&2
        exit 2
        ;;
    esac
done

if [ "$clean" -eq 1 ]; then
    rm -rf "$here/.build" "$here/build" "$here/PianoEmulator.xcodeproj"
    echo "cleaned"
    exit 0
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- the library

step "the Rust library (profile: dist — panic = abort, which is the one the C ABI needs)"
cd "$root"
cargo build --profile dist -p piano-emulator-ffi
test -f "$root/target/dist/libpiano_emulator_ffi.a" ||
    { echo "no libpiano_emulator_ffi.a in target/dist" >&2; exit 1; }

# ------------------------------------------------------- the things that must agree

step "consistency"

# The four-character codes are written twice — in the appex's Info.plist, which
# is what PluginKit reads, and in `PianoIdentity.swift`, which is what the
# standalone looks the appex up with. A silent disagreement between them is an
# app that cannot find its own plugin, so it is a build failure here instead.
plist_value() {
    /usr/libexec/PlistBuddy -c "Print :NSExtension:NSExtensionAttributes:AudioComponents:0:$1" \
        "$here/AUv3/Info.plist"
}
for field in type subtype manufacturer; do
    from_plist=$(plist_value "$field")
    if [ "$field" = type ]; then
        # `kAudioUnitType_MusicDevice`, which the Swift names rather than spells.
        from_swift=aumu
    else
        from_swift=$(awk -v want="$field" '
            $0 ~ ("static let " want ".*fourCharCode") {
                match($0, /"[^"]+"/); print substr($0, RSTART + 1, RLENGTH - 2)
            }' "$here/Shared/PianoIdentity.swift")
    fi
    if [ "$from_plist" != "$from_swift" ]; then
        echo "the appex's $field is '$from_plist' but PianoIdentity.swift says '$from_swift'" >&2
        exit 1
    fi
    echo "  $field  $from_plist"
done

sandbox_safe=$(plist_value sandboxSafe)
[ "$sandbox_safe" = "true" ] || { echo "sandboxSafe is not true" >&2; exit 1; }
echo "  sandboxSafe  true"

for preset in default salamander-c5; do
    test -f "$root/presets/$preset.toml" ||
        { echo "presets/$preset.toml is missing — it is a factory preset" >&2; exit 1; }
done
echo "  presets  default.toml, salamander-c5.toml"

# ---------------------------------------------------------------- the project

step "the Xcode project"
command -v xcodegen >/dev/null 2>&1 ||
    { echo "xcodegen is not installed: brew install xcodegen" >&2; exit 1; }
cd "$here"
xcodegen generate --quiet
echo "  PianoEmulator.xcodeproj (generated from project.yml — do not edit it, edit that)"

# ------------------------------------------------------------------ the build

step "xcodebuild ($configuration, arm64, ad-hoc signed)"
xcodebuild \
    -project "$here/PianoEmulator.xcodeproj" \
    -scheme PianoEmulator \
    -configuration "$configuration" \
    -derivedDataPath "$here/.build" \
    build 2>&1 | grep -E "error:|warning:|BUILD" || true
xcodebuild \
    -project "$here/PianoEmulator.xcodeproj" \
    -scheme ParityHarness \
    -configuration "$configuration" \
    -derivedDataPath "$here/.build" \
    build 2>&1 | grep -E "error:|warning:|BUILD" || true

products="$here/.build/Build/Products/$configuration"
app="$products/Piano Emulator.app"
appex="$app/Contents/PlugIns/PianoEmulatorAU.appex"
test -d "$app" || { echo "no app was produced" >&2; exit 1; }
test -d "$appex" || { echo "the appex is not embedded in the app" >&2; exit 1; }

rm -rf "$here/build"
mkdir -p "$here/build"
cp -R "$app" "$here/build/"
cp "$products/parity-harness" "$here/build/"

step "what was built"
codesign --verify --strict --verbose=1 "$here/build/Piano Emulator.app" 2>&1 | sed 's/^/  /'
echo "  app    $here/build/Piano Emulator.app"
echo "  appex  Contents/PlugIns/PianoEmulatorAU.appex"
du -sh "$here/build/Piano Emulator.app" | sed 's/^/  size   /'

# ---------------------------------------------------------------- the harness

if [ "$run_harness" -eq 1 ]; then
    step "offline parity: the AU's own render block against the C harness's md5"
    "$here/build/parity-harness" "$root/presets/default.toml" "$root/ffi/harness/phrase.mid"
fi

# ------------------------------------------------------------- registration

if [ "$register" -eq 1 ]; then
    step "registering the AUv3 with PluginKit"
    # An AUv3 is not a file in a plug-ins folder: PluginKit registers the
    # extension inside a *launched* app, which is why the container app has to
    # be run once (`DISTRIBUTION.md` §Plugin formats). /Applications is where
    # PluginKit is most reliable about noticing.
    rm -rf "/Applications/Piano Emulator.app"
    cp -R "$here/build/Piano Emulator.app" /Applications/
    open -a "/Applications/Piano Emulator.app"
    sleep 6
    pluginkit -m -v -p com.apple.AudioUnit-UI 2>/dev/null | grep -i piano || true
    auval -a 2>/dev/null | grep -i "Pemu\|Piano Emulator" || true
fi

if [ "$run_auval" -eq 1 ]; then
    step "auval -v aumu Pemu KsNi"
    auval -v aumu Pemu KsNi || true
fi

step "next"
cat <<'EOF'
  open "app/build/Piano Emulator.app"          play it; this also registers the AUv3
  ./app/build.sh --register --auval             register and validate in one go
  app/build/parity-harness - ffi/harness/phrase.mid          the built-in preset
  app/build/parity-harness presets/salamander-c5.toml ffi/harness/phrase.mid
  app/build/parity-harness --component presets/default.toml ffi/harness/phrase.mid
                                                the *registered* component, i.e.
                                                the appex, rather than the class
EOF
