#!/usr/bin/env bash
# Build the RWKV family release assets into ./dist (uncommitted). Does NOT call gh and creates no release or tag.
# Usage: bash build_release.sh            (builds every package whose folder exists, reports the missing ones)
# Override a package folder with an env var, e.g. PKG_SMALL_TWO_HEAD=/path/to/sm92 bash build_release.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="${WT:-C:/Users/jonat/workspace/wt}"
DIST="$HERE/dist"
mkdir -p "$DIST"

# name -> package folder. Defaults: finished packages live on their own branches (see inventory.md);
# the TODO ones get a path once the package exists (add it to the repo or point the env var at it).
declare -A PKG=(
  [rwkv-big-two-head]="${PKG_BIG_TWO_HEAD:-$WT/big-package/artifacts/models/sm91s2}"   # branch big-package 708b8062
  [rwkv-big-instant-only]="${PKG_BIG_INSTANT_ONLY:-$WT/big-package/artifacts/models/sm99}"   # branch big-package 708b8062
  [rwkv-small-two-head]="${PKG_SMALL_TWO_HEAD:-$WT/sm101-package/artifacts/models/sm92}"
  [rwkv-small-instant-only]="${PKG_SMALL_INSTANT_ONLY:-$WT/sm98-package/artifacts/models/sm98}"
  [rwkv-tiny-two-head]="${PKG_TINY_TWO_HEAD:-$WT/tiny-package/artifacts/models/sm106}"   # branch tiny-package 4ae0b434
  [rwkv-tiny-instant-only]="${PKG_TINY_INSTANT_ONLY:-$WT/sm101-package/artifacts/models/sm101}"
)
ORDER=(rwkv-big-two-head rwkv-big-instant-only rwkv-small-two-head rwkv-small-instant-only rwkv-tiny-two-head rwkv-tiny-instant-only)

zipdir() { # zipdir <src folder> <out zip>; the zip contains one top-level folder named after the zip
  local src="$1" out="$2" name; name="$(basename "$out" .zip)"
  local tmp; tmp="$(mktemp -d)"
  cp -r "$src" "$tmp/$name"
  rm -f "$out"
  if command -v zip >/dev/null 2>&1; then (cd "$tmp" && zip -qr "$out" "$name")
  else python - "$tmp" "$name" "$out" <<'PY'
import sys, shutil
tmp, name, out = sys.argv[1:4]
shutil.make_archive(out[:-4], "zip", tmp, name)
PY
  fi
  rm -rf "$tmp"
}

built=(); missing=()
for n in "${ORDER[@]}"; do
  src="${PKG[$n]}"
  # a package is complete when these exist (same layout as the earlier single-model releases)
  if [ -f "$src/INTEGRATION.md" ] && [ -f "$src/weights.f32.bin" ] && [ -f "$src/weights.int8.bin" ] && [ -f "$src/manifest.json" ] && [ -d "$src/testvectors" ]; then
    zipdir "$src" "$DIST/$n.zip" && built+=("$n.zip")
  else
    missing+=("$n ($src)")
  fi
done

# the integration manual is also attached on its own (as in the earlier releases), taken from the Small two-head package
[ -f "${PKG[rwkv-small-two-head]}/INTEGRATION.md" ] && cp "${PKG[rwkv-small-two-head]}/INTEGRATION.md" "$DIST/INTEGRATION.md"

(cd "$DIST" && rm -f SHA256SUMS && sha256sum *.zip INTEGRATION.md 2>/dev/null > SHA256SUMS)
echo "built: ${built[*]:-none}"
[ ${#missing[@]} -gt 0 ] && { echo "NOT BUILT (package missing):"; printf '  %s\n' "${missing[@]}"; }
echo "dist: $DIST"; cat "$DIST/SHA256SUMS"

# Final step, run by the main session only after the user's OK (never from this script):
# gh release create rwkv-family-2026-11-DD --repo JSchoreels/anki-rwkv --title "RWKV family: Big, Small and Tiny, two-head and Instant-Only" --notes-file "$HERE/RELEASE_NOTES.md" "$DIST"/rwkv-*.zip "$DIST"/INTEGRATION.md "$DIST"/SHA256SUMS
