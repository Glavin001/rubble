#!/usr/bin/env bash
# Prints one digest per entry of the Slang output cache ($RUBBLE_SLANG_CACHE): of every
# SPIR-V and WGSL file's path and bytes. Two runners that print the same line for the
# same entry compiled the same kernels, byte for byte (Linux, macOS and Windows each
# compile their own: this is how to see whether they agree).
set -euo pipefail
cache="${RUBBLE_SLANG_CACHE:?}"
sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum; else shasum -a 256; fi; }
shopt -s nullglob
for entry in "$cache"/*/; do
  digest="$(cd "$entry" && find spirv wgsl -type f | LC_ALL=C sort | while read -r f; do
    printf '%s ' "$f"; sum < "$f"; done | sum | cut -c1-16)"
  echo "Slang cache entry $(basename "$entry"): $(cd "$entry" && find spirv wgsl -type f | wc -l | tr -d ' ') files, digest $digest"
done
