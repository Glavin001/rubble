#!/usr/bin/env bash
# Static kernel budget guard for the world kernels (macOS + Xcode only; no GPU work).
#
# Builds shaders/generated/world.metal into world.metallib exactly as build.rs does
# (xcrun metal -fno-fast-math, metallib), then for every kernel reports driver instruction
# count, temporary registers, spilled bytes, spilled bytes x threads per threadgroup,
# static threadgroup memory, maxTotalThreadsPerThreadgroup and machine-code size, compares
# them with the measured limits (limits/<target>.toml: instruction-cache and stack knees)
# and with the committed baseline limits/kernel_budget.toml, and exits non-zero when a
# kernel regresses beyond the margin (default +5% instructions or code bytes, or any
# growth in spilled bytes).
#
# PRIVATE API: instruction/register/spill counts come from the undocumented Metal pipeline
# option 0x40000 (MTLComputePipelineReflection.performanceStatistics) and code size from
# Xcode's applegpu-nt translator. Either may disappear in a future OS/Xcode; the guard then
# warns, reports -1 and skips those comparisons.
#
#   scripts/kernel_budget.sh                 compare against the baseline
#   scripts/kernel_budget.sh --update        accept the current numbers as the baseline
#   scripts/kernel_budget.sh --margin 10     allow +10%
#   scripts/kernel_budget.sh --limits limits/m3_max.toml
#   scripts/kernel_budget.sh --json out.json    also write per-kernel numbers (tests/kernel_limits.rs)
set -euo pipefail
cd "$(dirname "$0")/.."
update=()
json=()
margin=5
limits=""
while [ $# -gt 0 ]; do
  case "$1" in
    --update) update=(--update) ;;
    --margin) margin=$2; shift ;;
    --limits) limits=$2; shift ;;
    --json) json=(--json "$2"); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
if [ "$(uname)" != Darwin ]; then
  echo "kernel_budget: needs macOS with Xcode (Metal)" >&2
  exit 2
fi
tools/limits/build.sh > /dev/null
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
xcrun -sdk macosx metal -fno-fast-math -c shaders/generated/world.metal -o "$work/world.air"
xcrun -sdk macosx metallib "$work/world.air" -o "$work/world.metallib"
arch=$(tools/limits/build/limits device | sed -n 's/^architecture = "\(.*\)"/\1/p')
if [ -z "$limits" ]; then
  chip=$(sysctl -n machdep.cpu.brand_string)
  limits="limits/$(echo "${chip#Apple }" | tr 'A-Z ' 'a-z_').toml"
fi
tools/limits/build/limits stats "$work/world.metallib" > "$work/stats.tsv"
# shellcheck disable=SC2046
python3 tools/limits/codesize.py --arch "$arch" "$work/world.metallib" $(grep -v '^#' "$work/stats.tsv" | cut -f1) > "$work/code.txt" \
  || echo "kernel_budget: WARNING machine-code size unavailable (applegpu-nt)" >&2
python3 tools/limits/budget.py --stats "$work/stats.tsv" --code "$work/code.txt" --slang shaders/world.slang \
  --limits "$limits" --baseline limits/kernel_budget.toml --margin "$margin" \
  --meta "architecture=$arch" --meta "os=$(sw_vers -productVersion)" \
  --meta "metal=$(xcrun -sdk macosx metal --version 2>/dev/null | head -1)" ${update[@]+"${update[@]}"} ${json[@]+"${json[@]}"}
