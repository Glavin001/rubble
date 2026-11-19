#!/bin/sh
# Instruments' Metal System Trace of a native command, bounded and under the GPU lock,
# with its key numbers exported as text and JSON (engine/docs/profiling-metal.md).
#
#   engine/scripts/metal-trace.sh [--time-limit 10s] [--name NAME] -- COMMAND [ARG ...]
#   engine/scripts/metal-trace.sh --summarize out/metal-trace/NAME.trace
#
# Records `xcrun xctrace record --template 'Metal System Trace' --launch -- COMMAND` into
# out/metal-trace/NAME.trace (open it in Instruments for the timeline), then runs
# metal-trace-summary.py on it: `xctrace export` of the GPU and command-buffer tables
# into out/metal-trace/NAME/ (XML), and NAME/summary.{txt,json}: GPU busy and idle
# time, command buffers and encoders per second, submit-to-GPU-start latency, the gaps
# between command buffers, and time per encoder label.
# Works for any Metal process: MetalSim and MetalExplicitBeams bins, and wgpu-on-Metal
# ones (city_scale, vehicle_lab). RUBBLE_METAL_LABELS=1 is set so Rubble's own command
# buffers and debug groups carry the plan's names. LIMIT=seconds (default 600) kills the
# recording.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/.." && pwd)"
outdir="$(cd "$engine/.." && pwd)/out/metal-trace"
if [ "$1" = "--summarize" ]; then
  exec python3 "$here/metal-trace-summary.py" "$2"
fi
limit_t=10s
name="trace-$(date +%Y%m%d-%H%M%S)"
while [ $# -gt 0 ]; do
  case "$1" in
    --time-limit) limit_t="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -eq 0 ] && { sed -n '2,18p' "$0"; exit 2; }
mkdir -p "$outdir"
trace="$outdir/$name.trace"
rm -rf "$trace"
RUBBLE_METAL_LABELS=1 "$here/gpu-lock.sh" --limit "${LIMIT:-600}" -- \
  xcrun xctrace record --template 'Metal System Trace' --time-limit "$limit_t" \
  --output "$trace" --env RUBBLE_METAL_LABELS=1 --env RUBBLE_GPU_LOCK=0 --launch -- "$@"
status=$?
[ -d "$trace" ] || { echo "metal-trace: no trace written (status $status)" >&2; exit 1; }
python3 "$here/metal-trace-summary.py" "$trace"
