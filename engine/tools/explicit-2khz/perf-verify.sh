#!/bin/sh
# The explicit kernels' GPU suites for a schedule switch (RUBBLE_PERF_FLAGS), one at a
# time as the machine's one GPU job, each under a kill timer:
#   tools/explicit-2khz/perf-verify.sh FLAGS [crate:suite ...]      (from engine/)
# FLAGS: the perf flags the suites run with (0: every switch off; `default`: the host's own). Each suite holds the GPU
# kernel to the CPU reference or the frozen reference bit for bit, so a pass with a flag
# on is that flag's proof against them. One PASS/FAIL line per suite; logs in
# $VERIFY_LOGS (default out/perf/verify).
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
flags="${1:-0}"
shift
logs="${VERIFY_LOGS:-$engine/out/perf/verify}"
mkdir -p "$logs"
all="webgpu:explicit_loop_cap webgpu:explicit_beams_bitwise webgpu:explicit_contact_gpu webgpu:explicit_pair_reuse_gpu webgpu:explicit_beam_order_gpu webgpu:debris_retire_gpu webgpu:showcase_car_gpu webgpu:vv_grid_gpu world:explicit_vehicles world:vehicle_lab_reset world:vehicle_lab_piece_windows"
suites="${*:-$all}"
cd "$engine" || exit 1
for s in $suites; do
  crate="engine-${s%%:*}"
  test="${s#*:}"
  log="$logs/$test.flags$flags.log"
  bin=$(cargo test --release --no-run -p "$crate" --test "$test" 2>&1 | sed -n 's/.*Executable .*(\(.*\)).*/\1/p' | tail -1)
  if [ -z "$bin" ]; then echo "FAIL(build) $crate $test"; continue; fi
  (cd "crates/$crate" && python3 "$engine/scripts/with-gpu-lock.py" --limit "${LIMIT:-1200}" -- env RUBBLE_GPU_LOCK=0 $( [ "$flags" = default ] || echo "RUBBLE_PERF_FLAGS=$flags" ) "../../$bin" --test-threads=1) > "$log" 2>&1
  code=$?
  res=$(grep -E '^test result' "$log" | tail -1)
  if [ "$code" -eq 0 ]; then echo "PASS flags=$flags $crate $test: $res"; else echo "FAIL($code) flags=$flags $crate $test: $res (log $log)"; fi
done
