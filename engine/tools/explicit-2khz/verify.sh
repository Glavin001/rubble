#!/bin/sh
# The explicit kernels' parity suites, one at a time, each under a kill timer:
# engine-webgpu (bitwise against the frozen reference, contact, beams, rigid, sleep,
# centred damping, JBeam ext), engine-world explicit_vehicles, engine-metal parity.
# Usage (from engine/): tools/explicit-2khz/verify.sh [crate:suite ...]   (default: all)
# Builds each suite, then runs its test binary (the timer kills the binary itself).
# One PASS/FAIL line per suite; logs in $VERIFY_LOGS
# (default ~/Development/rubble-artifacts/explicit-2khz/verify).
here="$(cd "$(dirname "$0")" && pwd)"
logs="${VERIFY_LOGS:-$HOME/Development/rubble-artifacts/explicit-2khz/verify}"
mkdir -p "$logs"
all="webgpu:explicit_beams_bitwise webgpu:explicit_contact_gpu webgpu:explicit_beams_gpu webgpu:explicit_rigid_gpu webgpu:explicit_sleep_gpu webgpu:explicit_centred_damping_gpu webgpu:explicit_jbeam_ext_gpu world:explicit_vehicles metal:explicit_parity metal:vehicle_parity metal:explicit_drive_parity"
suites="${*:-$all}"
for s in $suites; do
  crate="engine-${s%%:*}"
  test="${s#*:}"
  log="$logs/$test.log"
  bin=$(cargo test --release --no-run -p "$crate" --test "$test" 2>&1 | sed -n 's/.*Executable .*(\(.*\)).*/\1/p' | tail -1)
  if [ -z "$bin" ]; then echo "FAIL(build) $crate $test"; continue; fi
  dir="crates/$crate"
  (cd "$dir" && "$here/bounded.sh" 1200 "../../$bin" --test-threads=1) > "$log" 2>&1
  code=$?
  res=$(grep -E '^test result' "$log" | tail -1)
  if [ "$code" -eq 0 ]; then echo "PASS $crate $test: $res"; else echo "FAIL($code) $crate $test: $res (log $log)"; fi
done
