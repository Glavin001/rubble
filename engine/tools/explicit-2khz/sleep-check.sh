#!/bin/sh
# Vehicle sleep on the fast vehicle paths, checked on the saved fixtures (fast-check.sh
# --make) in one hold of the GPU lock: the first rung of AGENTS.md's "Checking a kernel
# change" for explicit_beams_sleep.slang and the paths' sleep hooks.
#
#   tools/explicit-2khz/sleep-check.sh            (from engine/; about a minute)
#
# 1. Sleep on with nobody asleep against sleep off, every fixture: the physics state bit
#    for bit (`--ab-switch sleep`; the sleep records, anchors and flags are sleep's own).
# 2. Cars asleep: the resting fixtures (REST, default the eight-car pile at t 20 and 30 s
#    and the one-car fixture at t 20 s) with sleep on and thresholds loose and short
#    enough for the cars to fall asleep within the frames (SLEEP, default 0.5 m/s, 1 mm,
#    200 steps; FRAMES_REST frames, default 40):
#    - the split paths against a workgroup a vehicle or contact group, the reference
#      for sleep's rule (`groupsplit`): every state word and every sleep word;
#    - the awake-indexed dispatch against every vehicle dispatched (`sleeplist`);
#    - the fleet-wide passes against a workgroup a vehicle (`wide`; one car).
#    Each line's sleep states are printed (AB_SLEEP_STATES): a check in which no car fell
#    asleep proves nothing about sleeping cars.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
cd "$engine" || exit 1
if [ -z "$FAST_CHECK_LOCKED" ]; then
  cargo build --profile "${PROFILE:-fast}" -p engine-world --example vehicle_lab 2>&1 | grep -E "^error" -A12
  exec python3 "$engine/scripts/with-gpu-lock.py" --limit "${LIMIT:-590}" -- env RUBBLE_GPU_LOCK=0 FAST_CHECK_LOCKED=1 sh "$0" "$@"
fi
sleep_opts="${SLEEP:---vehicle-sleep 1 --sleep-speed 0.5 --sleep-reach-mm 1 --sleep-dwell 200}"
rest="${REST:-cars8_t20 cars8_t30}"
bad=0
echo "== sleep on, nobody asleep, against sleep off (every fixture)"
sh "$here/fast-check.sh" sleep || bad=1
echo "== cars asleep: the split paths against a workgroup a vehicle or group ($sleep_opts)"
# shellcheck disable=SC2086
AB_SLEEP_STATES=1 FRAMES="${FRAMES_REST:-40}" EXTRA="$sleep_opts" sh "$here/fast-check.sh" groupsplit $rest cars1_t20 || bad=1
echo "== cars asleep: the awake-indexed dispatch against every vehicle dispatched"
# shellcheck disable=SC2086
AB_SLEEP_STATES=1 FRAMES="${FRAMES_REST:-40}" EXTRA="$sleep_opts" sh "$here/fast-check.sh" sleeplist $rest || bad=1
echo "== a car asleep: the fleet-wide passes against a workgroup a vehicle"
AB_SLEEP_STATES=1 FRAMES="${FRAMES_REST:-40}" EXTRA="$sleep_opts" sh "$here/fast-check.sh" wide cars1_t20 || bad=1
exit $bad
