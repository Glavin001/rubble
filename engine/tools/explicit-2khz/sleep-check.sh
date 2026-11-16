#!/bin/sh
# Vehicle sleep on the fast vehicle paths, checked on saved states in one hold of the GPU
# lock: the first rung of AGENTS.md's "Checking a kernel change" for
# explicit_beams_sleep.slang and the paths' sleep hooks
# (engine/docs/explicit-vehicles-in-world.md §10.7).
#
#   tools/explicit-2khz/sleep-check.sh            (from engine/; about four minutes)
#
# 1. Sleep on with nobody asleep against sleep off, every mountain fixture
#    (fast-check.sh --make): the physics state bit for bit (`--ab-switch sleep`; the
#    sleep records, anchors and flags are sleep's own).
# 2. Cars asleep. The mountain's cars never meet sleep's rule (vehicle-perf-first-
#    principles.md §8.26), so these run on the flat arena: eight cars and one car parked,
#    saved at t 5 s (out/fixtures_flat, made here when missing), sleep on at the default
#    thresholds. The lone car sleeps 1.0 s after the fixture, the row of seven (one
#    contact group) 2.4 s after it, so 180 frames hold both and the frames after:
#    - the split paths against a workgroup a vehicle or contact group, the reference
#      for sleep's rule (`groupsplit`): every state word and every sleep word;
#    - the awake-indexed dispatch against every vehicle dispatched (`sleeplist`);
#    - one car: self-collision split against its workgroup (`selfsplit`, wide off), and
#      the fleet-wide passes against self-collision split (`wide`).
#    Each check prints the cars' sleep states after its frames (still steps, asleep): a
#    check in which no car fell asleep proves nothing about sleeping cars.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
profile="${PROFILE:-fast}"
car="${CAR:-$HOME/Development/rubble-artifacts/sedan-live/mods/sedan_new}"
cd "$engine" || exit 1
if [ -z "$FAST_CHECK_LOCKED" ]; then
  [ "$profile" = release ] && flag="--release" || flag="--profile $profile"
  # shellcheck disable=SC2086
  cargo build $flag -p engine-world --example vehicle_lab 2>&1 | grep -E "^error" -A12
  exec python3 "$engine/scripts/with-gpu-lock.py" --limit "${LIMIT:-590}" -- env RUBBLE_GPU_LOCK=0 FAST_CHECK_LOCKED=1 sh "$0" "$@"
fi
bin="$engine/target/$profile/examples/vehicle_lab"
flat="$engine/out/fixtures_flat"
bad=0
echo "== sleep on, nobody asleep, against sleep off (every mountain fixture)"
sh "$here/fast-check.sh" sleep || bad=1
for n in 8 1; do
  if [ ! -f "$flat/cars${n}_t5.0.fx" ]; then
    mkdir -p "$flat"
    "$bin" --mode gpu --measure mountain-perf --cars "$n" --car "$car" --rockslide 0 --self-split 1 --timing step --scene flat \
      --ab 5 --ab-switch perf=0 --ab-frames 1 --save-fixtures "$flat" --secs 5.3 2>&1 | grep -E "fixture|^Error"
  fi
done
asleep() {
  # asleep SWITCH FIXTURE [more vehicle_lab options]: 180 frames each way, sleep on.
  switch="$1"; fixture="$2"; shift 2
  AB_SLEEP_STATES=1 FIXTURES="$flat" FRAMES="${FRAMES_ASLEEP:-180}" EXTRA="--scene flat --vehicle-sleep 1 $*" sh "$here/fast-check.sh" "$switch" "$fixture" || bad=1
}
echo "== cars asleep: the split paths against a workgroup a vehicle or contact group"
asleep groupsplit cars8_t5
echo "== cars asleep: the awake-indexed dispatch against every vehicle dispatched"
asleep sleeplist cars8_t5
echo "== a car asleep: self-collision split against its workgroup"
asleep selfsplit cars1_t5 --wide 0
echo "== a car asleep: the fleet-wide passes against self-collision split"
asleep wide cars1_t5
exit $bad
