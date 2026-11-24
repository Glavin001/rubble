#!/bin/sh
# The baselines and checks of engine/docs/profiling-metal.md, in bounded parts, each one
# GPU job (scripts/gpu-lock.sh). From engine/, after `cargo build --release -p
# engine-scenes --bin metal_profile` (and the engine-metal tests for `tests`):
#   tools/metal-profile/baseline.sh checks   state hashes: twice default, labels, debug libs
#   tools/metal-profile/baseline.sh small    Brick Ring 9k/28k, Box Pile 32k, 1 and 8 cars
#   tools/metal-profile/baseline.sh large    Brick Ring 110k, 64 cars (and wide)
#   tools/metal-profile/baseline.sh tests    vehicle_parity, metalsim, explicit_parity
#   tools/metal-profile/baseline.sh all      small, large, captures, an Instruments trace and
#       tests in one hold of the lock (the queue is long), each run under its own alarm
# Outputs: out/metal-profile/<part>/ at the repository root.
here="$(cd "$(dirname "$0")" && pwd)"
engine="$(cd "$here/../.." && pwd)"
cd "$engine" || exit 1
bin="$engine/target/release/metal_profile"
lock="$engine/scripts/gpu-lock.sh"
part="$1"
case "$part" in
  checks)
    exec "$lock" --limit 900 -- sh -c "
      for v in default default2; do $bin --scene 'rigid:Brick Ring (9k)' --scene cars:8 --warm 30 --steps 30 --split 0 --no-replay --tag checks/\$v || exit 1; done
      RUBBLE_METAL_LABELS=1 $bin --scene 'rigid:Brick Ring (9k)' --scene cars:8 --warm 30 --steps 30 --split 0 --no-replay --tag checks/labels || exit 1
      RUBBLE_METAL_DEBUG_LIB=1 $bin --scene 'rigid:Brick Ring (9k)' --scene cars:8 --warm 30 --steps 30 --split 0 --no-replay --tag checks/debuglib || exit 1"
    ;;
  small)
    exec "$lock" --limit 900 -- "$bin" --scene 'rigid:Brick Ring (9k)' --scene 'rigid:Brick Ring (28k)' \
      --scene 'rigid:Box Pile (32k)' --scene cars:1 --scene cars:8 --scene cars:8:wide --tag small
    ;;
  large)
    exec "$lock" --limit 900 -- "$bin" --scene 'rigid:Brick Ring (110k)' --scene cars:64 --scene cars:64:wide --reps 5 --tag large
    ;;
  tests)
    exec "$lock" --limit 1200 -- sh -c "
      cargo test --release -p engine-metal --test vehicle_parity --test metalsim --test explicit_parity -- --test-threads=1"
    ;;
  all)
    # Each run bounded by its own alarm (perl), the hold by --limit.
    b() { echo "perl -e 'alarm $1; exec @ARGV' --"; }
    trace="$(cd "$engine/.." && pwd)/out/metal-trace"
    mkdir -p "$trace"
    exec "$lock" --limit 3000 -- sh -c "
      $(b 900) $bin --scene 'rigid:Brick Ring (9k)' --scene 'rigid:Brick Ring (28k)' --scene 'rigid:Box Pile (32k)' --scene cars:1 --scene cars:8 --scene cars:8:wide --tag small
      $(b 900) $bin --scene 'rigid:Brick Ring (110k)' --scene cars:64 --scene cars:64:wide --reps 5 --tag large
      RUBBLE_METAL_DEBUG_LIB=1 RUBBLE_CAPTURE_STEP=40 RUBBLE_CAPTURE_OUT=$engine/../out/metal-capture/ring9k-step40.gputrace $(b 300) $bin --scene 'rigid:Brick Ring (9k)' --warm 30 --steps 20 --split 0 --no-replay --tag capture
      RUBBLE_METAL_DEBUG_LIB=1 RUBBLE_CAPTURE_STEP=40 RUBBLE_CAPTURE_OUT=$engine/../out/metal-capture/cars8-frame40.gputrace $(b 300) $bin --scene cars:8 --warm 30 --steps 20 --split 0 --no-replay --tag capture
      rm -rf $trace/ring9k.trace $trace/cars8.trace
      $(b 300) xcrun xctrace record --template 'Metal System Trace' --time-limit 8s --output $trace/ring9k.trace --env RUBBLE_METAL_LABELS=1 --env RUBBLE_GPU_LOCK=0 --launch -- $bin --scene 'rigid:Brick Ring (9k)' --warm 30 --steps 300 --split 0 --no-replay --tag xctrace
      $(b 300) xcrun xctrace record --template 'Metal System Trace' --time-limit 8s --output $trace/cars8.trace --env RUBBLE_METAL_LABELS=1 --env RUBBLE_GPU_LOCK=0 --launch -- $bin --scene cars:8 --warm 30 --steps 300 --split 0 --no-replay --tag xctrace
      $(b 1200) cargo test --release -p engine-metal --test vehicle_parity --test metalsim --test explicit_parity -- --test-threads=1
      echo ALL-DONE"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
