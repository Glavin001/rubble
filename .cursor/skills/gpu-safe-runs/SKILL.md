---
name: gpu-safe-runs
description: Run GPU work (native wgpu/Metal/CUDA binaries, GPU cargo tests, headless Chrome WebGPU pages) without wedging the machine. Use before any GPU run, when adding or editing a compute kernel with loops, or when a run hangs, the GPU seems busy after a process died, or timings look contended.
---

# Safe GPU runs

On macOS, Metal cannot abort a kernel that is already running. A runaway loop keeps the GPU busy after its process is killed, and only a reboot clears it. This happened three times in one day. Prevention has to live inside the kernels and in how runs are launched.

## In kernels
- **Every loop has a bound:** a real maximum, or the per-frame pass budget (`engine/slang/kernels/explicit_loop_cap.slang`: `ebBudget`/`ebSpent`; `ebEndMost` for loops that contain barriers, which need uniform caps).
- **Hitting a cap sets a status word,** and the host fails the step: `World::step` returns the error and `kernel_status()` shows it.
- **Size budgets from the work.** The host computes the explicit frame's budget per frame (2^22–2^31); don't hard-code a small one.
- **The census test enforces this:** `cargo test -p engine-webgpu --test explicit_loop_cap` fails on any uncapped loop. Run it after kernel edits.
- **Distrust indices read from GPU state** (headers, counts). Validate them in the kernel, and on the CPU before submit where it's cheap (NaN, capacities, header sanity).

## Launching runs
1. **CPU reference first.**
2. **Then a tiny GPU run:** a few steps, with a deliberately low budget.
3. **Then the real run.**

Rules for every run:
- **One GPU job on the machine at a time.** Native binaries take `~/.rubble/gpu.lock` via `hold_gpu_lock()` (override the path with `RUBBLE_GPU_LOCK`; `RUBBLE_GPU_LOCK=0` turns it off).
- **Wrap everything else,** including headless Chrome and GPU cargo tests:
  ```bash
  python3 engine/scripts/with-gpu-lock.py --limit 900 -- env RUBBLE_GPU_LOCK=0 cargo test -p engine-world --release --test explicit_vehicles
  ```
  Set `RUBBLE_GPU_LOCK=0` inside the wrapper, or a binary that takes the lock itself deadlocks waiting on yours.
- **Hard kill timer on every run, 10–20 minutes at most.** The wrapper kills the whole process group.
- **Never `pgrep -f`/`pkill -f`:** it matches unrelated processes. Wait on PIDs.
- **Benchmark one thing at a time.** Two GPU jobs make both timings meaningless.

## If the GPU is already wedged
Symptoms: tests that used to pass stall, and Activity Monitor shows GPU at 100 % with nothing running. Don't keep launching jobs; they queue behind the wedge and look like new bugs. Tell the user it needs a reboot.
