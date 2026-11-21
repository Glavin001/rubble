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
- **On native Metal the explicit vehicle kernels run without wgpu's index clamps and division guards** (3 to 9 % of every frame; the default since `vehicle-perf-first-principles.md` §8.48). An index out of range there reads or writes other memory instead of the last element. When a kernel edit gives garbage or a result that moves between runs, run it once with `RUBBLE_SHADER_CHECKS_OFF=0` (every check back): if the result changes, an index is out of range. The World's own kernels keep the index clamps (`RUBBLE_RIGID_SHADER_CHECKS_OFF=0` is their fully checked build). Chrome and the other backends always have the checks.

## Where to run: CUDA on Modal
Modal's target is CUDA (`engine-cuda`): nvcc compiles the kernels locally (free), Modal
only runs the binary. No lock, no queue, any number of agents at once, and a runaway
kernel cannot wedge the Mac (`engine/tools/modal/README.md`).
- CUDA tests, examples, benches:
  `engine/tools/modal/gpu_run.sh [--fanout | --fanout-tests] -- <cargo selection>`.
  `--fanout` runs each test binary in a container of its own, `--fanout-tests` each
  test, all at once; one summary and exit code. The build is local Docker (free).
- Vulkan on Modal (wgpu crates, the World, `run.sh --modal`) is opt-in and never after
  a kernel edit: NVIDIA compiles each new kernel version on Modal's hosts (over 28
  minutes for one explicit-beams pipeline).
- The scenario matrix on CUDA: `run.sh --cuda SWITCH` (Modal, no lock).
- The scenario matrix on Metal (`run.sh SWITCH`), Metal, headless Chrome / WebGPU pages and
  wgpu work run on the Mac's GPU, following the rules below.

## Launching runs
1. **CPU reference first.**
2. **Then a tiny GPU run:** a few steps, with a deliberately low budget.
3. **Then the real run.**

Rules for every run on the Mac:
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
