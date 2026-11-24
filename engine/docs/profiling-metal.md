# Profiling on native Metal (Apple Silicon)

One command answers "where does every microsecond of a step go": `scripts/metal-profile.sh`.
The other tools go deeper (Xcode capture, shader profiler) or wider (Instruments timeline).
Everything here runs on the native Metal hosts, which replay the same step plans and the same
Slang kernels as the browser:

- `MetalSim`: the shared rigid step plan (`engine_exec::rigid_plan`), the commands the WebGPU
  host encodes, kernels compiled Slang → MSL in the browser's buffer layout.
- `MetalExplicitBeams`: the explicit node–beam (soft vehicle) plan (`engine_exec::explicit_plan`).

The legacy `MetalWorld` (its own step graph) is not instrumented. The World API (`engine-world`:
`city_scale`, `vehicle_lab`) runs on wgpu; on this machine wgpu runs on Metal, so the Instruments
script (section 4) records it too. The per-kernel tools (sections 1 to 3 and 5) need the native
hosts. The WebGPU side has its own tools: `profiling-webgpu.md`.

All GPU runs go through `scripts/gpu-lock.sh` (the machine's one GPU job, a kill timer): see
[GPU lock](#gpu-lock). Outputs go to the repository's gitignored `out/`.

## Quick start

```bash
cd engine
scripts/metal-profile.sh --scene "rigid:Brick Ring (9k)" --scene cars:8 --tag mine
```

This builds `metal_profile` (engine-scenes), runs it under the GPU lock and writes for each scene
`out/metal-profile/mine/<scene>.{txt,json,trace.json}`:

- `.txt`: the tables below (read this first);
- `.json`: every number (lockstep stats, phases, each replayed op with all its samples), to diff
  before and after a change;
- `.trace.json`: open in <https://ui.perfetto.dev> or `chrome://tracing`: host laps (encode and
  commit, wait), each step's GPU interval, the phase-split steps and the replayed kernels on one
  timeline (host and GPU times are on one clock, `mach_absolute_time`).

Options: `--warm N` (60) steps before timing, `--steps K` (60) lockstep steps, `--split P` (5)
phase-split steps, `--reps M` (7) runs of each replayed op, `--no-replay`, `--out DIR`.
`LIMIT=s` (600) is the kill timer. Scenes: `rigid:<name>` is any `engine_scenes::all_scenes()`
scene without joints or hulls (`Brick Ring (9k|28k|110k)`, `Box Pile (4k|32k)`,
`Box Columns (100k)`, `Pyramid`, `Wall Smash (2k)`, ...); `cars:N` is N procedural sedans
(`explicit_bench`'s fleet, 8 m apart, vehicle contact on), `cars:N:wide` with the wide passes.

What one run measures, per scene:

1. **Lockstep steps**: submit a step, wait for it, K times. Host encode+commit time, host wait,
   wall time, each command buffer's GPU interval (`GPUStartTime`/`GPUEndTime`), and the GPU
   idle time between consecutive steps.
2. **Phase-split steps** (rigid): the existing `step_profiled` split, one command buffer per
   plan phase. Coarse, and it pays a command buffer per phase; use it to rank phases.
3. **Kernel replay** (section 5): every dispatch of the next step alone, from that step's real
   input, M times. Per kernel: median and p90 ms, its share, threads, and a lower bound of the
   bytes it wrote (128-byte lines it changed) and of its write bandwidth.

## 1. Labels: traces that read like the plan

`RUBBLE_METAL_LABELS=1` (on by itself while a capture is requested) names each command buffer
(`rubble step N`, `rubble explicit frame N (33 steps)`), opens a debug group per plan phase
(`broadphase`, `narrowphase`, `coloring`, `primal`, `dual`, `sleep`, ...: the plan's own
`Cmd::Phase` names) and one per dispatch, named after the plan's kernel, its compiled entry when
the plan resolved it to another, its pass and its invocations: `solve_primal_color
(solve_primal_color_compact) pass 3 indirect`, `narrowphase x8192`. Off by default: it adds a few
hundred Objective-C calls a step (host time only, no GPU work). Xcode's capture outline,
Instruments' encoder list and the shader profiler then show these names.

## 2. One-step GPU capture (Xcode)

```bash
RUBBLE_CAPTURE_STEP=80 RUBBLE_CAPTURE_OUT=out/metal-capture/ring9k-step80.gputrace \
  scripts/metal-profile.sh --scene "rigid:Brick Ring (9k)" --no-replay --split 0 --steps 30
```

`MTLCaptureManager` writes a `.gputrace` of exactly step N (frame N for `cars:`): the host waits
for the steps in flight, starts the capture on the queue, encodes and commits step N, waits for
it and stops. N counts steps from the world's start (the warm-up included). Any native program on
`MetalSim` or `MetalExplicitBeams` honours the variables (tests, `explicit_bench`, ...); the
hosts set `MTL_CAPTURE_ENABLED=1` themselves before creating the device.

Open it: `open out/metal-capture/ring9k-step80.gputrace` (Xcode). Then:

- **Replay** (the toolbar's replay button) runs the captured step again on the GPU and turns on
  the **shader profiler** (Xcode 15: "Profile" after replay). The outline's debug groups are the
  plan's phases and kernels (section 1).
- **Per-dispatch time**: the "Performance" view, or each dispatch's row: GPU time per dispatch,
  with the **limiter** (ALU, memory, texture, threadgroup/imageblock, control flow), **occupancy**
  and **ALU / memory utilisation** counters.
- **Per-line cost**: open a dispatch's kernel (the shader editor): each source line shows its
  share of the kernel's samples. With section 3's debug libraries the source shown is the
  `.slang` file; otherwise it is the generated MSL.
- **Memory**: the "Memory" view lists every buffer with its size: the scene's footprint.

## 3. Shader profiler on Slang lines (debug libraries)

The MSL that build.rs generates already carries `#line N "<abs path>.slang"` directives (slangc's
default), so per-line cost can be mapped to Slang without changing the Slang compile. What is
missing at run time is debug information: `newLibraryWithSource` libraries have none.
`RUBBLE_METAL_DEBUG_LIB=1` compiles each kernel offline instead:

```
xcrun -sdk macosx metal -c -gline-tables-only -frecord-sources -ffast-math|-fno-fast-math K.metal
xcrun -sdk macosx metallib K.air -o K.metallib
```

(the kernel's own math mode: fast, or strict for `ENGINE_STRICT_KERNELS`), cached in
`out/metal-debuglib/` by a hash of the MSL and flags, and loads the `.metallib`. Line tables and
the recorded source let the shader profiler attribute samples to the `#line` files, the `.slang`
sources. Combine with a capture:

```bash
RUBBLE_METAL_DEBUG_LIB=1 RUBBLE_CAPTURE_STEP=80 scripts/metal-profile.sh --scene "rigid:Brick Ring (9k)" --no-replay --split 0
```

The first run compiles every kernel offline (about a minute); later runs hit the cache.

**Why not `slangc -g`**: `-g` changes the generated MSL itself (Slang then routes every global
through a `KernelContext` struct and renames parameters), so the profiler would time a different
program than the one that ships. The debug libraries profile the shipped MSL: the Slang output is
byte-identical by construction (no Slang flag changes), and default runs are unchanged (the
runtime compile path is taken unless the variable is set). The offline compile is checked against
the runtime one in [Checks](#checks).

## 4. Instruments: the CPU ↔ GPU timeline

```bash
scripts/metal-trace.sh --time-limit 10s --name ring9k -- \
  target/release/metal_profile --scene "rigid:Brick Ring (9k)" --no-replay --split 0 --steps 300
```

Records Instruments' **Metal System Trace** with `xcrun xctrace record --launch` under the GPU
lock (kill timer `LIMIT`), with labels on, into `out/metal-trace/ring9k.trace` (open it in
Instruments for the timeline: command buffer submission, encoders, GPU channels, the driver,
CPU threads). Then `scripts/metal-trace-summary.py` exports the trace's GPU tables with
`xcrun xctrace export` (`--toc`, then `--xpath '/trace-toc/run[@number="1"]/data/table[@schema="…"]'`)
into `out/metal-trace/ring9k/*.xml` and writes `summary.{txt,json}`: for every interval table,
busy time (the union of its intervals), idle time, the gaps between intervals (median, p90,
max) and time per label (channel, command buffer, encoder). `--summarize X.trace` re-reads a
trace. It works for any Metal process, wgpu-on-Metal ones too (`city_scale`, `vehicle_lab`).

## 5. Kernel replay: per-kernel isolated timing

On Apple Silicon `MTLCounterSampleBuffer` samples only at stage boundaries, not between
dispatches, so a step's per-kernel cost cannot be read from one run. The replay measures each
dispatch alone instead (`profile::replay`, `MetalSim::replay_ops`,
`MetalExplicitBeams::replay_ops`). Measurement only: it never runs in the sim loop, and the live
world does not advance (the ops are built from the next step's commands without submitting it;
the explicit planner's state is restored).

1. Copy every buffer of the host (the step's input: bindings, upload ring, parameter ring) into
   a pristine set and a work set.
2. **Whole step**: M times, restore the work set and run the step's ops in one command buffer
   (one serial compute encoder per run of dispatches, blits between): the replay's reference
   time, comparable to the lockstep step.
3. **Each op alone**, in step order: copy aside the buffers it may write (the ones not `const`
   or `constant` in its MSL signature), run it in its own command buffer, restore those
   buffers, run again: M runs from the same input. The work set is then the state the op left,
   the next op's input. Indirect dispatches read their workgroup counts from the replayed
   state, as in the step.
4. A **launch floor**: an empty kernel's command buffer, the fixed cost every op's time
   includes.

Columns: `ms` is the median of M; `share` is of the sum of the ops' medians; `changed` is
128-byte lines whose content the op changed ×128, a lower bound of its DRAM writes (it does not
see reads, nor writes of a value a line already held); `>=GB/s` is changed bytes over the median,
a lower bound of its write bandwidth. Real traffic, limiter and occupancy come from the shader
profiler (section 2).

Reading the gap between the ops' sum and the whole step: each op alone pays the launch floor and
starts with the caches holding only what the restore blit left; in the step, serial dispatches
in one encoder wait for each other (a barrier: the GPU drains between dependent dispatches) but
pay no command-buffer launch. So `sum - n × floor` estimates the step without launch costs, and
`whole step - (sum - n × floor)` is what the step pays for barriers and drain between dependent
dispatches, net of cache effects.

## GPU lock

`scripts/gpu-lock.sh [--limit s] -- CMD...` runs CMD as the machine's one GPU job through
`with-gpu-lock.py`: the flock `~/.rubble/gpu.lock` that native binaries take
(`hold_gpu_lock`), with the command in its own process group, killed after `--limit` seconds.
The command runs with `RUBBLE_GPU_LOCK=0` (its binaries must not wait on the flock the wrapper
holds). Other sessions' GPU jobs queue on the same lock: run profiles in the background.

## Checks

2026-11-19 and 2026-11-22, M3 Max (`tools/metal-profile/baseline.sh checks`):

- **Soft cars (`cars:8`):** the same end state, bit for bit, across two default runs, labels
  on and debug libraries on, at the same GPU time (0.712–0.714 ms a frame). Labels and debug
  libraries change nothing on the explicit solver.
- **Rigid (`MetalSim`, Brick Ring 9k):** labels and debug libraries add no GPU time
  (1.73–1.77 ms). Labels roughly double host encode time (0.13 against 0.06 ms), which is
  why they are opt-in. Two free-running default runs end in different states, so compare
  only from one input (the replay's state hash).
- **One step from one input** (replayed M = 7 times): Brick Ring 9k and Box Pile 32k end in
  one state, while **Brick Ring 28k ends in 7 different states.** The rigid step is not
  reproducible from a fixed input at that scale.

## Baseline (M3 Max, 40-core GPU)

2026-11-22, `main` d5a1d480 plus this branch, `tools/metal-profile/baseline.sh small`
(`out/metal-profile/small/`). Rigid: 4 iterations. Cars: one 60 Hz frame (33 solver steps
at 2 kHz).

| Scene | Lockstep GPU | Replay whole step | Primal colour solve | Next largest |
|---|---|---|---|---|
| Brick Ring 9k (9,132 bodies) | 1.75 ms | 1.41 ms | 57 % (44 dispatches) | broadphase_query 7.5 %, narrowphase_fresh 7.1 % |
| Brick Ring 28k (11.7k awake) | 2.55 ms | 2.57 ms | 44 % | solve_dual 9.8 %, narrowphase_fresh 7.1 % |
| Box Pile 32k (32k awake) | 1.59 ms | 2.73 ms | 42 % | narrowphase_fresh 16.6 %, sleep_island_gate 7.8 % |
| cars:1 | 0.68 ms | 0.68 ms | — | one kernel (`explicit_beams_wide_nosurf`, 512 threads) |
| cars:8 (2,936 nodes, 17,360 beams) | 0.72 ms | 0.72 ms | — | `explicit_beams_solo_wide_nosurf` 96 % |
| cars:8:wide | 0.72 ms | 0.72 ms | — | the same |

In lockstep the GPU sits idle 0.16–0.31 ms between steps: the host round trip, which
pipelined stepping hides. On native Metal the isolated sum is 0.3–0.9 ms larger than the
whole step (each op alone starts with cold caches).

## Optimization opportunities (ranked)

Leads with their evidence. Each needs a capture for its limiter before any change, and the
scenario matrix to admit it.

1. **The primal colour solve is 42–57 % of every rigid step** (44 dispatches at 9k–28k).
   Its write bandwidth is 4–10 GB/s, so it is not bandwidth-bound. Capture it
   (`RUBBLE_CAPTURE_STEP` with `RUBBLE_METAL_DEBUG_LIB=1`) for occupancy and limiter, and
   look at per-colour dispatch overhead against per-thread work.
2. **A car's frame barely grows with car count**: 0.68 ms for 1 car against 0.72 ms for 8.
   That is about 20 µs for each of the 33 serial solver steps, at about 1 GB/s. The frame is
   bound by the latency of serial steps, not by work. More cars are nearly free; one car is
   only faster with fewer serial barriers in a step (`measurements/vehicle-perf-first-principles.md`).
3. **`narrowphase_fresh_copy_persistent` on piles** (16.6 % of Box Pile 32k).
4. **`sleep_island_gate`: 8 dispatches, 5–8 % of rigid steps,** at 0.1–0.6 GB/s.
5. **The 28k ring's step is not reproducible from one input** (Checks). This is not a
   cost, but it bears on the bit-exact test mode: find which kernel's atomics order leaks
   into state before relying on bit-for-bit A/Bs of rigid scenes at that scale.
