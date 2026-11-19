# GPU runs on Modal

Run Rubble's GPU tests, examples and A/Bs on a cloud NVIDIA GPU instead of this Mac's
one GPU, so several agents can run them at once.

```bash
engine/tools/modal/gpu_run.sh [options] -- <cargo selection> [-- <run args>]
```

Two routes, picked by the selection:

- **WebGPU (Vulkan)**: any engine-webgpu / engine-world test or example, unchanged, on
  the NVIDIA driver's Vulkan. The driver compiles each pipeline on the GPU host (billed);
  the runner warms them in parallel and caches them per kernel version.
- **CUDA** (`-p engine-cuda`): the same Slang kernels, built to device code locally in
  Docker by nvcc; the GPU host only runs them (seconds). `examples/explicit_ab` is the
  old-vs-new harness.

## Which route for which job

Measured on 2026-11-14 (Modal T4, M3 Max host building in OrbStack Docker; Modal bills
the container seconds; the build is free).

| Job | Route | Wall (local build + Modal) | Billed |
|---|---|---|---|
| Old vs new bit for bit, CPU reference, per-kernel ms (explicit solver: parked, crash, tumble, 32 apart, 8 / 6 / 2 in contact) | CUDA `explicit_ab` | 18-23 s a scenario (build up to date); first build of a worktree 6 min | 2.7-9 s, $0.001-0.003 |
| The same after editing a kernel | CUDA `explicit_ab` | local rebuild of the modules you list (`ENGINE_CUDA_EXPLICIT_MODULES`, 8 big modules: ~6 min) + 20 s | seconds |
| WebGPU tests/examples (the engine-webgpu / engine-world suites, `vehicle_lab --ab-switch`, World coupling), kernel version already warmed on the host's driver | WebGPU | 39 s `vv_grid_gpu` (was 1,203 s cold), 57 s `explicit_vehicles` | 13.5 s / 41 s, $0.004 / $0.012 |
| The same, kernel version new on that driver (`--allow-cold-compile`) | WebGPU | cold warm-up 18 min (16 cores; the slowest module alone 17 min) + the run | about $0.43 |
| Tests without the explicit kernels (e.g. `explicit_loop_cap -- a_runaway`) | WebGPU | 34 s | 5.6 s, $0.0013 |
| Final timings | the Mac | | |

1. **Build** in local Docker (x86_64 Ubuntu 22.04, the CUDA build image with Rust and
   slangc; emulated on Apple silicon; free). Cargo's target dir is a Docker volume of
   this worktree's own (`rubble-gpurun-target-<hash of the worktree path>`), so builds
   are incremental and worktrees never take each other's outputs. `CARGO_BUILD_JOBS`
   defaults to 6.
2. **Pack** only the stripped executables, engine-webgpu's `gpu_adapter` probe and the
   small run-time files they read (`gpu_run_pack.py`: the `.truck` acceptance cars, the
   vehicle-lab scripts, the generated WGSL the kernel variants load at run time, and
   `@sedan`'s JBeam text) into `engine/target/gpu-run/<unique id>/archive.tar.gz`
   (about 20 MB). No source tree, no credentials.
3. **Run** in one single-use Modal GPU container (`gpu_run.py`): `gpu_adapter` first
   (prints the adapter; the run stops if there is none or it is a software
   rasteriser), then the selection, its output streamed. The Modal CLI runs in Docker
   with only that run directory and `gpu_run.py` mounted, the login volume read-only.
   Tests run in their crate's directory (created in the container), as with
   `cargo test`.
4. Exit with the run's exit code (124: the time cap killed it; 125: no GPU within the
   queue limit) and delete the run directory (`--keep` keeps it). Ctrl-C stops the
   Modal call too.

Every invocation has its own run directory, its own ephemeral Modal app run and its
own container: calls from different worktrees, or the same one, run in parallel. Two
calls from one worktree share its target volume, so their builds take turns (cargo's
lock); the GPU runs still overlap.

**Hard time cap:** each run's container is killed after `--timeout` seconds, default
**600 (10 min)**, at most 1800. Inside, the running process is stopped 45 s before
the cap so the summary and the shader cache still get written.

## On another branch

The runner needs engine-webgpu's `gpu_adapter` bin
(`engine/crates/engine-webgpu/src/bin/gpu_adapter.rs` and its `[[bin]]` entry in
`engine/crates/engine-webgpu/Cargo.toml`); `gpu_run.sh` stops at once with that message
when a branch lacks it. Merge or cherry-pick `tools/modal-gpu-runner` (the runner, the
bin, the README).

Quick check that a branch builds and starts on Modal (no kernels compiled, ~1 min once
built):

```bash
engine/tools/modal/gpu_run.sh --timeout 120 -- -p engine-world --test explicit_vehicles -- --list
```

## Options

| Option | |
|---|---|
| `--gpu T4\|L4\|L40S` | Default T4. L4 (Ada) may queue: on 2026-11-14 an L4 call waited 12 min without a container. L40S only with a reason (about 2.6x T4) |
| `--timeout SECS` | Hard cap on the container, default 600 |
| `--queue SECS` | How long to wait for a GPU before giving up, default 300 (queueing is not billed) |
| `--env K=V` | A variable for every run (repeatable) |
| `--variant NAME:K=V,K2=V2` | Run the selection once per variant in the same container (repeatable). `NAME:` alone is the unchanged run. A comma part without `=` stays in the value before it (`EV_AB_MASKS=1,2`) |
| `--repeat N` | All variants N times, interleaved (a,b,a,b), for the spread |
| `--timing` | Print the directional-timing banner without variants |
| `--monitor SECS` | While a process runs, a line of GPU use and the process's CPU use every SECS (default 30; 0 off). GPU 0 % with one core busy is the driver compiling |
| `--no-shader-cache` | Neither read nor write the shared shader cache |
| `--data PATH` | An extra repo-relative file or directory to ship (32 MB cap, not counting the generated WGSL) |
| `--sedan DIR` | The generated sedan's JBeam export (default `~/Development/rubble-artifacts/sedan-live/mods/sedan_new`). Shipped only when an argument, `--env` or `--variant` says `@sedan`, which becomes `/scenarios/sedan_new`; only `files.json`, `*.jbeam` and `*.pc` (under 1 MB), never the `.glb`/`.rskin` |
| `--fanout` | One container per executable of the selection (each test binary), all at once (see Fan-out) |
| `--fanout-tests` | One container per test, all at once |
| `--matrix` | The scenario matrix, one container per row (`tools/scenario-matrix/run.sh` calls it) |
| `--build-only` | Build and pack, no Modal call |
| `--keep` | Keep `engine/target/gpu-run/<id>/` |

The selection is what you would give `cargo test` / `cargo build` from `engine/`:
`-p CRATE --test NAME` (several `--test` are fine), `--example NAME`, `--bin NAME`,
`--features ...`. Always release. Arguments after the inner `--` go to every
executable (libtest's filter and flags for tests; the example's own arguments). Tests
run in their crate's directory, as `cargo test` does, one test thread unless you pass
`--test-threads`, and with `--nocapture` (unless you pass `--show-output`), so output
streams as it happens; examples and bins run in `engine/`. When the time cap kills a
binary, the runner names the test that was running and reprints its last 60 lines;
the summary marks it KILLED and any binary after it NOT RUN (exit 124). The container sets
`RUBBLE_GPU_LOCK=0` (one job per container), `WGPU_BACKEND=vulkan` and
`SEDAN=/scenarios/sedan_new`.

## Fan-out: many GPU containers from one build

What the Mac does and what Modal does: the Mac builds (free, local Docker) and runs
CPU-only tests natively; Modal runs only GPU work, every job in a T4 container of its
own (the smallest GPU Modal has), so no two runs share a GPU and nothing queues.

```bash
# Each test binary in its own container, all at once
engine/tools/modal/gpu_run.sh --fanout -- -p engine-webgpu \
  --test explicit_pair_reuse_gpu --test vv_grid_gpu --test debris_retire_gpu
# Each test in its own container (names listed at build time; the run args' filter applies)
engine/tools/modal/gpu_run.sh --fanout-tests -- -p engine-world --test explicit_vehicles
# The scenario matrix: each row in its own container (the default for run.sh)
engine/tools/scenario-matrix/run.sh perf=8388608
```

1. One build and pack, as for a single run.
2. The archive uploads once to the Volume `rubble-gpu-run-archives`, named by its
   SHA-256: a rerun of the same build uploads nothing; archives older than 2 days are
   pruned.
3. Every shard is spawned at once (Modal starts as many containers as the workspace's
   GPU limit allows and queues the rest). Each streams its output prefixed with
   `[label]`.
4. One summary (status and container seconds a shard, its GPU and driver), the summed
   billed estimate, and the first failure's exit code.

Cold kernels: a shard whose kernel version is not compiled on its host's driver stops
at once (the guard, ~5 s billed, `COLD` in the summary). With `--allow-cold-compile`
a fan-out first runs **one** warm-up container that compiles every pipeline and saves
the cache; then the shards, which start warm. Modal mixes driver versions (610.57.04
and 620.70.06 seen; the cache is per driver), so a shard that lands on the other
driver reruns up to twice hoping for a warm host (~5 s each), and only then compiles
its own pipelines. Never let every shard compile: 12 cold matrix rows would each pay
the ~18-minute warm-up.

The scenario matrix's own pieces: `--matrix` builds `-p engine-world --example
vehicle_lab`, ships `tools/scenario-matrix` (matrix.py, scenes) and the generated
sedan (`CAR`, default `~/Development/rubble-artifacts/sedan-live/mods/sedan_new`),
forwards `FRAMES ROUNDS RETRIES THRESHOLD NOISE_MS WARM_MS PROFILE_SPLIT`, and gives
each row `engine/out/scenario-matrix/ROW.fx`, uploaded once by hash to the Volume
`rubble-scenario-states`. Each container runs matrix.py for its row (both halves, the
retries and the median in that one container, so the speed comparison is between
runs on one GPU), and the local side prints one table and one verdict. `run.sh --make`
saves new states on Modal and writes them back to `engine/out/scenario-matrix/`.
Options for gpu_run go in `MODAL_OPTS`, e.g. `MODAL_OPTS="--allow-cold-compile
--timeout 1500"` for a kernel version new on Modal. This route is opt-in
(`run.sh --modal`): on 2026-11-19 the warm-up for one kernel version hit the 1800 s cap
on both drivers with 16 cores (one `explicit_beams_split_member_wide` pipeline compiled
for over 1,700 s on one thread, so more cores do not help), about $2 across tries with
no states made. Kernel edits go to the CUDA route instead.

## Cold kernel compiles (WebGPU route; read this first)

wgpu creates each compute pipeline from SPIR-V, and NVIDIA's driver compiles it on the
CPU when the pipeline is created: one core a pipeline, but different pipelines in
parallel (8 modules: 19.4 s serial, 7.7 s on 8 threads). Most of Rubble's kernels take a
second or less (the rigid World's 255: 54 s on 16 cores). The current explicit-beams
modules take minutes each on a T4 host (driver 620.70.06):

| Module (WGSL) | Cold compile |
|---|---|
| `explicit_beams_wide_nosurf`, `_solo_wide_nosurf`, `_contact_big_wide_nosurf` | 171 / 179 / 189 s |
| `explicit_beams_solo` (692 kB, with surface contact) | 504-518 s |
| `explicit_beams_contact_wide_nosurf` | 1024 s |
| `explicit_ref_solo` (the frozen reference, 96 kB) | 2.5-3.3 s |

Naga's WGSL→SPIR-V of the same module is 0.1 s; wgpu's runtime checks off (bounds
checks, loop bounding, integer-division checks, or all) change nothing (504-518 s), so
it is the driver's compiler on a very large single entry point (the current modules
are one entry point with ~475 functions to inline; the reference had 59). Nothing in
the build or the wgpu options shortens it without changing the kernels.

What the runner does about it:

- **One executable name.** NVIDIA's shader disk cache keeps one file per executable
  *basename* (measured: the same basename in another directory hits; another basename
  misses). Every process in the container runs as `.../rubble-gpu`, so every test and
  the warm-up share one cache file. (Before this, a pipeline compiled by one test
  binary was cold for every other binary: tonight's 1,203 s `vv_grid_gpu` run was its
  own cold compile of the contact module, 1024 s, plus the rest.)
- **Parallel warm-up.** Before the selection, `gpu_adapter --warm` creates on every
  core the pipelines `ExplicitBeamsGpu::compute_specs` lists for the usual fleets (the
  repository sedan with the shipped settings, `@sedan` when given; 1, 2 and a
  group-sized fleet; with and without coupling: 18 pipelines) and the rigid World's
  kernels. A cold kernel version then costs about its slowest module (18 min on 16
  cores), not the sum. Kernels outside those (the bitwise tests' `_det` / `_probe` /
  baseline / reference variants, other layouts) compile when first used.
  `--no-warm` skips it.
- **Cache per kernel version.** The cache lives in the Modal volume
  `rubble-gpu-run-shader-cache`, a snapshot per kernel version (the hash of the
  generated explicit WGSL) and driver, so agents on different kernels never overwrite
  each other. A version new to the volume starts from the newest snapshot of any
  version (the rigid and unchanged modules hit). Within one version the last run to
  finish wins.
- **Cold guard.** A run whose kernel version has no completed warm-up on the host's
  driver stops at once (exit 3, ~4 s billed) with the estimate, unless you pass
  `--allow-cold-compile` (then give it `--timeout 1500` or more and `--cpus 16`).
  Modal's hosts run different drivers (610.57.04 and 620.70.06 seen), each warmed
  separately, so a warmed version can be cold on another host.
- `--cpus N` (default 8): the warm-up uses them all.

Tonight's hit rate, before these fixes: close to zero for the big modules. Each test
binary had its own cache file, so a pipeline was warm only for a rerun of the same
binary on the same driver (e.g. `explicit_beams_bitwise` reused the 2 big pipelines its
earlier run had compiled); the warm-up's entries reached no test; and one shared
snapshot per driver was overwritten by whichever agent finished last (a 39 MB warm
snapshot was replaced by another agent's 26.5 MB one 8 minutes later).

## CUDA route: old vs new in seconds

`-p engine-cuda` selections build device code locally (Slang → CUDA → fatbin in
engine-cuda's build.rs; SASS for the selected GPU only, so nothing JIT-compiles) and the
GPU host only runs it: no shader cache, no warm-up, no cold guard.

```bash
# The modules a run loads (CPU only; on the Mac: ENGINE_CUDA_EXPLICIT_MODULES=none
# cargo run --release -p engine-cuda --example explicit_ab -- --modules --scenario contact --count 8)
export ENGINE_CUDA_EXPLICIT_MODULES="explicit_beams_wide_nosurf,explicit_beams_solo_wide_nosurf,explicit_beams_contact_wide_nosurf,explicit_beams_contact_big_wide_nosurf,explicit_beams_aero_wide_nosurf,explicit_beams_solo_aero_wide_nosurf,explicit_beams_contact_aero_wide_nosurf,explicit_beams_contact_big_aero_wide_nosurf,explicit_beams_vv_*,explicit_beams_wide_*"

engine/tools/modal/gpu_run.sh -- -p engine-cuda --example explicit_ab -- \
  --scenario contact --count 8 --frames 30 --a pair_reuse=0 --b pair_reuse=0.1
engine/tools/modal/gpu_run.sh -- -p engine-cuda --example explicit_ab -- \
  --scenario contact --count 2 --car @sedan --frames 30 --a vv_grid=0 --b vv_grid=1
engine/tools/modal/gpu_run.sh -- -p engine-cuda --example explicit_ab -- \
  --scenario tumble --at 0.3 --frames 30 --a wide=0 --b wide=1
engine/tools/modal/gpu_run.sh -- -p engine-cuda --example explicit_ab -- --spec out/my-ab.json
```

`ENGINE_CUDA_EXPLICIT_MODULES` (comma list; a trailing `*` is a prefix) builds only those
modules: the list above (the procedural, repository and generated sedans, without
surface contact) builds in ~6 min from a cold target; every module is over an hour. A
run that needs a module left out stops and names it. Unset it to build everything.

Scenarios: `parked`, `crash` (56 km/h into a wall), `tumble` (dropped, pitched, rolled,
spinning), `apart` (N side by side), `contact` (N nose to tail, alternate ones driving
into the next), `overlimit` (a chain larger than the small groups' limit: the big-group
module). Cars: `procedural` (default), `repo`, a JBeam export directory, `@sedan`.
`--at S` starts both hosts from the CPU reference's state at S seconds. Switches per
variant (`--a`, `--b`, `--c`, `--d`, or JSON `{"scenario", "car", "count", "frames",
"steps", "at", "variants": [{"name", "switches": {...}}]}`): `pair_reuse=M`,
`vv_grid=0|1`, `wide=0|1`, `contact=0|1`, `contact_margin=M`, `env.NAME=V` (read at
upload), `param.W=V` (a planner word, e.g. a branch's `W_BEAM_ORDER`). Each variant gets
its own host built from the same state; the harness prints, per variant, the CPU
comparison (positions; broken/bent beams within the parity tests' tolerances), GPU ms
a frame, ms per kernel, a rerun from its own snapshot (bit for bit), and every state
word (`engine_exec::explicit_plan::state_words`, the same set the WebGPU host compares)
against the first variant. Exit 0 only if all hold.

Measured (T4, 2026-11-14, every run PASS, variants bit for bit): parked 60 frames;
crash, `wide=0` vs `1`, 224/224 broken; tumble from 0.3 s; 32 apart; 8 in contact and the
6-car over-limit chain (`ContactBig` 6-8.6 ms a frame); 2 generated sedans in contact
(31.5 ms a frame). Not available on CUDA (the harness says so): `coupling_list`,
`coupling_fold`, `piece_windows` (no World coupling or piece labels in these scenarios;
use `vehicle_lab --ab-switch` on the WebGPU route) and `ablate` (no ablation modules are
built for CUDA).

What a stream branch needs before it runs here (checked 2026-11-14):

- `perf/beam-records`: nothing beyond merging: `W_BEAM_ORDER` is a planner word and the
  kernel change is in `explicit_beams.slang`; `param.96=0` / `param.96=<on>` A/Bs it
  (or add a named `beam_order` switch).
- `perf/contact-groups`: the split group path is WebGPU-host code. To run on CUDA it
  needs (1) its modules (`explicit_beams_split_{member,group,tests}{,_wide,_aero,_aero_wide}`,
  built today only by engine-webgpu's build.rs) in `engine_exec::explicit_plan::MODULES`
  or a list engine-cuda's build.rs compiles; (2) its dispatch sequence and parameter
  copies (`dispatch_group_split`, copies steps + 1, steps + 2 + s, 2 steps + 2 + s) moved
  into `FramePlanner::commands` as `Cmd`s with `Kernel` variants of their own; (3) its
  indirect arguments (offsets 64/80/96) are written within a substep and read in the same
  substep, which the CUDA host's one readback a plan does not cover: either a readback a
  substep (a stream sync each, ~17-34 a frame) or grid-stride entries launched at a
  fixed grid that read the count on the device (as terrain's `_gs` entries). The CUDA
  host now reads the whole `ARGS_BYTES` buffer (it read 8 words: the larger groups'
  offset 48 indexed past them, so `contact` and `overlimit` panicked).
- `perf/selfcol-per-pair`: a design document only so far; its indirect walk/pairs
  dispatches will need the same as (3).

## The three kinds of run

### 1. Correctness: GPU against the CPU reference, new path against old, bitwise

Bitwise A/Bs on one state are built into the tools; run them as they are.

```bash
# The current kernel against the frozen reference and baseline kernels, frame by
# frame, bit for bit (one test; cold: several kernel modules, see above)
engine/tools/modal/gpu_run.sh --timeout 1500 -- -p engine-webgpu --test explicit_beams_bitwise \
  -- sedan_alone --nocapture

# The coupling's candidate list against every node testing every shape, on the
# mountain crash state at t = 2 s: 20 frames each way from one snapshot, every state
# word compared ("state bit for bit: true"; the example fails if not).
engine/tools/modal/gpu_run.sh --timeout 1500 -- -p engine-world --example vehicle_lab -- \
  --mode gpu --measure mountain-perf --secs 3 --ab 2 --ab-switch coupling
# --ab-switch fold | reuse | pieces | (none: surface culls); --ab T1,T2 for several times.
# --car @sedan: the generated sedan's export instead of the showcase car.
```

For a switch that is an environment variable, run the variants side by side (each is
a fresh process on the same GPU and the same kernels, so the second is warm) and
compare what they print:

```bash
engine/tools/modal/gpu_run.sh --variant old:RUBBLE_COUPLING_LIST=0 --variant new: \
  -- -p engine-world --test explicit_vehicles -- --nocapture
```

### 2. The existing GPU test suites

```bash
engine/tools/modal/gpu_run.sh --timeout 1800 -- -p engine-webgpu \
  --test explicit_beams_bitwise --test explicit_pair_reuse_gpu --test explicit_contact_gpu \
  --test showcase_car_gpu --test vv_grid_gpu --test debris_retire_gpu --test explicit_loop_cap
engine/tools/modal/gpu_run.sh --timeout 1800 -- -p engine-world --test explicit_vehicles \
  --test vehicle_lab_piece_windows --test vehicle_lab_pose_time --test vehicle_lab_reset
```

On a warmed kernel version these take a minute or two. On a new one, pass
`--allow-cold-compile --cpus 16 --timeout 1500` once (the warm-up, ~19 min); the
bitwise tests' `_det` / `_probe` / baseline / reference variants are not in the warm-up
and still compile when first used (minutes each for the `_det` ones). Tests that only use small
kernels are quick, e.g. `--test explicit_loop_cap -- a_runaway` (5.6 s in the
container). Source-census tests read the Slang sources they check, which are shipped
for them (`explicit_loop_cap`: `engine/slang/kernels` and `engine/slang/physics`); a
new test that reads other sources needs its entry in `TARGET_DATA` in
`gpu_run_pack.py`.

### 3. Timing A/B (directional only)

```bash
# Old and new in one container, interleaved three times
engine/tools/modal/gpu_run.sh --repeat 3 \
  --variant old:RUBBLE_COUPLING_LIST=0 --variant new: \
  -- -p engine-world --example vehicle_lab -- --mode gpu --measure mountain-perf --secs 4

# fleet.sh's run (explicit_vehicles_bench with the generated sedan), "apart" mode;
# "town": drop EV_NO_CONTACT / EV_NO_VV. EV_VV_GRID=0 etc. as --variant.
engine/tools/modal/gpu_run.sh --timing \
  --env EV_ETK=@sedan:sedan/sedan.pc --env EV_SPAN=1 --env EV_SECS=3 \
  --env EV_NO_CONTACT=1 --env EV_NO_VV=1 \
  -- -p engine-world --example explicit_vehicles_bench -- 1 8 32
```

Every run with variants (or `--timing`) prints, before and after: **timing is
directional only**. A T4 through Vulkan inside Modal's gVisor sandbox is not the M3
Max under Metal nor Chrome's WebGPU: different architecture, driver, subgroup width,
memory system and submission cost. Use it to rank variants **run in the same
container** (`--variant`, `--repeat`) and to catch large regressions; final numbers
come from the Mac. Exclude the first run of a cold kernel (its compile lands in
whatever it times) or let a first variant warm it.

## What compares with the Mac

| | Comparable? |
|---|---|
| Pass/fail of a test; a bitwise A/B on one GPU ("state bit for bit") | Yes: the same Rust host, the same Slang → WGSL kernels and command lists |
| GPU vs CPU reference within the tests' tolerances | Yes, it is what the tests check. NVIDIA's results are not bit-identical to the Mac's (other float hardware and shader compiler), so a check that pins exact Mac bits would differ here |
| Timing between variants in one container | Directionally |
| Absolute ms, ms/step against Mac numbers, scaling curves | No |
| Chrome / the browser, Metal-only paths (`engine-metal`, `metal_interop`) | Not run |

Known results on Modal (T4, driver 620.70.06, 2026-11-15): `explicit_vehicles` passes 7
of 10; three coupling bit-exactness checks fail, the same way on two runs:
`a_vehicle_contact_margin_changes_no_result` (13 coupling record words differ with an
8 m margin at step 90), `the_coupling_candidate_list_is_bit_exact` (10 words, town step
90) and `the_coupling_candidate_list_meets_fast_and_thin_bodies` (3 words, step 0). The
runner did not cause them (the other bit-for-bit tests pass); whether they pass on the
Mac was not checked here.

## Cost and turnaround

Modal bills per second while the container runs; `gpu_run.py` prints an estimate at
the end. With the default 8 cores and 8 GiB: T4 ≈ $0.00029/s (≈ $0.017/min); 16 cores
≈ $0.024/min (GPU prices of 2026-10 from modal.com/pricing, plus CPU
$0.0000131/core/s and memory $0.00000222/GiB/s). The 10-minute cap bounds a T4 run at
about $0.17.

Measured 2026-11-14/15 (M3 Max host, OrbStack Docker with Rosetta, Modal T4):

| Step | Time |
|---|---|
| First build in a worktree: engine-webgpu test / engine-world test / engine-cuda (8 explicit modules) | 9 min / 6 min / 6 min |
| Rebuild, nothing changed / a small Rust change | 3-5 s / 20-26 s |
| Modal: upload 30-45 MB, container start, adapter check | 6-15 s (images build once, ~50 s, then Modal caches them) |
| CUDA `explicit_ab`, one scenario, two variants | 18-23 s end to end, 2.7-9 s billed |
| WebGPU, warmed kernel version: warm-up from the cache | 5 s (18 explicit pipelines 3.8 s, 255 rigid 0.9 s) |
| WebGPU, warmed: `vv_grid_gpu` / `explicit_vehicles` | 39 s / 57 s end to end; 13.5 s / 41 s billed |
| WebGPU, new kernel version: parallel warm-up, 16 cores | 1,103-1,157 s (the contact module alone ~1,024 s), ~$0.45 |
| `explicit_loop_cap -- a_runaway` (no explicit kernels) | 34 s (5.6 s billed, $0.0013) |

## What does not work

- macOS-only code: `engine-metal`, `metal_interop`, anything that wants Metal.
- The browser, Chrome WebGPU, the web demos; Node tests (`*.test.mjs`).
- Shell wrappers (`fleet.sh`, `with-gpu-lock.py`): pass the cargo command they run
  (see above). `perl -e 'alarm ...'` is not needed: `--timeout` is the kill timer.
- engine-world tests whose data is not shipped (too large, or outside the repo):
  `town_columns_probe`, `town_solver_probe`, `glass_tower_felling_probe` (need
  `RUBBLE_TOWN_PACK`, a town pack outside the repo; they return early without it), and
  `lab_api` / `stress_block`'s `fracturedTower` parts (`engine-webgpu/generated/`,
  not in the repo; they skip). Shipped for engine-world: `fracture/packs`,
  `stress-oracle/packs`, the `.truck` cars, and `masonry-arch.glb` (4.7 MB) for
  `equivalence`, `threshold`, `destruction` and the `quickstart` example. Not shipped
  (52-170 MB `.glb`): `veneer-house` (`destruction`'s house cases fail), `galleon-stern`
  (`broadside`, `broadside_take`, the broadside play preset) and `timber-yard`
  (`timber_yard`, `timber_take`, the timber-yard preset): those cannot run here.
- Files outside the repo other than `@sedan`: put them in the repo (a gitignored
  `out/` works) and pass `--data`. `RUBBLE_INSPIRATION` (Rigs of Rods content) is not
  shipped, so tests that need it skip as they do without it locally.
- Preemption: Modal may preempt a GPU container; the call is then restarted from the
  start (with a fresh time cap, billed again). Seen twice in long cold runs; Modal offers no
  non-preemptible GPU containers.
- Stack traces of a hung process: gVisor does not allow ptrace (no gdb); `--monitor`
  tells a compile (GPU 0 %, a core busy) from GPU work.
- `engine-cuda` (the CUDA host) has its own runner, `run_cuda_tests.sh` (below).

## Under the hood

Modal's GPU containers mount the NVIDIA driver with all capabilities, including its
Vulkan ICD (`/etc/vulkan/icd.d/nvidia_icd.json` → `libGLX_nvidia.so.0`). The ICD needs
the Vulkan loader plus `libEGL.so.1`, `libX11` and `libXext` from the image; without
`libEGL` it loads but returns no `vkCreateInstance` ("Found no drivers"). The image
(`gpu_run.py`) is Debian bookworm with `libvulkan1 libegl1 libx11-6 libxext6`
(`--no-install-recommends`: no Mesa), and `VK_ICD_FILENAMES` names only the NVIDIA ICD,
so no software adapter can be picked; `gpu_adapter` checks anyway (exit 3 on llvmpipe,
lavapipe, SwiftShader or a CPU device type). Headless: no display. The adapter it
prints:

```
adapter: Tesla T4 | backend Vulkan | type DiscreteGpu | driver NVIDIA 620.70.06 | vendor 0x10de device 0x1eb8
```

`gpu_adapter --compile NAME...` times the shader module and pipeline creation of the
generated `NAME.wgsl` modules (diagnostics; note the engine passes some modules through
`uniformize_workgroup_lanes` first, so these are not always its exact SPIR-V).

## engine-cuda (`cuda_tests.py`, `run_cuda_tests.sh`)

The CUDA host's own suite, same idea (build in Docker, run on Modal). It moves the
kernel compile to build time (nvcc embeds SASS for the T4 and Ada), but that compile is
as slow: in this Docker build each current explicit-beams module took about 3 min of
nvcc/ptxas under Rosetta, two at a time, and the engine-cuda build had not finished
after 53 min. It is also a narrower host (no World, rigid coupling or fleets), so an
A/B harness there was not built; use the wgpu runner above. See
`engine/TOOLCHAIN.md` → CUDA tests on Modal. `modal.sh` is the plain Modal CLI in Docker
(`modal.sh setup` logs in once; the login lives in the `rubble-modal-config` volume).
