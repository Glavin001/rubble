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
| `--sedan DIR` | The generated sedan's JBeam export (default `engine/out/gen/sedan_new (the generated sedan from the current code, AGENTS.md)`). Shipped only when an argument, `--env` or `--variant` says `@sedan`, which becomes `/scenarios/sedan_new`; only `files.json`, `*.jbeam` and `*.pc` (under 1 MB), never the `.glb`/`.rskin` |
| `--fanout` | One container per executable of the selection (each test binary), all at once (see Fan-out) |
| `--fanout-tests` | One container per test, all at once |
| `--matrix` | The scenario matrix, one container per row (`tools/scenario-matrix/run.sh` calls it) |
| `--matrix-cuda` | The scenario matrix on CUDA, every row in one container with 1 core and 2 GiB (`run.sh --cuda`; see below) |
| `--base REF` / `--head REF` | `--matrix-cuda`: also ship `explicit_replay` built from commit REF (`build_ref.sh`, kept in `engine/target/gpu-run-refs/`) for `matrix.py --two-builds`; the head defaults to the working tree's build (see "Two builds, head to head") |
| `--pack N` | Matrix: the rows in N containers, in turn within each (default one per row; 1 for `--matrix-cuda`) |
| `--cpus N` / `--mem GIB` | The container's cores (fractional allowed) and memory: default 8 and 8 (the Vulkan driver compiles on the CPU); 1 and 2 for `--matrix-cuda` |
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

The scenario matrix's own pieces: `--matrix` builds `-p engine-world-lab --example
vehicle_lab`, ships `tools/scenario-matrix` (matrix.py, scenes) and the generated
sedan (`CAR`, default `engine/out/gen/sedan_new (the generated sedan from the current code, AGENTS.md)`),
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

## The scenario matrix on CUDA

`tools/scenario-matrix/run.sh --cuda SWITCH [ROW ...]` (from `engine/`) runs every row in
one T4 container, in turn (`gpu_run.sh --matrix-cuda`; `MODAL_OPTS="--pack N"` spreads them
over N containers): engine-cuda's `explicit_replay`
replays the row's portable fixture (`engine_exec::explicit_plan::portable`: every binding
as the GPU held it, the layout, the planner's settings and the host's frame paths) with
the switch on and off, and matrix.py prints the usual table and verdict. No World, scene
or vehicle is built on Modal, so nothing compiles there; the device code is built locally
(the 33 modules the rows need, listed by `explicit_replay --modules`).

Once, after `run.sh --make` (on the Mac, under the GPU lock; reads each `ROW.fx` back as
`portable/ROW.pfx`): `python3 scripts/with-gpu-lock.py -- env RUBBLE_GPU_LOCK=0 python3
tools/scenario-matrix/matrix.py --make-portable`.

Bit for bit is exact within CUDA. A switch that runs other modules (the split paths)
needs `ENGINE_CUDA_FMAD=0` for that: by default nvcc fuses multiply-adds per module, so two
modules doing the same arithmetic can differ in the last bit. Measured 2026-11-19 (T4):
`perf=0` every row bit for bit, deltas within 2.4 %; `groupsplit` (strict) every row bit for
bit, the contact rows 3 to 20 times faster with it. A run: 2.5 to 4 minutes of local build
after a kernel edit (seconds after a Rust change or an edit undone; "Build speed"), then about half a minute on Modal and half a cent for the 12 rows.

What one verdict costs (2026-11-21, T4, `perf=0` strict, all 12 rows; each row's on and off
halves always run in the same container, so a verdict never reuses a stored control):

| Containers, cores, memory | Billed | Cost | Wall on Modal |
|---|---|---|---|
| 12 (one per row), 8 cores, 8 GiB (the old default) | 53-63 s | $0.015-0.018 | 65-495 s (twelve GPUs to find) |
| 12, 1 core, 2 GiB | 57 s | $0.0104 | 37 s |
| 1, 8 cores, 8 GiB | 24 s | $0.0069 | 27 s |
| **1, 1 core, 2 GiB (the default)** | **26 s** | **$0.0048** | **29 s** |
| 1, 0.5 core, 2 GiB | 25 s | $0.0044 | 28 s |
| L4: 1, 1 core, 2 GiB | 21 s | $0.0051 | 330 s (queued for an L4) |

- Twelve containers bill twelve starts (about 2.5 s each) and wait for twelve GPUs; one
  container bills one and waits for one. The rows' own work is about 21-25 s either way.
- The container only launches GPU work: 1 core and 2 GiB time the same as 8 and 8, at 63 % of
  the rate. Half a core saves another 8 %; one core is kept as the margin for host-heavy
  rows.
- One container also gave steadier numbers: no row needed its two extra runs, while the
  per-row layout reran two rows, and its truss row read 2.18 ms against 1.26 ms on a
  different container.
- The T4 stays the default: the L4 costs the same per verdict (faster, at 1.32 times the
  rate) and Modal often has none free. The L4 is for large scenes, where GPU time dominates.
- At $0.0048 a verdict, $30 a month is about 6,000 matrix verdicts (200 a day).

`PROFILE_SPLIT=1` adds each row's frame on CUDA (`explicit_replay --ab-profile`, both
halves): wall and GPU time, the kernels' sum, the gaps between launches (GPU minus
kernels), the host (wall minus GPU), and launches, bank writes and argument reads a frame,
then each kernel's time. Indirect dispatches read their workgroup counts on the GPU
(each module's `{entry}_gs`, engine-slang-build `grid_stride_entry`), so a frame has no
wait for the host; switch `devindirect` (or `RUBBLE_CUDA_HOST_INDIRECT=1`) reads them
back instead. Measured 2026-11-19 (T4, strict floats): every row bit for bit, 10 to 32 %
faster (pile32 1 %); the gaps went from 0.5 to 5.7 ms a frame to none, the host's share is
under 0.2 ms (pile32 0.5 ms), and what remains is the kernels' own time.

Tried and not kept (2026-11-19, same matrix): replaying each frame as a CUDA graph. Bit for
bit, but within noise: -2.6 to +1.3 % against the same launches queued one by one. A graph
cannot hold the banks' pointer rewrites as host writes, and the device copies that replace
them cost more than the small host copies the driver puts in the stream. With no gaps left
between launches, a graph has nothing to remove.

Where CUDA stands against the Mac (2026-11-19, `run.sh --cuda perf=0`, ms a frame; Mac =
`run.sh --baseline`, wgpu on an M3 Max, the same fixtures). One thing changed at a time:

| row | T4 strict | T4 default floats | T4 strict, L1 loads | L4 strict | Mac |
|---|---|---|---|---|---|
| drive | 3.90 | 3.71 | 3.65 | 3.28 | 3.93 |
| tumble | 4.35 | 4.31 | 3.79 | 3.76 | 3.79 |
| wall | 4.22 | 4.01 | 3.66 | 3.65 | 4.73 |
| pile2 | 11.79 | 11.87 | 9.90 | 9.58 | 8.44 |
| pile8 | 24.37 | 24.48 | 21.41 | 16.17 | 15.40 |
| pile32 | 75.39 | 74.95 | 65.92 | 42.68 | 38.03 |
| headon | 17.91 | 17.98 | 16.64 | 14.69 | 13.99 |
| side | 13.48 | 13.45 | 12.12 | 10.64 | 9.88 |
| truss | 1.68 | 2.34 | 1.27 | 1.33 | 1.53 |

- Strict floats (`ENGINE_CUDA_FMAD=0`) cost nothing measurable: the bit-for-bit test mode is
  also the timing mode.
- `-dlcm=cg` (plain loads cached in L2 only, the correctness rule in `fatbin_threads`) costs 3
  to 16 % a row; `ENGINE_CUDA_LOAD_CACHE=ca` is measurement only (a plain load may then read a
  value another block changed by atomics). Keeping the speed means making those reads atomic
  or coherent in the CUDA build, not flipping the flag.
- The rest of the T4's gap on contact rows is the T4: on an L4 the same build is 0.75 to 1.1
  times the Mac's time (pile32 1.12). Use the T4 for verdicts and ranking (cheapest, nearly
  always available); an L4 for absolute numbers (Modal had none for some rows at times:
  `NO RESULT (NO GPU)`).

## The World's own cost on CUDA (`run.sh --world --cuda`)

The World step around the explicit frame runs on CUDA too: `CudaWorld` (engine-cuda) owns
the rigid step (`CudaSim`), the explicit frame (`CudaExplicitBeams`) and the explicit
vehicles' coupling (`vehicle_coupling.slang`, built for CUDA) in one context on one
stream, and replays the same plans as the wgpu World
(`engine_exec::explicit_plan::coupling::world`). So the scenario matrix's `--world` gate
has a CUDA form, with nothing compiled on Modal:

```bash
# Once, on the Mac (under the GPU lock; about a minute): each --world row's scene run to
# its moment and saved as a portable World, out/scenario-matrix/world/ROW.wfx. Needs no
# saved .fx state. Again after a change of the buffers' layout or of the scenes.
tools/scenario-matrix/run.sh --make-world
# Any number of times, on Modal (a T4, one container, 1 core): engine-cuda's world_replay.
ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --world --cuda [ROW ...]
```

For each row it prints every step's GPU ms (three rounds of 20 steps), the median, p95
and worst step and which one it was, the split by part (emit, rigid step, plan, explicit
frame, loads), the gate's line (`World step`, `frame alone`, `the World adds`; wall
medians, as on the Mac), a hash of the end state and whether a second replay from the
file ends bit for bit; then one table and a verdict (PASS: every row ran, its status
words stayed zero, its repeat was bit for bit). `WORLD_MS=2.5` also fails a row whose
World adds more than that; unset, a T4's timing is reported, not judged. Measured
2026-11-22: 8 rows in 16.4 s of replay, 20.8 container seconds, $0.0038; the World adds
0.30 to 1.05 ms on a T4 (0.49 to 1.43 ms on the Mac the same day, wall). Numbers by row:
`engine/docs/world-on-cuda.md` §6, "C4".

What to know before reading a number from it:

- **A portable World holds the scene, not the solver's caches** (bodies, colliders, hull
  arena, joints, materials, parameters; the coupling's clock, pending emit and work
  buffer; the vehicles' whole explicit state; 30 steps of drive inputs). The replay
  starts its rigid contacts cold, so it runs 5 warm-up steps before it times
  (`--warm N`), and its bodies follow the saved World's only as closely as that allows.
  The vehicles' state is exact.
- **The plan is the scan on CUDA.** The candidates from the World broadphase's resting
  table (`cpl_query`) do not run there (the rigid host keeps no such table yet); the
  proxies are the same. The matrix's rows are planned by the scan on the Mac too.
- **The first replay of a container is not reported.** A T4 runs its first seconds of
  work slower (the first row read 14 to 25 ms a step in its first round against 10.5
  later, with bit-identical results), so `world_replay` replays the first file once
  before it times anything.
- **It reports one build.** There is no on/off switch and no two-build mode for the
  World step yet: to compare two commits, run it for each (both in the same half hour)
  and look at rows whose difference is well past the run-to-run spread, or use the
  explicit frame's A/B (`run.sh --cuda --base`) for what a change does to the frame.
- The correctness of the coupling on CUDA is held by `engine-cuda/tests/world.rs`
  (momentum closure through the World's impulse path, the CPU reference coupling, the
  wgpu World's numbers for one scene, two runs bit for bit):
  `tools/modal/gpu_run.sh -- -p engine-cuda --test world` with the modules
  `cargo test -p engine-cuda --release --test world world_test_modules -- --nocapture`
  prints (7.9 container seconds, $0.0023).

## The rest of the World on CUDA

Terrain, the stress solver and fracture are not on CUDA yet, so they still need the
Vulkan route and its cold compiles; so do the rigid step's riders the matrix does not use
(raycast wheels, characters, queries, the sleep archive). The plan, the inventory of
passes, the host work each needs and the cost per verdict: `engine/docs/world-on-cuda.md`.

## Two builds, head to head (commit against commit)

For performance experiments, prefer comparing two builds to adding a runtime switch.
A switch adds branches you have to remove later. A branch inside a hot kernel can change
register use and spill values, so you end up timing the branch. An architecture change
(a new buffer layout, a new schedule, rewritten kernels) cannot sit behind a switch at
all. Two builds time exactly what would ship.

From `engine/`:

```bash
ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda --base main --head my-experiment [ROW ...]
```

- Without `--head`, the head is the working tree's build.
- `ENGINE_CUDA_FMAD=0` (strict floats) keeps bit for bit meaningful between builds that
  launch different modules. It costs nothing measurable.

What happens:

1. **Building.** Each ref is built once in local Docker by `tools/modal/build_ref.sh`
   (`git archive` of the commit, the CUDA build image, device code for the modules the
   rows' fixtures need, as that commit names them). It is kept in a library under
   `engine/target/gpu-run-refs/<commit>-<settings>/` (gitignored), with an `info.txt`
   giving the commit, subject, branches, settings, modules, fixtures and build time.
   - `build_ref.sh --list` shows the library.
   - `--prune N` keeps the newest N builds.
   - The key covers the commit, every `ENGINE_*` setting (SASS 75 for the T4, 89 for the
     L4; strict floats or not) and the saved states (every row's, whichever rows a run
     names, so one build serves any selection). A commit built before costs no build.
   - A new commit takes about 8 minutes (491–512 s measured; the dependencies are shared
     between commits, and the workspace crates and nvcc modules rebuild).
   - The working tree is never kept, since it has no commit to name it by. Commit an
     experiment to keep it.
2. **Shipping.** Both binaries go up in one archive (`gpu_run_pack.py --exe`).
3. **Running.** One container. For each row, `matrix.py --two-builds` alternates a base
   process and a head process, A B, B A, A B (`PROCS=3`), each running
   `explicit_replay --run` on the row's portable fixture: `ROUNDS` x `FRAMES` frames, each
   frame's best round, mean. Its end state is saved word by word (`--save-state`).
4. **Judging.**
   - The median process of each build is compared.
   - `noise` is the spread between the base's own processes. Trust a delta well above it.
   - `bit4bit` compares the end states word for word, base against head, and each build
     against its own other processes.
   - A changed layout compares the bindings both builds have and names the rest.
   - The verdict fails a row that differs, or that is slower than 5 % and 0.05 ms.
   - `ALLOW_DIFF=1` reports differences without failing, for a change that reorders
     arithmetic on purpose. Equivalence then needs its own evidence (rung 3).

Both builds must read the same portable fixtures. A head that changes what a fixture
holds needs fixtures made from the head (`run.sh --make`, `matrix.py --make-portable`)
that the base can also read, or a fixture per build. A base older than
`explicit_replay --run` (before 58a0cfe1) is timed through its no-op A/B and has no end
state to compare (`bit4bit` reads `n/a`).

Measured 2026-11-21 (T4, strict floats, all 12 rows):

| Comparison | Result | Billed |
|---|---|---|
| 58a0cfe1 against itself (base from the library, head from the tree) | PASS: every row bit for bit, deltas −0.1 to +0.7 %, noise 0.0–0.5 % (drive 9.5 %: one base process ran at another level; the median ignores it) | 112 s, $0.020 |
| 58a0cfe1 against `claude/fake-ab-slow`: a demo commit that turns off GPU-side indirect counts (the host reads them back), lossless and slower | FAIL as it should: 11 rows slower by 13–42 %, pile32 +2.4 % (its long frame hides the waits), every row bit for bit, noise ≤ 0.5 % (drive 4.1 %) | 103 s, $0.019 |

A two-build verdict costs about four times a switch verdict (six processes a row, each
starting CUDA and loading its fixture). That is about 2 cents, so about 1,500 verdicts for
$30.

### Hardware counters (`COUNTERS=1`)

`COUNTERS=1` adds NVIDIA's counters for each kernel to every row, both builds:
time, registers, achieved occupancy, SM and DRAM throughput against peak, and L1 and L2
hit rates. Builds that differ show the base's value in brackets. Nsight Compute replays
every launch several times, so this runs after the timing, never inside it. It runs in
its own Modal image with `ncu` (`run_counters` in `gpu_run.py`, built once).
`COUNTER_METRICS` replaces the metric list; `COUNTER_TOP` sets how many kernels are shown
(default 12).

- Modal's sandbox (gVisor) refuses to lock the GPU clocks, so ncu runs with
  `--clock-control none`. Every counter works that way (probed 2026-11-21), but ratios
  can come out wrong (one probe showed 142 % achieved occupancy). Use the counters to
  explain where time goes; the timed A/B decides what is faster.
- Nsight Systems (`nsys`, the launch timeline) also works there. It is not wired in.
- Profiling is slow: about 6.5 minutes and 11 cents for one small row (truss, both
  builds, 8 cores; 388 s billed). Name one or two rows (`... --head X truss`).
  - `COUNTERS=1` gives the container 8 cores, 8 GiB and the 1800 s cap: Nsight works on
    the CPU, and on 1 core two rows passed the 600 s cap with the GPU at 0–1 %.
  - Nsight runs with `--replay-mode application` (the whole replay once per counter pass,
    valid because a replay is bit for bit deterministic) and `--cache-control none` (hit
    rates are the frame's, not a flushed cache's).
- Kernel names show which path ran. A kernel present in one build and absent in the
  other reads −100 % or `new`.
- Measured 2026-11-21, truss, base 58a0cfe1 against `claude/fake-ab-slow` (host readback
  of indirect counts):
  - The head was 36.8 % slower per frame, yet its kernels summed to less GPU time: 899 µs
    against 1,162 µs.
  - The head skips launches whose count is zero, and the base's grid-stride `self_pairs`
    and `self_walk` launch with no work, about 254 µs a frame (about 4 µs a launch).
  - So the head's cost is all host waits, and the base has an optimization lead.
  - That run's timing was noisier than usual: the base's truss processes spread 81 %,
    against 0.4 % on 1 core, and the cause is not established. Judge speed by a run
    without `COUNTERS`.

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
engine/tools/modal/gpu_run.sh --timeout 1500 -- -p engine-world-lab --example vehicle_lab -- \
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
  -- -p engine-world-lab --example vehicle_lab -- --mode gpu --measure mountain-perf --secs 4

# fleet.sh's run (explicit_vehicles_bench with the generated sedan), "apart" mode;
# "town": drop EV_NO_CONTACT / EV_NO_VV. EV_VV_GRID=0 etc. as --variant.
engine/tools/modal/gpu_run.sh --timing \
  --env EV_ETK=@sedan:sedan/sedan.pc --env EV_SPAN=1 --env EV_SECS=3 \
  --env EV_NO_CONTACT=1 --env EV_NO_VV=1 \
  -- -p engine-world-lab --example explicit_vehicles_bench -- 1 8 32
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
the end. With the default 8 cores and 8 GiB: T4 ≈ $0.00029/s (≈ $0.017/min); with 1 core and
2 GiB (the CUDA matrix's default) ≈ $0.00018/s (≈ $0.011/min); 16 cores
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

## Building on Modal (`remote_build.sh`; the CUDA matrix's default)

`tools/scenario-matrix/run.sh --cuda ...` builds on Modal's CPUs, so this machine compiles
nothing: it hashes its source files, sends the ones that changed, and reads the result.
Many sessions can build and test at once, each in containers of its own. The archive
never comes back here: the GPU stage reads it from the volume it was written to.
**Choosing where to compile**, for any `gpu_run.sh` or `run.sh --cuda` run:
`RUBBLE_BUILD=modal` or `RUBBLE_BUILD=local` (or `gpu_run.sh --build modal|local`;
`MODAL_OPTS="--build local"` through `run.sh`). Every run then says where it built and
how long that took (`gpu_run: built on Modal in 36 s` / `built locally in Docker in ...`),
so the two are easy to compare.

- **Default:** modal for the CUDA matrix, local for other selections.
- **The two-build A/B** (`--base/--head`) builds its commits here (`build_ref.sh`). An
  exported `RUBBLE_BUILD=modal` leaves it local with a note; an explicit `--build modal`
  is refused.
- `--fanout-tests` builds on Modal too (its test lists come from the remote build).
- `GPU_RUN_LOCAL_BUILD=1` and `--remote-build` are kept as aliases.
- `remote_build.sh -- <selection>` only builds.

```bash
RUBBLE_BUILD=local ENGINE_CUDA_FMAD=0 tools/scenario-matrix/run.sh --cuda perf=0
```

How it works (`remote_build.py`, a deployed Modal class, redeployed when its code or the
build image changes):

- **Source.** Each worktree is a tree on Modal, keyed by its path. The Mac sends only the
  files whose content changed since that tree's last build, inside the call (a typical
  edit: one file, 40 KB, under a second). The container keeps the tree on its local disk;
  a cold container restores it from one zstd snapshot.
  - Not sent: 3D models and Blender files, and anything over 8 MB; no build reads them.
  - `engine/Cargo.lock` is gitignored in this repo, so the worktree's own lockfile is sent,
    and the remote build resolves the same crate versions.
- **Cargo works on local disk.** With its target directory on a Modal Volume, every
  fingerprint check was a network round trip: nothing changed took 59 s, a Rust edit
  120 s. Now the target comes in as one zstd snapshot (about 570 MB, 4–15 s) when a
  container starts cold. It is written back when the container stops, not after each
  build, so the next build on a warm container restores nothing.
- **Warm containers.** A container stays up 60 s after a build (at most $0.015 of idle 16
  cores), so back-to-back builds pay no start-up and no restore.
- **Kernel caches.** nvcc's fatbins and slangc's outputs are kept on a Volume by content
  (`rubble-rb-kernels`), shared by every tree and session. An edit recompiles only the
  modules whose CUDA source changed; an edit undone compiles nothing.
- **The matrix's module list** (`explicit_replay --modules`) runs on Modal too, in a
  target directory of its own. The rows' fixtures are uploaded to the states volume by
  content, once, and the GPU stage finds them there.

Measured 2026-11-23 (16 Modal cores; `explicit_replay` with the matrix's explicit
modules, SASS for the T4; end to end from the Mac):

| Build | Remote, warm container | Cost | Mac, local Docker (main's figures) |
|---|---|---|---|
| Nothing changed | 12 s | $0.0024 | 1 s |
| A Rust edit in engine-cuda | 22 s | $0.0048 | 4 s |
| The edit undone | 21 s | $0.0043 | 5 s |
| A line added to `explicit_beams_fleet_levels.slang` (4 modules recompile) | 197 s | $0.048 | 142 s |
| That kernel edit undone (from the kernel cache) | 24 s | $0.0046 | 5 s |
| First build from empty caches | 446 s | $0.11 | 324 s |

- **A cold container adds 15–30 s:** restoring snapshots, plus Modal's start-up (up to
  about 30 s after a redeploy).
- **End to end (matrix, two rows):** build 109 s on a cold container after new fixtures,
  then the GPU stage 8 s and $0.0008. Verdict PASS, bit for bit.
- **Kernel edits are bounded by one module's nvcc on one core.** Modal's cores are about
  25 % slower than the M3 Max for it (see Build speed). Spreading modules over more
  containers would not shorten the slowest one. What would shorten it is a kernel change:
  stable status codes instead of `__LINE__`, or smaller modules.
- **Cost model.** About $0.0003 a second for 16 cores and 16 GiB: a Rust-edit verdict is
  well under a cent, build and GPU together.

**After #158 (Slang sources split into parts), 2026-11-24, same setup:**

| Build | Remote, warm container | Cost |
|---|---|---|
| Nothing changed | 14 s | $0.0028 |
| A Rust edit in engine-cuda, or that edit undone | 23 s | $0.0050 |
| A line added in `explicit_beams_vv_levels.slang` (one part of `explicit_beams.slang`): 3 modules recompile | 178 s | $0.042 |
| That kernel edit undone | 23 s | $0.0049 |

- **An edit to one part now recompiles only the modules that use it.** The line added in
  `vv_levels` made nvcc compile 3 of the 34 explicit modules:
  - `split_group_wide`: 137–216 s, depending on which Modal host ran it;
  - `contact_wide_nosurf`: 77–101 s;
  - `split_grouplevels_wide`: 38–52 s.

  Before the split, a line shift in `explicit_beams.slang` changed most modules' CUDA,
  because every later line moved and the loop-cap words carry `__LINE__`. Now a module
  that doesn't use the edited part produces the same CUDA, and its fatbin comes from the
  cache.
- **An edit to a part every module uses** (`explicit_beams_beam_phase.slang`) took
  307–322 s, before the module-list fix below. It reaches the largest module.
- **Each build reports what compiled:**
  `[remote_build] nvcc ran 3 time(s), slangc 148: explicit_beams_split_group_wide.cu 137 s (done at 156 s), ...`.
  A sampler in the builder reads `/proc` twice a second, so the build and its cache keys
  are untouched; a wrapper at `NVCC` would change the fatbin key, which covers the
  compiler's own bytes.
- **The module list is kept between builds.** `explicit_replay --modules` (33–38 s
  after any kernel edit, because its build reruns slangc for every module) is kept in
  `rubble-rb-kernels/module-lists/`. Its key is the fixtures, the Rust sources and
  manifests, and the ENGINE_* settings. Kernel source never changes the list, so a
  kernel-only edit skips that step. The client reads the entry and passes it in, because
  a container's view of a Volume didn't show another container's entry even after
  `reload()`.
- **What bounds a kernel edit now is the largest module the edit reaches,** compiled on
  one core: `split_group_wide` at 137–216 s, depending on the Modal host (the Mac's
  figure for it is 136–160 s). Shortening it is a kernel-side change: split that module.

## Build speed (CUDA route)

Measured 2026-11-21 on the M3 Max (16 cores, OrbStack), `engine-cuda`'s `explicit_replay`
in release with the matrix's 33 explicit modules and SASS for the T4 (what
`run.sh --cuda` builds). Every number is the build alone, nvcc and slangc timed per call.

| Step | Before (x86_64 image, Rosetta, no caches) | Now (arm64 host, x86_64 nvcc, caches) |
|---|---|---|
| First build (empty target and caches) | 570 s | 324 s |
| Nothing changed | 1 s | 1 s |
| A Rust change in engine-cuda | 19 s | **4 s** |
| A line added to `explicit_beams.slang` | 327 s (305 slangc, 237 nvcc) | 246 s (101 slangc, 34 nvcc) |
| A line added to `explicit_beams_fleet_levels.slang` | 325 s (305 slangc, 237 nvcc) | 142 s (24 slangc, 4 nvcc) |
| The edit undone (or any state built before) | 327 s | **5 s** (all from the caches) |

What changed (`gpu_run.sh`, `build_ref.sh`, `build/Dockerfile`, engine-slang-build):

- **Kernels kept by content, shared by every worktree and commit.** nvcc's fatbins
  (`ENGINE_CUDA_FATBIN_CACHE`, which only CI used before) and slangc's CUDA sources and
  reflection (`RUBBLE_SLANG_OBJ_CACHE`; the CUDA path now uses the same `compile_key` as
  the other targets) live in the Docker volume `rubble-cuda-kernels`. Before, any change
  under `slang/` ran slangc 305 times and nvcc 237 times (126 rigid, 77 terrain, 34
  explicit), though an explicit-beams edit leaves the rigid and terrain kernels alone.
  A fatbin is reused only when the CUDA source, nvcc's version and arguments and the
  compilers' own bytes match (checked byte for byte on a hit). `fatbin_cache.py` keeps
  the volume to 12 GiB (fatbins) and 4 GiB (slangc), dropping entries unused for a week.
- **Rust and slangc native, nvcc the x86_64 one.** The CUDA route builds in the arm64
  image (cross-compiling the host to x86_64, as the other routes already did); nvcc is
  the x86_64 image's own toolkit run through Rosetta, with the cross toolchain's g++ as
  its host compiler. `GPU_RUN_X86=1` builds in the old image.
- **`build_ref.sh`**: one tree for every commit, updated by content (`rsync -c`, so
  unchanged files keep their times and cargo rebuilds only the crates that changed), and
  the same kernel caches. Before, each commit rebuilt every workspace crate and all its
  kernels (491-576 s). Now (the matrix's twelve fixtures): a first commit 349 s, then a
  commit that changed only a doc 11 s, one that added a line to
  `explicit_beams_fleet_levels.slang` 149 s.

Device code is unchanged. All 237 modules (rigid, terrain, explicit) were built both ways
from one tree and compared with `cuobjdump -sass` and `-ptx`: identical, apart from the
translation unit's name hash (`_INTERNAL_xxxxxxxx_`, nvcc hashes the source's path, which
differs between the two targets' build directories) and the debug sections that name
that path. At the same path the fatbins are byte-identical. End to end,
`run.sh --cuda perf=0` (strict floats) built this way passed on a T4: every row bit for
bit, $0.0047.

Findings that decided this:

- **NVIDIA's aarch64 nvcc is not the x86_64 one.** Same release (12.8.93), same
  `--version`, but it writes different SASS for 15 of 237 modules (`SGXT` sign extension
  where x86_64 has `PRMT`, call offsets moved): on aarch64 `char` is unsigned. It would
  have been about 3x faster on small modules. Not used; the fatbin cache's key now
  includes the compilers' bytes so the two can never be mixed.
- **Rosetta is not what makes nvcc slow.** The largest module (`explicit_beams`, 866 KB
  of CUDA) took 140 s with the native aarch64 nvcc, 147 s with x86_64 under Rosetta and
  187 s on a Modal x86_64 CPU container. Rosetta costs about 2.5x on short processes
  (a small module 0.9 s vs 2.4 s; slangc 1.0 s vs 1.7 s), little on long ones.
- **Building on Modal CPU containers does not help.** Modal's cores were slower than the
  Mac's for nvcc (187 s vs 147 s), the wall time is bounded by the slowest single module
  either way, and a cold kernel build would cost about $0.03 plus upload. Not built.
- **What is left after a kernel edit is one module's nvcc.** nvcc compiles a module on
  one core: `explicit_beams_split_member_wide` takes 207-258 s, `split_group_wide` 136-160
  s, and the edit waits for the slowest module it reaches. An edit to `explicit_beams.slang`
  reaches every explicit module: everything after the edit moves down a line, and the
  kernels' loop-cap status words carry `__LINE__` (`SEL R0, RZ, 0x1001319` became
  `0x100131a` after one blank line), so even a line shift changes the SASS of 24 of 34
  modules. Keying the cache on the source without `#line` would therefore be lossy, and
  is not done.
- Ways to go further (each changes kernels or their output, so not done here): stable
  status codes instead of `__LINE__`, so an edit leaves the modules it does not touch
  unchanged; smaller modules, so the slowest compile is shorter.

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
