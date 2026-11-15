# GPU runs on Modal

Run Rubble's wgpu tests and examples on a cloud NVIDIA GPU instead of this Mac's one
GPU, so several agents can run correctness checks and A/Bs at once.

```bash
engine/tools/modal/gpu_run.sh [options] -- <cargo selection> [-- <run args>]
```

**Read "Cold kernel compiles" below before you plan a run**: the current
explicit-beams kernels take minutes each to compile on NVIDIA's Vulkan driver, once
per kernel version.

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
| `--build-only` | Build and pack, no Modal call |
| `--keep` | Keep `engine/target/gpu-run/<id>/` |

The selection is what you would give `cargo test` / `cargo build` from `engine/`:
`-p CRATE --test NAME` (several `--test` are fine), `--example NAME`, `--bin NAME`,
`--features ...`. Always release. Arguments after the inner `--` go to every
executable (libtest's filter and flags for tests; the example's own arguments). Tests
run in their crate's directory, as `cargo test` does, one test thread unless you pass
`--test-threads`; examples and bins run in `engine/`. The container sets
`RUBBLE_GPU_LOCK=0` (one job per container), `WGPU_BACKEND=vulkan` and
`SEDAN=/scenarios/sedan_new`.

## Cold kernel compiles (read this first)

wgpu creates each compute pipeline from SPIR-V, and NVIDIA's driver compiles it on the
CPU, single-threaded, when the pipeline is created. For most of Rubble's kernels that
is about a second. For the **current explicit-beams kernels it is minutes**:
`gpu_adapter --compile explicit_beams_solo` (692 kB of WGSL, one entry point) took
**508-514 s on a T4 host** (driver 620.70.06); the frozen reference kernel
`explicit_ref_solo` (96 kB) takes 2.5 s. Naga's own WGSL→SPIR-V for the same module is
0.1 s, so it is the driver's compiler. A test or example that builds an
`ExplicitBeamsGpu` compiles one or more of these modules (frame or solo, contact,
contact-big, the `_det` / `_probe` / baseline / reference variants in the bitwise
tests, the `_nocpl`/`_nosurf` twins), so a cold run spends most of its time with the
GPU idle (`--monitor` shows `gpu 0 %`, one core at ~95 %).

The runner keeps NVIDIA's shader disk cache between runs in a Modal volume
(`rubble-gpu-run-shader-cache`): from the cache the same pipeline is created in 0.09 s.
The driver keys entries on the SPIR-V, so:

- an **unchanged kernel** (a rerun, an A/B of runtime switches such as
  `RUBBLE_COUPLING_LIST`, `--ab-switch`, `EV_VV_GRID`, a host-side change) is warm
  after the first run that compiled it;
- a **kernel you edited** compiles cold once (minutes per module that changed), then
  is warm; budget `--timeout` for it (up to 1800). A run killed by the cap still saves
  what it finished compiling, so the next run continues from there;
- Modal's hosts run different driver versions (610.57.04 and 620.70.06 seen); each
  version has its own cache, so a warm kernel can be cold again on another host.

Sharing the cache between worktrees is safe: a binary is found only for the identical
SPIR-V on the same driver. Each run copies the newest snapshot in and, if it compiled
anything, writes a new snapshot directory (never in place); the 4 newest per GPU and
driver are kept. The driver keeps its entries in one file pair, so snapshots cannot be
merged: when two runs on the same driver both compile, the one that finishes last wins
and the other's new kernels compile again on a later run. `--no-shader-cache` opts out; `modal volume delete
rubble-gpu-run-shader-cache` (via `engine/tools/modal/modal.sh`) resets it.

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

Cold, these suites need more than the 30-minute cap; run one test binary (or one
filter) at a time until the cache is warm for your kernels. Tests that only use small
kernels are quick, e.g. `--test explicit_loop_cap -- a_runaway` (5.6 s in the
container).

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

## Cost and turnaround

Modal bills per second while the container runs; `gpu_run.py` prints an estimate at
the end. With the default 4 cores and 8 GiB: T4 ≈ $0.000234/s (≈ $0.014/min), L4 ≈
$0.000292/s, L40S ≈ $0.000612/s (GPU prices of 2026-10 from modal.com/pricing, plus
CPU $0.0000131/core/s and memory $0.00000222/GiB/s). The 10-minute cap bounds a T4 run
at about $0.14; the 30-minute maximum at about $0.42.

Measured on 2026-11-14 (M3 Max host, OrbStack Docker with Rosetta, Modal T4):

| Step | Time |
|---|---|
| First build in a worktree (empty target volume, engine-webgpu test) | 9 min 10 s |
| Rebuild, nothing changed / a small Rust change | 3 s / 20-26 s |
| Modal: upload ~20 MB, container start, adapter check | 6-10 s (the image builds once, ~50 s, then Modal caches it) |
| `--test explicit_loop_cap -- a_runaway` end to end | 34 s (container 5.6 s, $0.0013) |
| A pipeline of a current explicit-beams module, cold / from the cache | 508-514 s / 0.09 s |
| `explicit_beams_bitwise sedan_alone`, cold cache | not done at 1455 s (two big modules compiled, 6.6 MB cached); $0.34 |

A typical warm correctness run (unchanged kernels) costs cents; a run that compiles
edited explicit kernels costs $0.10-0.40 per attempt, mostly an idle GPU.

## What does not work

- macOS-only code: `engine-metal`, `metal_interop`, anything that wants Metal.
- The browser, Chrome WebGPU, the web demos; Node tests (`*.test.mjs`).
- Shell wrappers (`fleet.sh`, `with-gpu-lock.py`): pass the cargo command they run
  (see above). `perl -e 'alarm ...'` is not needed: `--timeout` is the kill timer.
- Files outside the repo other than `@sedan`: put them in the repo (a gitignored
  `out/` works) and pass `--data`. `RUBBLE_INSPIRATION` (Rigs of Rods content) is not
  shipped, so tests that need it skip as they do without it locally.
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
