# The World on CUDA (Modal): plan

Status: **plan, 2026-11-21. Nothing here is built yet** unless a line says "exists".
Goal: every GPU pass of a World step runs on CUDA on Modal the way the explicit soft-body
frame already does (`tools/scenario-matrix/run.sh --cuda`): kernels built locally by nvcc,
Modal only runs the binary, a verdict costs cents and needs no cold driver compile. Today
anything outside the explicit frame needs the Vulkan route, whose new kernel versions cost
18–28 minutes and about $0.45 of driver compile on Modal (`tools/modal/README.md`, "Cold
kernel compiles").

Rules this plan keeps (AGENTS.md, memory):

- **One source.** Slang kernels plus one backend-neutral Rust schedule per stage
  (`engine_exec::rigid_plan`, `engine_exec::explicit_plan::schedule`, and new ones below).
  Each backend host (wgpu, Metal, CUDA) only allocates, binds and replays. No hand-written
  CUDA, no dispatch order duplicated in a host.
- **Lossless.** Every CUDA port is checked bit for bit against itself with a switch off
  (`ENGINE_CUDA_FMAD=0` when the two sides run different modules) and against the CPU
  reference / the wgpu host within the existing suites' tolerances. No pass is skipped,
  coarsened or reordered to make it fit CUDA.
- **Forward-only destruction.** Nothing here adds a rewind; `GpuWorld::correct_frame`
  stays a measurement behind `STRESS_CORRECTION_LIMIT=1` and is not ported.

## 1. Where it stands

Correction to the starting assumption ("engine-cuda only hosts the explicit frame"):
**the rigid step already runs on CUDA.** `CudaSim` (`crates/engine-cuda/src/lib.rs`, since
2026-10-17, DECISIONS 8ez) replays `rigid_plan::encode_step` with the same buffers, host
words and commands as `WgpuSim` and MetalSim. Its gates (`tests/cudasim.rs`: the columns
at 640, 10k with and without sleep, 100k; pipelined against waiting; colour-list
coverage) pass on a T4, and 10,241 awake bodies step in 1.88 ms on a T4 (0.66 ms on an
L40S). What CUDA lacks is the World **around** the rigid step, and the device-side
indirect dispatch the explicit frame got in Glavin001/rubble#142.

| Stage of a World step (`World::step`, `crates/engine-world/src/lib.rs:2215`) | Slang | Built for CUDA today | Host on CUDA today | Schedule |
|---|---|---|---|---|
| Rigid step: broadphase, narrowphase, convex SAT, colouring, AVBD solve, sleep/islands, CCD, archive, contact events | `world_step`, `convex_step`, `convex_sat_body`, `ccd_step`, `compact_solve`, `persistent_solve`, `reuse_copy` | **Yes**: the 126 `PERSISTENT_ENTRIES` (engine-cuda `build.rs`, `CUDA_KERNELS`) | **Yes**: `CudaSim` | shared: `rigid_plan::encode_step` |
| Rigid step's rider entries: characters / queries / projectiles, vehicle controllers, guidance, `pose_pack` | `character_step`, `vehicle_step`, `guide_step`, `pose_pack` | The ones listed in `PERSISTENT_ENTRIES` (e.g. `char_move`, `vehicle_post`); `pose_pack` not | Partly: whatever `encode_step` emits for the modes `CudaSim` sets; no character / query / vehicle input or readback API | shared (`encode_step`) for the dispatches; the inputs and readbacks are wgpu-host code |
| Explicit soft-body frame (node–beam vehicles, contact, surface) | `explicit_beams*.slang` | **Yes** (`CUDA_EXPLICIT_KERNELS`, `ENGINE_CUDA_EXPLICIT_MODULES`) | **Yes**: `CudaExplicitBeams`, device-side indirect (`{entry}_gs`) | shared: `explicit_plan::schedule` |
| Soft↔rigid coupling, explicit side (proxy records in the explicit buffers, window impulses) | `explicit_beams_rigid.slang` (inside the explicit modules) | Yes (part of the explicit modules) | Records only from the CPU (`set_proxies`, `decode_proxies`) | shared (`explicit_plan::coupling` layout) |
| Soft↔rigid coupling, World side: gather proxies from the rigid bodies, apply the window's impulses to them, candidate list / fold, status | `vehicle_coupling.slang` (9 entries) | **No** | **No** | wgpu-host code (engine-world `explicit_vehicles.rs`; `explicit_before_step` / `explicit_after_step`) |
| Legacy soft bodies (cloth / jelly, not the explicit vehicles) and skinning | `soft_step` (`SOFT_ENTRIES`), `soft_skin` | No | No | wgpu-host code |
| Deformable terrain (MPM) | `terrain_mpm`, `terrain_mpm_fast`, `terrain_mpm_cuda` (16 `_gs` entries), `terrain_surface`, `terrain_watch` | **Yes** for the plan's 32 kernels, both fast paths and the `_gs` entries (`CUDA_TERRAIN_KERNELS`); not `terrain_surface` / `terrain_watch` | **No** (never run on a GPU; executor designed in `docs/deformable-terrain.md`, "CUDA plan") | shared: `engine_terrain::plan` |
| Stress solver, fracture, ingest, routing, GPU split, GPU punch, threshold predictor (the `blast_parity` profile and the World's `encode_fracture*`) | `stress-oracle/crates/stress-gpu/shaders/*.slang` (43 modules; `fracture.slang` 4.7k lines, `world.slang` 3.5k) | **No**: compiled to Metal and WGSL only, by `stress-gpu/scripts/compile_shaders.sh`, outside `engine-slang-build` | **No** | **Hand-recorded wgpu** (`solver.rs`, `ingest.rs`, `resident.rs`, `routing.rs`, `gpu_pass.rs`, `gpu_split.rs`; 16 indirect dispatch sites, ~90 read sites). `docs/stress-5d-one-command-buffer.md` requirement 4 (a backend-neutral plan) is "not met" |
| World fracture's own narrowphase for chunk contacts | `fracture_step.slang` (7 entries) | No | No | wgpu-host code |
| Learned impact (opt-in) | `learned_impact.slang` (14 entries) | No | No | wgpu-host code |
| Host plumbing every step: host uniforms, timer, readback ring (`wait_oldest`), loop-cap and coupling status | — | — | `CudaSim::drain` reads counters; no ring | engine-world `lib.rs` / `readback.rs` |

## 2. What each stage needs from a CUDA host

The rules the two CUDA hosts already follow, which every new stage inherits:

- **Indirect dispatch → `{entry}_gs`.** CUDA has no indirect launch. Each indirectly
  dispatched entry gets a second entry from `engine_slang_build::cuda_layout::grid_stride_entry`
  that walks the group count the GPU wrote, launched over a fixed grid without a host
  wait. Bit for bit the indirect dispatch (each index runs the same code once); proven on
  the explicit frame (every matrix row bit for bit, 10–32 % faster, no gaps) and built
  (not run) for terrain. Entries with workgroup barriers loop at workgroup granularity
  (every thread of a block takes the same trip count), as terrain's `mpm_p2g` does.
- **Readbacks → one status read a frame.** A pinned host buffer filled by
  `memcpy_dtoh_async` behind an event, read one frame late like the wgpu readback ring;
  no `stream::synchronize` in a step. Anything the host must know before encoding the
  next step (counts that size dispatches) moves to the device (`_gs`) instead.
- **Atomics and the L2-only rule.** Slang lowers `Atomic<T>.load()/.store()` to plain
  global accesses on CUDA; kernels are built with `-Xptxas -dlcm=cg` (plain loads cached
  in L2 only) so a block never reads a stale L1 line of a counter or hash slot another
  block changed by atomics in the same launch. Every new module is built the same way
  (`cuda_layout::fatbin_threads`). The cost is 3–16 % a row on the explicit frame and
  ~12 % on the rigid step; the lossless way back (atomic or `ld.cg` loads for `Atomic<T>`
  only) is an open follow-up that would help every stage at once, not part of this plan.
- **Deferred-load hazard check.** Every emitted `.cu` passes
  `cuda_layout::deferred_load_hazards` (Slang's CUDA backend can load an element after a
  store to it; it broke `prefix_color_tiles_parallel`). Stress shaders have never been
  through it.
- **Barriers in uniform control flow.** `__syncthreads` in divergent code is undefined on
  CUDA as in Chrome. The Chrome rule (`gsUniformLanes`) already makes the engine kernels
  safe; stress shaders run on Chrome WGSL too, so they should be, but `synccheck`
  (`run_cuda_tests.sh --sanitize synccheck`) is the check for each new module.
- **Bindings.** CUDA has no binding limit: each module's `SLANG_globalParams` bank is
  filled from reflection (`cuda_layout::global_fields`) with device pointers, and a
  host rewrites only the per-pass pointer between launches. A pass that binds buffers of
  two stages (the coupling binds rigid bodies and explicit records) is just a bank with
  pointers from both, provided both stages' buffers live in one CUDA context.
- **Strict floats.** `ENGINE_CUDA_FMAD=0` for any A/B whose halves run different modules.

Per stage:

| Stage | Indirect | Readbacks the host waits on today | Atomics / other | New host work |
|---|---|---|---|---|
| Rigid step (`CudaSim`) | ~57 entries dispatched indirectly (`Cmd::Indirect`); host reads binding 23 back after one of the 5 kernels that write it: **7.1 stream syncs a step** on the 10k columns | the counters, every step (`drain`) | contact hash, colour lists, archive, warm hash: atomic counters and CAS; already under the L2 rule | `_gs` for the indirect entries (as #142 did for the explicit modules); counters into a one-frame-late pinned read |
| Coupling, World side | to be counted: candidate list and fold sizes come from the explicit frame's counts | coupling status (read with the step, no wait on wgpu) | impulse accumulation into rigid bodies: integer (fixed-point) atomics like the explicit accumulators; float atomics must not be introduced | move its dispatch order from `explicit_vehicles.rs` into a shared plan (`explicit_plan::coupling` gains `Cmd`s); one CUDA context shared by `CudaSim` and `CudaExplicitBeams` |
| Terrain | 16 `_gs` entries built | `RegionClock` substep count (one frame late), counters | P2G sums ordered; built under the L2 rule; no hazards found | the executor in `docs/deformable-terrain.md` "CUDA plan" step 4: bindings 37–54 in `CudaSim`, `TerrainPlan` lowered with `_gs`, counters to pinned memory |
| Stress / fracture | 16 indirect sites | many: `docs/stress-5d-one-command-buffer.md` §3 lists the resident path's extra submissions and reads still to remove | `world.slang`, `fracture.slang`, `coupling.slang` use atomics (CAS and counters); the barrage has a latent GPU race (charter log) that CUDA will expose, not hide | (a) build the 43 modules through `engine-slang-build` for CUDA; (b) the stress stage as a backend-neutral command list (5d requirement 4, needed for Metal and the web anyway); (c) a `RigidEngine` over `CudaSim` (stress-rubble drives `WgpuSim` today); (d) the one status read |

## 3. Order

Each step ends in a Modal verdict on a T4 that costs cents, before the next starts. The
first two steps are the rigid step the `run.sh --world` gate times and the 10k scale
tests; nothing else depends on later steps.

**C0. Re-baseline what exists (no code).** `engine/tools/modal/run_cuda_tests.sh` and
`--perf` on the current branch: the columns gates at 640 / 10k / 100k still pass, and
today's 10k awake ms/step and syncs/step on a T4. The October number (1.88 ms) predates a
month of rigid changes. If a gate fails, fix CudaSim first.

**C1. Rigid step without host waits.** The rigid plan's indirect entries get `_gs`
entries (engine-cuda `build.rs`, the same `grid_stride_entry` as the explicit modules),
and `CudaSim` launches them over a fixed grid with a pointer to binding 23's slot, as
`CudaExplicitBeams` does; `RUBBLE_CUDA_HOST_INDIRECT=1` keeps the readback path for the
A/B. Counters move to a one-frame-late pinned read. Verdict: every cudasim gate passes,
and the 10k / 100k columns run bit for bit against the readback path (`download_bodies`
after N steps, every word), timed both ways. Expected: the 7.1 syncs a step go to 0; on
the explicit frame the same change was 10–32 % a frame.

**C2. The 10k scale suite on CUDA.** `CudaSim` already has the columns (640, 10k, 100k).
Add the rest of the wgpu host's rigid scale tests: `engine-webgpu/tests/columns_hold.rs`
(`columns_25k_holds_height`), `ring_hold.rs`
(`ring_110k_stands_away_from_impact_and_ball_rolls_through`, the brick ring of the
drive-through goal) and `ccd_scale.rs` (100 and 500 CCD balls through brick walls).
Scenes that are built in Rust code move to a shared scene function both hosts call;
scenes from assets go through portable fixtures (bodies and colliders in the
`rigid_plan` layout, written once on the Mac), so nothing on Modal loads an asset. Verdict: the same assertions as on wgpu (heights held, colour
coverage, no overflow), plus ms/step at 10k and 100k. This is the CUDA answer to "10,000+
active rigid bodies" for every later perf change.

**C3. `run.sh --world --cuda`: the rigid step of the World gate.** A portable World
fixture per matrix row (`matrix.py --make-portable` extended): the rigid buffers and host
words at the row's moment, beside the explicit fixture it already writes. A
`world_replay` example runs N World steps as **rigid step + explicit frame**, each
replayed by its CUDA host in one context and stream, and prints the gate's line (`World
step`, `frame alone`, `the World adds`). With the coupling still missing (C4), rows with
rigid contact (wall, bricks, rocks, barrier) are marked "coupling not on CUDA" rather
than passed. Verdict: the gate's table for the free-standing rows (drive, tumble, pile*,
headon, side, truss), wall clock.

**C4. The coupling.** Move the World side's dispatch order (gather proxies, candidate
list, fold, apply window impulses, status) from engine-world into
`engine_exec::explicit_plan::coupling` as `Cmd`s, so wgpu follows it first (bit for bit
against today on the Mac: `vehicle_lab --ab-switch coupling` and the matrix) and then
CUDA replays it. Build `vehicle_coupling.slang` for CUDA. Verdict: every matrix row in
`run.sh --world --cuda`, and `explicit_ab` gets the `coupling_list` / `coupling_fold`
switches it reports as unavailable today. After C4 the whole `run.sh --world` gate
runs on CUDA.

**C5. Terrain.** The executor in `docs/deformable-terrain.md` (kernels and `_gs` entries
exist). Verdict: `terrain_parity` scenarios against the wgpu host's tolerances.

**C6. Stress and fracture.** The largest step and the one with prerequisites outside
CUDA:

1. Build the 43 stress modules through `engine-slang-build` (the CUDA target beside
   Metal and WGSL), with the hazard check and the L2 rule; CPU-only, any machine.
2. The stress stage as a backend-neutral command list (5d requirement 4). This is shared
   work: Metal and the web need it too, and wgpu follows it first, bit for bit against
   the recorded digests (`destruction_digests.json`, the quick tier) before any CUDA run.
3. `RigidEngine` over `CudaSim` (what `stress-rubble` implements over `WgpuSim`).
4. A CUDA executor for the stress plan; one status read a tick.
Verdict: the `blast_parity` quick tier's digests on CUDA, and the punch / routing suites.

**C7. The rest, as demand appears:** characters, queries and projectiles (their inputs
and the ray results' ring), legacy soft bodies, skinning, learned impact.

## 4. Cost per verdict (Modal T4)

Prices as in `tools/modal/README.md` ("Cost and turnaround"): a T4 container with 8 cores
and 8 GiB is about $0.00029 a second ($0.017 a minute). The build is local and free;
Modal bills container seconds only. Measured numbers are marked; the rest are estimates
from the closest measured run, to be replaced by the first real run of each.

| Verdict | Containers | Billed (container s) | Cost | Local build after a kernel edit |
|---|---|---|---|---|
| cudasim correctness gates (640 / 10k / 100k columns, coverage) | 1 | ~6 (measured, 2026-10) | ~$0.002 | rigid kernels only: ~70 s when all recompile; the fatbin cache skips unchanged ones |
| C1: 10k + 100k columns, `_gs` against readback, timed | 1 | 20–40 (est.) | ~$0.01 | as above |
| C2: 10k scale suite from portable fixtures | 1–4 (fan-out) | 30–90 (est.) | $0.01–0.03 | as above |
| C3/C4: `run.sh --world --cuda`, 12 rows | 12 | 60–150 (est., from the matrix) | $0.02–0.05 | rigid + explicit + coupling modules the rows load: ~4 min (the matrix's figure) |
| Explicit matrix `run.sh --cuda SWITCH`, 12 rows (exists) | 12 | 50–120 (measured, 2026-11-19) | $0.02–0.04 | ~4 min |
| C5: `terrain_parity` | 1 | 10–30 (est.) | <$0.01 | terrain modules |
| C6: `blast_parity` quick tier digests | 1–N | 60–300 (est.; the tier is minutes on the Mac) | $0.02–0.09 | 43 stress modules: unmeasured; run `tools/cuda/measure_nvcc.py` on them first |
| For comparison: any new kernel version on the Vulkan route | 1 warm-up + runs | 1,100–1,800 + runs (measured) | ~$0.45 per kernel version, up to ~$2 when the warm-up hits the cap | none (the driver compiles on Modal) |

Every CUDA verdict above stays under $0.10, and most under $0.05: one Vulkan cold compile
pays for about ten to twenty of them. Use the T4 for verdicts and ranking (cheapest,
nearly always available); an L4 for absolute numbers.

## 5. Open questions to settle while building (not blockers)

- C3's one-context composition: `CudaSim` and `CudaExplicitBeams` each create a context
  and stream today. Sharing one (or importing the other's pointers) is a host change of
  a few lines but decides whether the coupling's banks can point into both.
- Whether the wgpu host's World step encodes anything around `encode_step` that the
  shared plan does not (pose copies, timer marks, the terrain's and fracture's passes
  between engine substeps). Those belong in a World-level schedule, not in each host.
- The Mac stays the reference for final timings; CUDA numbers rank changes and catch
  regressions, as the README's "What compares with the Mac" says.
