# The World on CUDA (Modal): plan

Status (2026-11-28): **C0 to C4 done and measured; C6 steps 1 and 2 done** (§6: the
fracture world's kernels are built for CUDA by a crate, and its frame is a command list
the wgpu host replays; no CUDA host runs it yet). The World
step of the `run.sh --world` gate (the coupling's emit, the rigid step, the plan, the
explicit frame, the loads) runs on CUDA in one context on one stream: `CudaWorld`,
`run.sh --world --cuda`. Next: C6 steps 3 to 5 (the destruction plan below).
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

`World::step` records one encoder and one submission (`lib.rs:2472`; a second only to
resolve timestamps when timing is on). Its GPU work, in order:

| Stage of a World step (`World::step`, `crates/engine-world/src/lib.rs`) | Slang | Built for CUDA today | Host on CUDA today | Schedule |
|---|---|---|---|---|
| Terrain MPM coupled substeps (first: the rigid step applies their reactions) | `terrain_mpm` (56 `mpm_*`), `terrain_mpm_fast`, `terrain_mpm_cuda` (16 `_gs` entries) | **Yes** (`CUDA_TERRAIN_KERNELS`, engine-cuda `build.rs:349-420`) | **No**: never run on a GPU; executor designed in `docs/deformable-terrain.md`, "CUDA plan" | shared: `engine_terrain::plan::TerrainPlan` (`gpu_terrain.rs:553`) |
| Terrain live surface, terrain watch | `terrain_surface` (9), `terrain_watch` (3) | No (Metal builds `terrain_watch`) | No | surface: shared (`SurfacePlan::frame`); watch: wgpu host (`gpu_terrain_watch.rs:95`) |
| Coupling emit, before the step: the last window's impulses onto the rigid bodies | `vehicle_coupling.slang`: `cpl_emit`, `cpl_emit_end` | **Yes** (C4: `CUDA_COUPLING_KERNELS`, engine-cuda `build.rs` `coupling_kernels`) | **Yes** (C4: `CudaWorld`, `cpl_emit_gs` over the proxied list's count on the GPU) | shared (C4): `explicit_plan::coupling::world::emit`; the wgpu host replays it (`explicit_vehicles.rs` `Replay`) |
| **Rigid step** (broadphase, narrowphase, convex SAT, colouring, AVBD, sleep, CCD, archive, contact events; and the riders `encode_step` emits from `StepInputs`: vehicles, characters, projectiles, queries, legacy soft bodies, guidance, explosions, locks, `pose_pack`) | `world_step`, `convex_step`, `convex_sat_body`, `ccd_step`, `compact_solve`, `persistent_solve`, `reuse_copy`, `vehicle_step(_persistent)`, `character_step`, `soft_step`, `guide_step`, `pose_pack` | **Partly**: the 126 `PERSISTENT_ENTRIES`. `encode_step` can emit 37 entries that are not among them (after `resolve_entry`): the sleep archive (`archive_*`, 12), the legacy soft bodies (`soft_*`, 15), guidance (`guide_*`, 5), `lock_predict`, `lock_project`, `explode`, `pose_pack`, `clear_solver_contacts` | **Partly**: `CudaSim` replays the same `encode_step`, but with fixed inputs (no joints, convex, heightfield or soil, vehicles, soft bodies, commands, guidance or explosions; `lib.rs:393-441`), and reads binding 23 back (7.1 stream syncs a step on the 10k columns) | shared: `rigid_plan::encode_step` (wgpu `gpu.rs:3471` replays exactly it) |
| In-step contact fracture, hooked into the rigid step at phase `"warm"` | stress-gpu `fracture.slang`: `fr_cs_*`, `fr_gi_solve` or `fr_csl_*`, `fr_gate`, the split | No | No | wgpu host, inserted into the shared plan through a phase hook (`threshold.rs:2382`, stress-gpu `fracture.rs:2128`) |
| Coupling after the step: bounds, candidates from the World's resting table, plan, share, loads, sleep mask | `vehicle_coupling.slang`: `cpl_bounds`, `cpl_query`, `cpl_plan`, `cpl_share`, `cpl_loads`, `cpl_loads_end`, `cpl_sleep_mask` | **Yes** (C4, all nine entries) | **Yes** (C4: `CudaWorld`), but the plan is always the scan of every collider: `cpl_query` is built and never launched, because `CudaSim` keeps no resting table for walkers (its awake-indexed broadphase is off and `pose_pack` is not built). The same proxies either way | shared (C4): `explicit_plan::coupling::world::frame` |
| Explicit soft-body frame (node–beam vehicles; the coupling's explicit side: proxies, window impulses, `coupling_list` / `coupling_fold`, piece windows) | `explicit_beams*.slang` incl. `explicit_beams_rigid` | **Yes** (`CUDA_EXPLICIT_KERNELS`, `ENGINE_CUDA_EXPLICIT_MODULES`) | **Yes**: `CudaExplicitBeams`, device-side indirect (`{entry}_gs`); proxies from the CPU (`write_proxies` / `read_proxies`) or, in a `CudaWorld`, from `cpl_plan` on the GPU | shared: `explicit_plan::schedule`, `explicit_plan::coupling` |
| Fracture frame: ingest, loads, breaks, strikes, crush, Blast stress solver (`blast_parity`), split, crush apply, log | stress-gpu `coupling.slang` (`ingest_*`), `fracture.slang` (`fr_*`), `blast_fracture`, `blast_topology`, `blast_motion`, `blast_cgls` or `blast_native(_wide)`, `blast_wide_gate`, `blast_virial`, `blast_material`, `blast_view`, `fracture_log` (of stress-gpu's 43 modules) | **No**: compiled to Metal and WGSL only (`stress-gpu/scripts/compile_shaders.sh`), outside `engine-slang-build` | **No** | **hand-recorded wgpu** (stress-gpu `ingest.rs`, `fracture.rs`, `blast.rs`; from engine-world `threshold.rs:2418`). `docs/stress-5d-one-command-buffer.md` requirement 4 (a backend-neutral plan) is "not met" |
| Blast pressure loading (`stress` feature), before encoding | (CPU windows and breaks, then stress-gpu `encode_split_crush`) | No | No | wgpu host, **blocking**: reads bodies and frames, submits the split on its own, reads counters, crush events, chunk bodies, bonds (`blast_threshold.rs:576`, `:715-750`) |
| Soft skins (render), ray queries (`raycast_batch`), overlap queries (`overlap_query`) | `soft_skin` (3), `vehicle_step` `raycast_batch`, `world_step` `build_aabbs` + `overlap_query` | `build_aabbs` only | No | wgpu host (`soft_skin.rs:189`, `gpu.rs:4732`, `gpu.rs:2347`) |
| Readback ring: counters, events, moved poses, terrain / coupling / fracture extras, loop-cap and coupling status | copies | — | `CudaSim::drain` reads counters (a sync); no ring | wgpu host (`readback.rs:114`); the step waits only when every slot (3) is in flight and the step must block (fracture, contact events, queries, rays on) |

Not in the World step (checked): the GPU punch and `threshold_predict` (stress-gpu
`threshold.slang`) and `GpuWorld::correct_frame` belong to the stress lab's own
`GpuWorld`; `fracture_step.slang` is used only by the `fracture_export` bin;
`learned_impact.slang` by no host. No World-path kernel uses subgroup operations
(stress-gpu `world.slang`, the stress lab's, does).

What `run.sh --world` times (`vehicle_lab.rs:2964-3009`): `before_step; world.step();
poll_blocking()` against the explicit frame alone, so the rigid step, both coupling
groups, the ring copies and the host's encoding. No fracture, no terrain on the matrix
rows.

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
| Rigid step (`CudaSim`) | ~57 entries dispatched indirectly (`Cmd::Indirect`); host reads binding 23 back after one of the 5 kernels that write it: **7.1 stream syncs a step** on the 10k columns | the counters, every step (`drain`) | contact hash, colour lists, archive, warm hash: atomic counters and CAS; already under the L2 rule | `_gs` for the indirect entries (as #142 did for the explicit modules); counters into a one-frame-late pinned read; the World's `StepInputs` instead of fixed ones (joints, convex, heightfield, soil, vehicles, commands), and the 37 missing entries built as the inputs that emit them are taken on |
| Coupling, World side (`cpl_*`) | `list_args`: the count `cpl_plan` writes is copied to the args and dispatched in the same submission (`explicit_vehicles.rs:760, 770, 806`); `cpl_emit` reads the args the previous step wrote | none waited on: the status word every `STATUS_EVERY` windows, loop-cap / status / sleep words through the ring | integer `Atomic<uint>.add` on `gWork` / `gScratch` (impulse append, `vehicle_coupling.slang:1069`) and groupshared atomics; 8 `GroupMemoryBarrierWithGroupSync`; no float atomics | `_gs` for the list dispatches; its dispatch order out of `explicit_vehicles.rs` into a shared plan (`explicit_plan::coupling` gains `Cmd`s); one CUDA context for `CudaSim` and `CudaExplicitBeams`, since the passes bind rigid bodies and explicit records together |
| Terrain | 16 `_gs` entries built | `RegionClock` substep count (one frame late), counters | P2G sums ordered; built under the L2 rule; no hazards found | the executor in `docs/deformable-terrain.md` "CUDA plan" step 4: bindings 37–54 in `CudaSim`, `TerrainPlan` lowered with `_gs`, counters to pinned memory |
| Stress / fracture | every group: a `sizes → args` copy then indirect dispatches in the same submission (the split alone: 19); the Blast gate writes its own args per iteration | in the World, none to run (splits are applied on the GPU, the host adopts them from the ring); the Blast pressure loading blocks (5 reads and its own submission); `docs/stress-5d-one-command-buffer.md` §3 lists the stress lab's resident-path reads | CAS union-find loops (`blast_topology.slang:82,146`, `fracture.slang:486,517`), 51 `Atomic<uint>.add` sites in `fracture.slang`; no float atomics (fixed-order reductions); the split writes engine bodies directly, so it shares the rigid step's buffers; the barrage has a latent GPU race (charter log) that CUDA will expose, not hide | (a) build the stress modules through `engine-slang-build` for CUDA; (b) the stress stage as a backend-neutral command list (5d requirement 4, needed for Metal and the web anyway), with the `"warm"` hook as a plan point; (c) a `RigidEngine` over `CudaSim` for the stress lab (stress-rubble drives `WgpuSim`); (d) the pressure loading's reads moved to the ring or the GPU |

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
`rigid_plan` layout, written once on the Mac), so nothing on Modal loads an asset. The World-level scenes of
`engine-world-lab/examples/rigid_gate.rs` (Brick Ring 110k, Ragdoll Pile 10k, Hinge Chains 10k,
Sphere Pile 10k, 100k Grid; `GATE_MODE=hash` prints body hashes at checkpoints) are the
next ones: the ragdolls and chains need joints, which `CudaSim`'s fixed inputs leave out
and C3 brings. If a scene turns on the sleep
archive, its 12 `archive_*` entries come here. Verdict: the same assertions as on wgpu (heights held, colour
coverage, no overflow), plus ms/step at 10k and 100k. This is the CUDA answer to "10,000+
active rigid bodies" for every later perf change.

**C3. The World's rigid step on CUDA (`run.sh --world --cuda`, first half).** `CudaSim`
takes the World's `StepModes` / `StepInputs` instead of its fixed ones (joints, convex,
heightfield, soil, vehicles, commands), and the entries those inputs emit are built (the
37 above, as each input is taken on; for the matrix rows at least what their scenes
use). A portable World fixture per matrix row (`matrix.py --make-portable` extended):
the rigid buffers, host words and inputs at the row's moment, beside the explicit fixture
it already writes, both written on the Mac. A `world_replay` example replays N steps of
**rigid step + explicit frame** in one CUDA context and stream and prints the gate's
line (`World step`, `frame alone`, `the World adds`), marked "coupling not on CUDA" until
C4. Verdict: the rigid step of every row bit for bit between two CUDA runs and its
`download_bodies` within the wgpu host's tolerances after N steps; wall clock per row.

**C4. The coupling (done, §6).** Move the World side's dispatch order (gather proxies, candidate
list, fold, apply window impulses, status) from engine-world into
`engine_exec::explicit_plan::coupling` as `Cmd`s, so wgpu follows it first (bit for bit
against today on the Mac: `vehicle_lab --ab-switch coupling` and the matrix) and then
CUDA replays it. Build `vehicle_coupling.slang` for CUDA. Verdict: every matrix row in
`run.sh --world --cuda`, and `explicit_ab` gets the `coupling_list` / `coupling_fold`
switches it reports as unavailable today. After C4 the whole `run.sh --world` gate
runs on CUDA.

**C5. Terrain.** The executor in `docs/deformable-terrain.md` (kernels and `_gs` entries
exist). Verdict: `terrain_parity` scenarios against the wgpu host's tolerances.

**C6. Stress and fracture: the fracture world on CUDA.** Everything the World's
destruction runs is one GPU stage: stress-gpu's `GpuFracture`, `EngineIngest` and
`BlastFrame` (13 of the stress shaders, 170 compute entries), behind every stress model
(`StrengthModel::Threshold`, `LoadPath`, `ImpactLocal`, `BlastGpu`) and crushing. The
stress lab's own `GpuWorld` (the `blast_parity` profiles, the explicit islands) is a
separate host and is not ported: it stays on wgpu as a reference.

1. **Done (2026-11-28).** The 13 modules built for CUDA by `engine-cuda` (feature
   `stress`, `build.rs` `fracture_kernels`): an entry a module, the hazard check, the
   `{entry}_gs` entry, fatbins with the L2 rule when nvcc is present.
2. **Done (2026-11-28).** The stage as a backend-neutral command list:
   `stress_layout::plan` (`frame`, `contact_fracture`, `split`, `crush`, `log`,
   `ingest::*`, `solver::*`). The wgpu host replays it; `STRESS_FRACTURE_RECORDER=legacy`
   keeps the old recorder as the reference.
3. **The host's set-up and I/O without wgpu.** `GpuFracture` still creates, writes and
   reads its buffers through wgpu (`Gpu::storage`, `write_buffer`, `read_range`: about 60
   sites), and builds its bind groups there. The buffers' sizes and first contents, the
   per-frame host writes (`sync_frame`, the quiet streak, the host's breaks, shots, crush
   flow) and the readbacks move behind a small device interface (allocate, write, copy,
   read a range, replay a list) that wgpu and CUDA implement, so one `GpuFracture` runs on
   both. Not a `RigidEngine` over `CudaSim`: that trait is the CPU coupling's. The seam
   the GPU path needs is the rigid host's buffers (bodies, colliders, contacts, counters;
   the split writes bodies and colliders in place), its contact ring offsets, and a hook
   at the rigid plan's `"warm"` phase, which only `WgpuSim` has today
   (`step_encode_with`).
4. **The CUDA executor** (`engine-cuda`, feature `stress`): banks filled from reflection
   and the plan's buffers, the lists replayed on the one stream (`Indirect` and
   `IndirectAt` through `{entry}_gs`), `CudaSim::step_submit_with(phase, hook)`, one
   pinned status read a frame, one frame late. A check that no `_gs` entry writes the
   arguments it is launched from (the wide solve's `w_gate` rewrites them between
   launches, which is in order on a stream).
5. **The fracture world on CUDA.** stress-rubble's `ThresholdWorld` over that host
   interface, `stress_bench --solver fracture-gpu --backend cuda`, then `World` generic
   over its rigid host (`World<H: RigidHost = WgpuSim>`) so the `engine-world` suites
   (`threshold`, `stress_models`, `strike_prediction`, `support_loss`, `crush`, `blast`,
   `reactions`, `city_scale`) run on CUDA unchanged. The city also needs the sleep
   archive, the awake-indexed broadphase and `pose_pack` on CUDA (C7's). The blast
   pressure loading is ported as it is first (its blocking reads counted), then its face
   loads move to a kernel.

Open before step 4: the loop audit of these kernels. Slang's CUDA for the 170 entries
holds about 490 `for(;;)` loops (fracture 225, the wide solve 101, `blast_native` 66,
`blast_motion` 52); the CAS union-find loops (`fracture.slang`, `blast_topology.slang`)
and any other without a bound from its data get one, with a status bit, before any of
them runs on a GPU that cannot be reset from here.

Verdict: `stress-rubble/tests/fracture_models.rs` and the threshold tests on a T4; two
CUDA runs of each scene bit for bit; CUDA against wgpu on the same scene (first break
identical, totals within `engine-world/tests/stress_models.rs`'s tolerance); the stress
solver's bond impulses within 3e-2 of its f32 reference.

The accuracy side of the same goal (every model scored against the CPU reference,
`stress-ref`, scene by scene, with the reference run on Modal's CPUs) is a second track
with no file in common with this one until a model changes.

**C7. The rest, as demand appears:** the rigid step's remaining riders (legacy soft
bodies, guidance, locks, explosions, the sleep archive if C2 did not need it,
`pose_pack`), ray and overlap queries and the ring's extras, skinning, terrain surface
and watch, and the Blast pressure loading's blocking reads.

## 4. Cost per verdict (Modal T4)

Prices as in `tools/modal/README.md` ("Cost and turnaround"): a T4 container with 8 cores
and 8 GiB is about $0.00029 a second ($0.017 a minute). The build is local and free;
Modal bills container seconds only. Measured numbers are marked; the rest are estimates
from the closest measured run, to be replaced by the first real run of each.

| Verdict | Containers | Billed (container s) | Cost | Local build after a kernel edit |
|---|---|---|---|---|
| cudasim correctness gates (640 / 10k / 100k columns, coverage) | 1 | 6.1 (measured, 2026-11-21) | $0.0018 | rigid kernels only: ~70 s when all recompile; the fatbin cache skips unchanged ones |
| C1: 640 / 10k / 100k bit for bit both ways, plus 10k / 100k timed both ways | 1 | 20 (measured) | $0.006 | as above |
| C2: 110k ring, both CCD fields, every cudasim gate | 1 | 38.6 (measured) | $0.011 | as above |
| C4: engine-cuda `tests/world.rs` (the coupling against the CPU reference, the wgpu World and itself) | 1 | 7.9 (measured, 2026-11-22) | $0.0023 | the nine coupling entries: seconds; the explicit modules the tests load, cached by content |
| C4: `run.sh --world --cuda`, the 8 `--world` rows | 1 (1 core, 2 GiB) | 20.8 (measured, 2026-11-22) | $0.0038 | rigid + coupling + the 36 explicit modules the rows' vehicles may launch (all cached by content that day: 15 s) |
| Explicit matrix `run.sh --cuda SWITCH`, 12 rows (exists) | 12 | 50–120 (measured, 2026-11-19) | $0.02–0.04 | ~4 min |
| C5: `terrain_parity` | 1 | 10–30 (est.) | <$0.01 | terrain modules |
| C6: `blast_parity` quick tier digests | 1–N | 60–300 (est.; the tier is minutes on the Mac) | $0.02–0.09 | the 32 stress compute modules: 42 s (measured, 3 jobs) |
| For comparison: any new kernel version on the Vulkan route | 1 warm-up + runs | 1,100–1,800 + runs (measured) | ~$0.45 per kernel version, up to ~$2 when the warm-up hits the cap | none (the driver compiles on Modal) |

Every CUDA verdict above stays under $0.10, and most under $0.05: one Vulkan cold compile
pays for about ten to twenty of them. Use the T4 for verdicts and ranking (cheapest,
nearly always available); an L4 for absolute numbers.

## 6. Done and measured (2026-11-21, Modal T4, strict floats)

**C0, baseline.** The 10 CudaSim tests pass on the branch as it was. 10k awake columns:
1.82 ms a step, 7.1 host waits a step. One run: 6.1 container seconds, $0.0018.

**C1, the rigid step without host waits** (`engine-cuda`: `{entry}_gs` for all 126 rigid
kernels, `CudaSim::set_device_indirect`, `RUBBLE_CUDA_HOST_INDIRECT=1` for the old path).
No indirect rigid kernel writes binding 23 (checked on the generated CUDA), so reading the
counts on the device sees what the slot held at launch.

- *Lossless.* `cudasim_device_indirect_matches_readback`: at 640, 10k and 100k bodies, the
  bodies, counters, colouring and indirect slots are bit for bit at every checkpoint.
- *A finding on the old path.* The append-ordered stores (contacts 4, AABBs 9, candidates
  10, awake list 11, constraint ids 14, manifolds 25, solver points 27 and pairs 28) differ
  between **two runs of the readback path** as much as between the paths: CUDA does not
  repeat atomic order, and the solver's ordering makes the bodies come out the same anyway.
  The test compares those as the control does.
- *Grid.* Resident blocks alone were 12 % slower at 100k (the static split's tail: primal
  4.6 → 6.8 ms); 4x / 16x the resident blocks were 1.6 % / 0.5 % faster at 100k but 4 % /
  46 % slower at 10k (idle blocks). Occupancy of `_gs` equals the direct entry's for all but
  two kernels. Kept: the resident blocks or one per 64 bodies, whichever is more.

| Columns, every body awake | readback (old) | device (`_gs`) | change |
|---|---|---|---|
| 10k bodies, ms a step (3 rounds) | 1.480 / 1.488 / 1.498 | 1.402 / 1.401 / 1.401 | **−6 %** |
| 100k bodies, ms a step (2 rounds) | 11.75 / 11.78 | 11.17 / 11.17 | **−5 %** |
| host waits a step | 7.1 | 0 | |

One run of the suite with the A/B and timing: about 20 container seconds, $0.006.

**C2, the scale gates on CUDA** (`engine-cuda/tests/scale.rs`, `cudasim.rs`). All pass:

| Gate (same assertions as the wgpu test) | Result on the T4 |
|---|---|
| `ring_hold`: 110k brick ring (`engine-scenes`), sleep off, 480 steps | 1,284,737 contacts at step 0; flanks stand; ball through both walls (z 105 m at 8 s); colours valid at every checkpoint; 32 ms a step waiting every step |
| `ccd_scale`: 128 CCD balls, 16 walls | 0 phase-throughs (293 without CCD); 5.06 ms a step (4.49 without) |
| `ccd_scale`: 512 CCD balls, 64 walls (ignored on wgpu as a benchmark) | 0 phase-throughs (2,376 without CCD); 10.18 ms (9.23 without) |
| `columns_hold` 25k and 100k, sleep on | hold |

The colour check moved to `rigid_plan::validate_contact_coloring` (WgpuSim and CudaSim call
it; wgpu `gables_hold` passes on the Mac). Both suites in one container: 38.6 s, $0.011.

**C6 step 1, the stress modules on CUDA (measured 2026-11-21; built by a crate since 2026-11-28, below).** slangc
2026.19 compiles all 32 compute modules of `stress-gpu/shaders` to CUDA (286 entries, 7 s);
the 3 render modules have no CUDA form and need none. The hazard scan
(`cargo run -p engine-slang-build --example cuda_hazards -- *.cu`) found one instance of the
deferred-load miscompile, in `layout.slang`'s `lay_bond_ranks`; fixed in the source the
same way as `prefix_color_tiles_parallel` (the sum before the store), regenerated, and
`resident_gpu` and `threshold_split` pass on the Mac. nvcc (`measure_nvcc.py`, SASS 75, 3
jobs, emulated x86 Docker): all 32 compile, 42 s in all; the slowest are the stress lab's
`world` / `world_ic` (27 s each), then `blast_motion` 9 s, `fracture` 7 s.

**C6 steps 1 and 2 (2026-11-28).** `cargo test -p engine-cuda --features stress --test
fracture_kernels` (CPU): 170 entries of 13 modules, each once, its globals bound without
overlap. `engine-slang-build` compiles a source outside the `slang` tree from a named base
directory (`cuda_layout::compile_kernel_at`), so the CUDA text is the same in every
checkout, and its compile cache now follows `import name;` (before, an import left the
compile uncached). `tools/build-time/kernel_digest.py`: 2,973 generated files identical to
main's. nvcc time for the 170 single-entry modules is not measured yet.

The command list (`stress_layout::plan`), each suite through the replay and through the
legacy recorder on the M3 Max: stress-gpu `fracture` 7/7 (and with
`STRESS_GPU_SHADERS=wgsl`), stress-rubble `ingest_gpu` 1/1, `threshold_split` 1/1,
`fracture_log` 1/1, `fracture_models` 11/11, `threshold_crush` 2/2, `blast_gpu_world` 9/9.
The plan's launch counts match the inventory above: a threshold frame 28 and the ingest's
9; the in-step contact fracture 34 (threshold), 37 (load path), 42 (analytic strike); the
solver's frame 492 at 16 iterations. Found on the way, broken on main since the quiet
gate: `ingest_any`, `ingest_gate` and `fr_quiet` were not in the hand-kept entry tables,
so the Metal library path could not build their pipelines (stress-gpu's `fracture` tests
failed); the new CPU test `entry_tables` holds every table against the generated WGSL.

The World's suites on the same machine, each through the replay and through the legacy
recorder: `stress_models` 3/3, `strike_prediction` 4/4, `support_loss` 10/10, `blast` 6/6,
`crush` 12/12, `reactions` 7/7. `threshold` 5 of 6 both ways: its
`cuboid_house_breaks_and_splits_as_in_the_lab` fails the same way on main's own build
(b1bf3814f: the World's first break at frame 95 with 38 bonds, the lab's at frame 2 with
2), so it is not this change's. Not run yet: the lab pages in Chrome.

C0 again on this branch (2026-11-28, `gpu_run.sh --build modal -- -p engine-cuda --test
cudasim`): 13 of 13 on a T4, 12.7 container seconds, $0.0036; the first build of the
branch on Modal took 650 s, the next 11 s.

Corrections to this document's earlier text, from the code: the split records 19 indirect
dispatches; the stress shaders are 44 `.slang` files of which 32 are compute modules; the
barrage's race is closed in the charter's log (six runs, one digest) and is the stress
lab's scene, not the World's; the blast pressure loading waits more than "5 reads" (every
ring slot, bodies and frames per slot before the split, six reads after it, and the
bodies again for the impulses, on steps without a split too).

**C3, first part: one rigid host state, joints and convex hulls on CUDA.**
`engine_exec::rigid_host` now holds what the plan hosts each copied: body motion flags,
collider flags (convex, height field, filters), joint slots and offsets, the automatic
broadphase grid; each body's reach and a joint's default torque arm are in
`engine_schema::joint`, so `HostWorld` (the scene export) shares them too. WgpuSim, CudaSim
and MetalSim call them. MetalWorld keeps its own hand-encoded step (it never replays the
plan) and is left as it is.

- *WebGPU unchanged, proven:* `rigid_gate` (11 scenes, sleep off) hashes identically before
  and after, and two runs of the old build agree with each other.
- *Two fixes that come with sharing:* CCD counts `BODY_CCD_FAST` on CUDA and MetalSim (only
  WebGPU did); MetalSim and CudaSim fall back to the hierarchical grid when the uniform
  grid's large list overflows (they failed the step).
- *CudaSim* takes joints (`add_joints`, `add_joint_desc`), the hull arena (`set_hulls`;
  larger than the capacity is an error) and feeds the scene's flags to `StepInputs` and
  the host words.
- *One scene gate* (`engine_scenes::invariants::gate`), run by engine-webgpu
  `scene_gate.rs` and engine-cuda `scenes.rs`: finite state, valid colouring, nothing
  through the ground, locked joints within bars set from the WebGPU reference (largest gap
  0.25 m, mean 0.03 m; the reference's Ragdoll Pile opens 0.137 / 0.023 m).

| 10k scene, all pass | WebGPU, M3 Max: ms a step | CUDA, T4: ms a step | joint gap max / mean, M3 / T4 |
|---|---|---|---|
| Convex Pile | 5.13 | 14.2 | — |
| Ragdoll Pile | 4.54 | 6.20 | 0.137 / 0.023, 0.216 / 0.023 |
| Hinge Chains | 1.35 | 1.77 | 0.020 / 0.013, 0.020 / 0.013 |

Observations from one run each, to check before relying on them: CUDA's ragdolls end with
more kinetic energy at step 300 (45.5 J against 10.0) and their largest gap is wider; the
convex pile's lowest body dips to −0.025 m at step 200 on CUDA (0.108 on WebGPU) and is back
at 0.375 by step 300. The convex pile costs 2.8 times the Mac on a T4 (the explicit rows'
T4/Mac ratio is 1.0 to 2.0): a CUDA convex profile is a candidate follow-up.

**C3, second part: height fields, soil and convex bodies.** `rigid_host` also builds a
height field's support body and collider, a convex body's cooked state and colliders, and
finds a soil in the hull arena; WgpuSim and MetalSim use them, CudaSim gains
`add_height_field` and `add_convex_body`. engine-cuda `tests/heightfield.rs` runs
engine-webgpu `heightfield.rs`'s two reference checks with its scene; both pass on a T4
(a sphere rolls 0.9139 m down the 15° field and the 15° plane alike; analytic 0.9068 m).

*Follow-up found on the way (pre-existing, every plan host):* without a hull in the scene
(the uniform grid) a lying capsule of radius 0.25 m rests at 0.2421 m on a plane and at
0.2354 m on a flat height field, on CUDA and on MetalSim to 1e-6 m; with a hull (the
hierarchical grid) it rests at 0.235 on both. Which depth is right, and why the grid
changes it, is for a rigid-solver session.

**C3 is closed here.** Every rigid input the World gate's rows use (boxes, spheres,
capsules, convex hulls, height fields, joints) runs on CUDA through the shared host state.
Not on CUDA yet, and not on those rows: vehicles' raycast wheels, characters and queries,
guidance, explosions, motion locks, the sleep archive (the 37 entries `encode_step` can
emit that CUDA does not build). **Decided:** the portable World fixture and `world_replay`
move into C4. The gate's number is the rigid step, the explicit frame *and* the coupling
together; a replay without the coupling would print a World that does not exist.

**C4, the coupling: the World step on CUDA (2026-11-22).** The whole `run.sh --world`
step now runs on CUDA: emit, rigid step, plan, explicit frame, loads, on one stream.

- *One plan.* The World side's order and parameters moved out of engine-world into
  `engine_exec::explicit_plan::coupling::world`: the nine entries, each binding's source
  and access, the `Params` block and its builder, the work buffer's layout, the window
  clock (`Clock`: the steps owed, the pending emit) and the commands of `emit()` and
  `frame()`. `rigid_host::GpuImpulses` is the flag `mark_gpu_impulses` sets (it becomes
  `MotionInputs::impulses`); WgpuSim and CudaSim both hold it.
- *The wgpu host follows it, unchanged, proven.* `explicit_vehicles.rs` only allocates,
  binds and replays (`Replay`). On Metal (M3 Max) `tests/explicit_world_digest.rs` (two
  sedans onto twelve boxes, 90 steps, bit-exact vehicles) prints the same three hashes
  before and after (vehicles `6f27744e66371c85`, bodies `4d43ab986f580753`, window
  impulses `d9f168781e626775`), and the coupling's suites pass (`explicit_vehicles` 11,
  `explicit_vehicle_broadphase` 3, `explicit_sleep`, `explicit_loop_budget`,
  `explicit_drive_world`, `wheels_hit_bodies_world`, `surface_pairs`,
  `vehicle_lab_piece_windows`, `vehicle_lab_reset` 3). `run.sh --world` on the Mac:
  PASS, the World adds 0.49 to 1.43 ms on the 8 rows.
- *`vehicle_coupling.slang` on CUDA.* engine-cuda `build.rs` `coupling_kernels`: an entry
  a module, `{entry}_gs` for the list dispatches (`cpl_share`, `cpl_loads`, `cpl_emit`),
  the hazard check (nothing found: no source change), the loop caps' presence, the L2
  rule. Slang keeps all fifteen globals in every entry's bank, so one set of pointers
  serves `cpl_query` too.
- *`CudaWorld`* (`engine-cuda/src/world.rs`) owns a `CudaSim` and a `CudaExplicitBeams`
  in the device's primary context (they shared it already) on the rigid host's stream,
  and the coupling's params, work and list-argument buffers. `step(dt)` queues the
  window's parameters, the emit, the rigid step, the plan, the frame and the loads with
  no host wait; the step's counters, the frame's loop-cap flag, the coupling's status
  word and the asleep bits go to pinned memory behind an event (a ring of three, as the
  wgpu readback ring) and are adopted by a later step: one step late when the caller
  waits each step, which is also what makes two runs bit for bit.
- *Not the same as wgpu, on purpose stated:* the plan is always the scan of every
  collider. `cpl_query` (candidates from the World broadphase's resting table) is built
  but never launched: `CudaSim` keeps no resting table for walkers (awake-indexed
  broadphase off, `pose_pack` not built). The proxies are the same either way
  (engine-world `explicit_vehicle_broadphase.rs`, `the_broadphase_plan_is_the_scan_*`);
  the matrix's rows were planned by the scan on wgpu
  too (their Worlds keep no resting table), so the gate compares like with like there.
  The threshold ingest (chunk loads into a fracture) has stand-in buffers: no fracture
  on CUDA (C6).

Verdicts on a T4, strict floats (`ENGINE_CUDA_FMAD=0`), `engine-cuda/tests/world.rs`
(7 tests, one container, 7.9 s, $0.0023):

| Check | Result |
|---|---|
| Explicit side against the CPU reference coupling (`ProxySet`, proxies written by the host; engine-metal's `coupled_sedan_into_a_rigid_box`) | first contact frames within 6.0e-8 m; box 10.8040 m/s against 10.8041; momentum closed to 0.0079 of 56,441 N s |
| World path, momentum (engine-world's `a_vehicle_shoves_a_free_box_and_momentum_closes`): sedan at 15 m/s into a free 1200 kg box, plan, frame, emit and rigid step on the GPU | worst error 0.023 of 56,441 N s (4e-7); box 13.10 m/s, between the plastic 11.37 and the elastic 22.75 |
| Two sedans dropped onto a static deck and onto dynamic boxes over a static slab, 120 steps | finite; lowest nodes 0.4996 m (deck top 0.5) and −0.0007 m (slab top 0); each vehicle's windows put −616.1 and −615.3 N s a window on the bodies (its weight: −615.0); boxes pushed, none thrown; status words zero |
| The same, two runs | explicit state bit for bit, every state word; 0 of 8 bodies differ |
| The pile with a dynamic deck against the wgpu World on Metal (M3 Max), step for step | lowest nodes within 9.5e-5 m at steps 19, 59 and 119; deck height equal to 1e-4 m; window impulses −614.5701 / −615.3270 N s against −614.5701 / −615.3169 |
| Vehicle sleep on against off (`cpl_sleep_mask` every window, no vehicle asleep) | no state section differs |

*The portable World* (`engine_exec::world_portable`, `ROW.wfx`): the rigid scene (bodies
with their previous rows, colliders, hull arena, joints, materials, parameters,
capacity), the coupling's carried state (clock, pending emit, work buffer with the
vehicle table, the proxied list and the broadphase proxies, the indirect arguments), the
vehicles' portable explicit state and each vehicle's drive input for the 30 steps after
the moment. Written by the wgpu World (`World::save_explicit_world`, `vehicle_lab
--save-world`, `run.sh --make-world`: the row's scene run to its moment, no `.fx`
needed); read by `CudaWorld::from_portable`. **It holds the scene, not the solver's
caches**: a replay starts its contacts cold, so `world_replay` runs 5 warm-up steps
before it times, and its bodies follow the saved World's only as closely as a cold start
allows. Beyond drive inputs, what the lab does before a step (piece labels, retired
nodes) is not replayed.

`tools/scenario-matrix/run.sh --world --cuda` (engine-cuda `world_replay`, all 8 `--world`
rows in one container, 1 core: 16.4 s of replay, 20.8 container seconds, **$0.0038**),
ms a step on a T4; the Mac column is `run.sh --world` on Metal from the same session
(wall, another session sharing the machine's CPU):

| Row | Soft bodies, rigid bodies | T4: World step | frame alone | the World adds | of it the rigid step (GPU) | the coupling's passes (GPU) | T4 step, GPU: p95 / worst | Mac (Metal): the World adds |
|---|---|---|---|---|---|---|---|---|
| drive | 1, 3 | 4.104 | 3.791 | **+0.313** | 0.254 | 0.047 | 4.210 / 4.292 | +0.505 |
| wall | 1, 4 | 4.383 | 4.067 | **+0.316** | 0.255 | 0.054 | 4.486 / 4.604 | +0.541 |
| headon | 2, 3 | 16.529 | 16.221 | **+0.307** | 0.268 | 0.052 | 17.300 / 17.814 | +0.593 |
| side | 2, 3 | 10.686 | 10.341 | **+0.345** | 0.266 | 0.051 | 11.094 / 12.523 | +0.520 |
| bricks | 1, 63 | 5.299 | 4.255 | **+1.045** | 0.975 | 0.089 | 5.474 / 5.551 | +1.428 |
| rocks | 1, 7 | 4.685 | 3.914 | **+0.772** | 0.524 | 0.058 | 4.899 / 4.969 | +0.984 |
| barrier | 2, 3 | 10.304 | 9.984 | **+0.320** | 0.260 | 0.049 | 10.659 / 10.740 | +0.574 |
| truss | 1, 3 | 1.562 | 1.264 | **+0.299** | 0.252 | 0.047 | 1.592 / 1.616 | +0.487 |

- *Every row's second replay from the file ends bit for bit* (explicit state words and
  bodies), and the end-state hashes are the same in three separate runs (three
  containers) of the same files.
- *What the World adds on CUDA is the rigid step's fixed cost.* With 3 bodies the rigid
  step is 0.25 to 0.27 ms: its launches (a fixed cost a step), not its work. The
  coupling's own passes are 0.05 to 0.09 ms together (emit 0.011 to 0.045, plan 0.018 to
  0.027, loads, asleep bits and the status read 0.017). The bricks row's 1.05 ms is
  0.98 ms of rigid step for 63 bodies. The Mac's 0.49 to 1.43 ms is wall time with
  the host's encoding in it, on a machine another session was using.
- *Per step, not only the median:* `world_replay` prints every step's GPU ms, the median,
  p95 and worst step and which it was. The worst step is 4 to 8 % over the median on
  seven rows; side's is 12.5 ms against 10.7 (17 %: the first step from the moment, in every round).
- *A measurement trap found and closed:* the first row a fresh container replays read 14
  to 25 ms a step in its first round against 10.5 in the next two, differently from run
  to run, with bit-identical results: a T4 runs its first seconds of work slower. It is
  not module loading (`CudaExplicitBeams::preload` loads every module before timing and
  changed nothing). `world_replay` now replays the first file once, unreported, before
  it times anything (`warm_gpu`); the barrier row's worst step went from 25.5 to
  10.7 ms. `explicit_replay` has its own warm-up frames (`AB_WARM_MS`).

*Found on the way (pre-existing, both hosts; for a coupling session):* a vehicle resting
on a **dynamic** body that itself rests on a static one sinks through it. The pile test's
sedan on a 3000 kg deck lying on the slab goes down at about 0.2 m/s (0.44 m to 0.11 m
between steps 19 and 119) on Metal and on CUDA alike, while its windows put its full
weight on the deck: within a window the explicit solver integrates the deck's proxy as a
free body, and the slab's support reaches the vehicle only as the deck's reset at the
next window. On a static body (or the vehicles' own ground) it rests (−0.0007 m).
`tests/world.rs` holds CUDA to the wgpu World's numbers for this scene; it does not
judge them.

**C4 is closed here.** Not done, and why: `cpl_query` on CUDA (needs the rigid host's
resting table: C7); `explicit_ab`'s `coupling_list` / `coupling_fold` switches (the
explicit host has `set_coupling_header`; the A/B example was not extended); a two-build
A/B of the World step (`world_replay` reports one build; the explicit frame's A/B is
`run.sh --cuda --base`); the rows' bodies against the wgpu host's after N steps (the
cold start makes that a tolerance, not a bit check; the pile test above is the
cross-host check that exists).

## 5. Open questions to settle while building (not blockers)

- *Settled (C4):* one context, one stream. `CudaSim` and `CudaExplicitBeams` each
  retained the device's **primary** context already, so their pointers were always valid
  in each other's modules; they only had a stream each. `CudaWorld` checks the contexts
  are one, moves the explicit host onto the rigid host's stream
  (`CudaExplicitBeams::use_stream`) and fills the coupling's banks with pointers into
  both hosts' buffers.
- Whether the wgpu host's World step encodes anything around `encode_step` that the
  shared plan does not (pose copies, timer marks, the terrain's and fracture's passes
  between engine substeps). Those belong in a World-level schedule, not in each host.
- The Mac stays the reference for final timings; CUDA numbers rank changes and catch
  regressions, as the README's "What compares with the Mac" says.
