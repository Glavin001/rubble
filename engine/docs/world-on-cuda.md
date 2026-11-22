# The World on CUDA (Modal): plan

Status (2026-11-21): **C0 to C3 done and measured; C6 step 1 measured** (§6). Next: C4
(the coupling as a shared plan, with the portable World fixture and `run.sh --world --cuda`).
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

| Stage of a World step (`World::step`, `crates/engine-world/src/lib.rs:2215`) | Slang | Built for CUDA today | Host on CUDA today | Schedule |
|---|---|---|---|---|
| Terrain MPM coupled substeps (first: the rigid step applies their reactions) | `terrain_mpm` (56 `mpm_*`), `terrain_mpm_fast`, `terrain_mpm_cuda` (16 `_gs` entries) | **Yes** (`CUDA_TERRAIN_KERNELS`, engine-cuda `build.rs:349-420`) | **No**: never run on a GPU; executor designed in `docs/deformable-terrain.md`, "CUDA plan" | shared: `engine_terrain::plan::TerrainPlan` (`gpu_terrain.rs:553`) |
| Terrain live surface, terrain watch | `terrain_surface` (9), `terrain_watch` (3) | No (Metal builds `terrain_watch`) | No | surface: shared (`SurfacePlan::frame`); watch: wgpu host (`gpu_terrain_watch.rs:95`) |
| Coupling emit, before the step: the last window's impulses onto the rigid bodies | `vehicle_coupling.slang`: `cpl_emit`, `cpl_emit_end` | **No** | **No** | wgpu host (`explicit_vehicles.rs:681`, from `:1210`) |
| **Rigid step** (broadphase, narrowphase, convex SAT, colouring, AVBD, sleep, CCD, archive, contact events; and the riders `encode_step` emits from `StepInputs`: vehicles, characters, projectiles, queries, legacy soft bodies, guidance, explosions, locks, `pose_pack`) | `world_step`, `convex_step`, `convex_sat_body`, `ccd_step`, `compact_solve`, `persistent_solve`, `reuse_copy`, `vehicle_step(_persistent)`, `character_step`, `soft_step`, `guide_step`, `pose_pack` | **Partly**: the 126 `PERSISTENT_ENTRIES`. `encode_step` can emit 37 entries that are not among them (after `resolve_entry`): the sleep archive (`archive_*`, 12), the legacy soft bodies (`soft_*`, 15), guidance (`guide_*`, 5), `lock_predict`, `lock_project`, `explode`, `pose_pack`, `clear_solver_contacts` | **Partly**: `CudaSim` replays the same `encode_step`, but with fixed inputs (no joints, convex, heightfield or soil, vehicles, soft bodies, commands, guidance or explosions; `lib.rs:393-441`), and reads binding 23 back (7.1 stream syncs a step on the 10k columns) | shared: `rigid_plan::encode_step` (wgpu `gpu.rs:3471` replays exactly it) |
| In-step contact fracture, hooked into the rigid step at phase `"warm"` | stress-gpu `fracture.slang`: `fr_cs_*`, `fr_gi_solve` or `fr_csl_*`, `fr_gate`, the split | No | No | wgpu host, inserted into the shared plan through a phase hook (`threshold.rs:2382`, stress-gpu `fracture.rs:2128`) |
| Coupling after the step: bounds, candidates from the World's resting table, plan, share, loads, sleep mask | `vehicle_coupling.slang`: `cpl_bounds`, `cpl_query`, `cpl_plan`, `cpl_share`, `cpl_loads`, `cpl_loads_end`, `cpl_sleep_mask` | **No** | **No** | wgpu host (`explicit_vehicles.rs:713-823`) |
| Explicit soft-body frame (node–beam vehicles; the coupling's explicit side: proxies, window impulses, `coupling_list` / `coupling_fold`, piece windows) | `explicit_beams*.slang` incl. `explicit_beams_rigid` | **Yes** (`CUDA_EXPLICIT_KERNELS`, `ENGINE_CUDA_EXPLICIT_MODULES`) | **Yes**: `CudaExplicitBeams`, device-side indirect (`{entry}_gs`); proxies only from the CPU (`set_proxies` / `decode_proxies`) | shared: `explicit_plan::schedule`, `explicit_plan::coupling` |
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
| Stress / fracture | every group: a `sizes → args` copy then indirect dispatches in the same submission (the split alone: 17); the Blast gate writes its own args per iteration | in the World, none to run (splits are applied on the GPU, the host adopts them from the ring); the Blast pressure loading blocks (5 reads and its own submission); `docs/stress-5d-one-command-buffer.md` §3 lists the stress lab's resident-path reads | CAS union-find loops (`blast_topology.slang:82,146`, `fracture.slang:486,517`), 51 `Atomic<uint>.add` sites in `fracture.slang`; no float atomics (fixed-order reductions); the split writes engine bodies directly, so it shares the rigid step's buffers; the barrage has a latent GPU race (charter log) that CUDA will expose, not hide | (a) build the stress modules through `engine-slang-build` for CUDA; (b) the stress stage as a backend-neutral command list (5d requirement 4, needed for Metal and the web anyway), with the `"warm"` hook as a plan point; (c) a `RigidEngine` over `CudaSim` for the stress lab (stress-rubble drives `WgpuSim`); (d) the pressure loading's reads moved to the ring or the GPU |

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
| C3/C4: `run.sh --world --cuda`, 12 rows | 12 | 60–150 (est., from the matrix) | $0.02–0.05 | rigid + explicit + coupling modules the rows load: ~4 min (the matrix's figure) |
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

**C6 step 1, the stress modules on CUDA (measured, not yet built by a crate).** slangc
2026.19 compiles all 32 compute modules of `stress-gpu/shaders` to CUDA (286 entries, 7 s);
the 3 render modules have no CUDA form and need none. The hazard scan
(`cargo run -p engine-slang-build --example cuda_hazards -- *.cu`) found one instance of the
deferred-load miscompile, in `layout.slang`'s `lay_bond_ranks`; fixed in the source the
same way as `prefix_color_tiles_parallel` (the sum before the store), regenerated, and
`resident_gpu` and `threshold_split` pass on the Mac. nvcc (`measure_nvcc.py`, SASS 75, 3
jobs, emulated x86 Docker): all 32 compile, 42 s in all; the slowest are the stress lab's
`world` / `world_ic` (27 s each), then `blast_motion` 9 s, `fracture` 7 s.

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

## 5. Open questions to settle while building (not blockers)

- C3's one-context composition: `CudaSim` and `CudaExplicitBeams` each create a context
  and stream today. Sharing one (or importing the other's pointers) is a host change of
  a few lines but decides whether the coupling's banks can point into both.
- Whether the wgpu host's World step encodes anything around `encode_step` that the
  shared plan does not (pose copies, timer marks, the terrain's and fracture's passes
  between engine substeps). Those belong in a World-level schedule, not in each host.
- The Mac stays the reference for final timings; CUDA numbers rank changes and catch
  regressions, as the README's "What compares with the Mac" says.
