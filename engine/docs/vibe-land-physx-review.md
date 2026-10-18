# What we must surpass: vibe-land's PhysX native GPU destruction stage

Read from the code on 2026-10-18:
- vibe-land at `3505ee5c`;
- PhysX at `ec8118341`. vibe-land's runtime profile builds against `0e0aba8a0`, installed at
  `/Users/glavin/Development/PhysX/out/install/garage-roof` (`scripts/fidelity/runtime.env:9-10`).

Below, the vibe-land repo is `VL` (`/Users/glavin/Development/vibe-land`) and the PhysX repo is
`PX` (`/Users/glavin/Development/PhysX`).

## 1. Which stress solver vibe-land actually runs

The PhysX repo holds several stress implementations. Only one is live in vibe-land's shipping
path and perf suite:

| Implementation | Path | Used by vibe-land? |
|---|---|---|
| **Native GPU destruction stage** inside the PhysX GPU pipeline (`PxDestructionScene`) | `PX/physx/include/PxDestructionScene.h`, `PX/physx/source/gpudestruction/src/` (`PxgDestructionRuntime.cu`, `PxgDestructionTopology.cu`, `PxgDestructionTransaction.cuh`, `PxgDestructionMotionSlots.cuh`, `PxgDestructionMotionState.cuh`, …), driven from `PX/physx/source/gpusimulationcontroller/src/PxgSimulationController.cpp:861` (`advanceDestruction`), called from `PX/physx/source/simulationcontroller/src/ScPipeline.cpp:3100` | **Yes: the live path.** `VL/server/src/perf_suite.rs:95` asserts `city.backend_name() == "native"`. Backend selection: `VL/server/src/city.rs:391-421` (`VIBE_CITY_DESTRUCTION`, `native` when built without Blast) |
| Blast GPU stress kernels (`stressgpu`) | `PX/blast/source/sdk/extensions/stressgpu/` (`NvBlastExtStressGpu.cu`, `detail/`) | Yes, **inside** the native stage: its structural solve (16 iterations, warm start). Not called on its own |
| Blast CPU `ExtStressSolver` | `PX/blast/source/sdk/extensions/stress/` | No (older `CityBackend::Physx` / "blast" backend) |
| Blast + PhysX adapter | `PX/blast/source/sdk/extensions/stressphysx/`; VL `physx-bridge/src/destruction.cc`, `destruction/src/runtime.rs`, `freeze.rs`, `settle.rs` | No (legacy backend; its freeze, aggregate and settle machinery is inert in the native path) |
| Rust stress oracle copy | `PX/destruction/stress-oracle/` | No (research) |

**On Mac it runs through CuMetal:** CUDA kernels translated to Metal. The app turns the GPU
keep-alive off (`VL/sim-native/src/city.rs:239-240`).

## 2. How a vibe-land tick runs (native path)

**Tick order** (`VL/server/src/perf_suite.rs:139-150`, the same as `MatchState::tick`):
1. players;
2. `city.pre_step` (a no-op for native);
3. **`arena.step_vehicles_and_dynamics(DT)`**: PhysX `simulate()` + `fetchResults()`. **The
   destruction stage runs inside it**, in `Sc::Scene::finalizationPhase`
   (`PX/…/ScPipeline.cpp:3053-3185`):
   1. trial collide and solve;
   2. stress solve on the solved contact impulses, read on the GPU;
   3. verdict;
   4. if bonds break: topology transaction, new bodies, **then one corrected collide+solve
      of the scene** (`internalCorrectionLimit = 1`).

   The host waits once, in `finish()` (`PxgDestructionRuntime.cu:2700-2704`);
4. `city.step`: observation, streaming. About 0.4 ms.

**Rigid-body settings:**
- **Solver:** PGS, 4 position and 1 velocity iteration (`VL/physx-bridge/include/solver_iterations.h:21-26`).
- **Step:** one 60 Hz step per tick, no substeps (`VL/physx-bridge/src/physx_bridge.cc:122, 2994`).
- **No CCD.** Scene sweep CCD disables the stage (`getDestructionScene()` returns null).
  Per-body speculative CCD on projectiles is off by default: "the same shot that broke 51
  bonds breaks zero" (`physx_bridge.cc:2436-2456`, `VIBE_CITY_BALL_SPECULATIVE_CCD`).
  - Walls thinner than about 0.4 m tunnel at 60 m/s.
  - Meteors at 140 m/s pass through without it.
- **Stabilization** on (`physx_bridge.cc:3903-3908`).
- **Fragment depenetration** 2 m/s while destructible vehicles exist
  (`VL/physx-bridge/src/native_destruction.cc:1507-1519`).
- **Damping** 0 on debris.
- **Chunks** collide with chunks (`VL/destruction/src/bridge_authoring.rs:10-19`). Chunk pairs
  produce no CPU contact reports (`physx_bridge.cc:574-594`).

**Stress settings:**
- **Iterations:** 16, forced by the town scene's `VIBE_CITY_NATIVE_STRESS_ITERATIONS`
  (`VL/scripts/perf/suite.json`). The default is 64 with destructible vehicles, 16 otherwise
  (`VL/destruction/src/native_runtime.rs:84-92`).
- **Tolerance:** 1e-3 (`native_runtime.rs:94-108`). Force tolerance 1e-3, from the suite's
  `app_settings` (`VL/server/src/perf_suite.rs:46-61`).
- **`PX_DESTRUCTION_ALLOW_UNCONVERGED=1`** (same place): unconverged solves publish their
  warm-started iterate. The town bombard is unconverged on 90 of 90 ticks
  (`VL/scripts/perf/suite-baseline.json`).
- **Correction limit:** 1 (`VL/physx-bridge/src/native_destruction.cc:32-50`). Above 1 is
  refused unless built with `VIBE_PHYSX_CORRECTION_LOOP`. The SDK default is 0
  (`PX/physx/include/PxDestructionScene.h:251`).
- **Incremental work, opt-in, set by the suite:** `PX_DESTRUCTION_INCREMENTAL_TOPOLOGY=1`
  (bit-exact; `PxgDestructionTopology.cu:111-130`), `BLAST_STRESS_INCREMENTAL_MOTION=1`,
  `BLAST_STRESS_BALANCED_OPERATOR=1`.
- **Bond verdicts sampled** every 60 ticks for observation (`native_runtime.rs:144-146`).

**How it handles a fracture on the GPU, with no per-body host work for physics data:**
- **`prepareCandidateBodies`** (`PxgDestructionRuntime.cu:426`, launched at `:2067`): mass,
  COM, inertia, principal frame and velocity of each new cluster body.
- **`checkUnchangedMotionCommit`** (`:412`): internal cuts that change no body commit directly.
- **Motion slots** allocated by GPU scans (`PxgDestructionMotionSlots.cuh`).
- **Bodies** initialized on the GPU (`PxgDestructionMotionState.cuh`).
- **Body storage** grows on demand, with a host stall (`reserveBodySlots`, `:2157`).
- **Rewind** is full-pool device copies (`captureRigidState` / `restoreRigidState`,
  `:2255` / `:2318`). The copies are cheap; the corrected re-solve is not.
- **Still on the CPU:** one `NpRigidDynamic` / `BodySim` per new fragment, from a pool. This is
  the stage's largest remaining cost at scale (`PX/docs/destruction/REPLACEMENT_PLAN.md:44`).

## 3. The numbers to beat (M3 Max, runtime profile)

`VL/scripts/perf/suite-baseline.json`, the town process.

**The process order** (`VL/scripts/perf/suite.json`, `processes[0]`):
- settle 30 ticks (not recorded);
- `town-idle` 60;
- `fleet-drive` 60;
- **`town-bombard` 90** (standard) or 60 (quick, the first 2 events only).

The scene has 22 destructible cars.

**`town-bombard` events.** Each fires before its tick (`perf_suite.rs:245-248`), as the suite
expands them (`VL/scripts/perf/suite.py:149-158`, `shot_from_place`, from
`vibe-town.meta.json` places):

| Tick | Kind | Place | Height |
|---|---|---|---|
| 0 | cannonball | elm-park/house-16 | 1.5 m |
| 10 | meteor | elm-park/house-18 | 2.0 m |
| 20 | cannonball | elm-park/house-23 | 1.5 m |
| 30 | meteor | market-quarter/cinema-1 | 2.0 m |
| 40 | cannonball | elm-park/house-25 | 1.5 m |

**Shot geometry** (`suite.py:149-158`, `perf_suite.rs:176-200`):
- target = (pavement x, height, the street-side face z);
- from the street, at 30 m and slope 0.02 for cannonballs, 140 m and slope 0.3 for meteors;
- velocity aimed with a gravity lead;
- cannonball: 10,650 kg steel (7,850 kg/m³) at 60 m/s (`VL/server/src/city.rs:172-198`);
- meteor: radius 2 m, 3,300 kg/m³, at 140 m/s (`VL/server/src/meteor.rs:47-48`).

| Tier | Median | p95 | Worst | Bonds broken | Peak awake | Peak contacts | Correction passes |
|---|---|---|---|---|---|---|---|
| **standard** (90 ticks, 5 events) | **11.46 ms** | **37.88 ms** | **48.24 ms** | 6,707 | 1,569 | 39,852 | 17 |
| quick (60 ticks, 2 events; run `20261017-152216`) | 11.38 | 15.32 | 35.5 | 638 | about 100 | 1,740 | |

- The 10–12 ms floor is the scene at rest: 22 cars plus the stress solve.
- The pack is `VL/target/perf-suite/packs/runtime/town/vibe-town.json`: 51,921 chunks and
  132,529 bonds, the same counts as `VL/structures/vibe-town/out/vibe-town.json`, with a
  different hash.

## 4. Where rubble must win, and how

| | vibe-land (PhysX native stage) | rubble target |
|---|---|---|
| **Fracture timing** | Trial solve, then **one corrected re-solve** of the whole scene per fracture tick. Splits from the last pass take effect at end of tick. The correction does not converge for fast impacts (`stress-oracle/findings/rewind-scaling.md`) | **Forward only:** stress evaluated and splits applied between rubble's substeps. The rest of the frame runs on the separated bodies. No re-solve |
| **Residency** | GPU stage plus one blocking wait per tick, but **CPU fragment bodies** (`NpRigidDynamic` pool) and host stalls on storage growth | **Everything on the GPU, in rubble's step graph:** contact ingest, cascade, splits, body creation. One submission and one status read per tick |
| **Contact fidelity** | No CCD: thin walls tunnel; at 140 m/s meteors pass through | rubble's CCD substeps (more faithful), made cheap enough |
| **Stress accuracy** | 16 PCG iterations, unconverged on every tick | Ours is more accurate per iteration: 0.93 vs 0.83 Pearson at 16 on house-1story |
| **Cost** | 11.46 / 37.88 / 48.24 ms | Below each, on the same events and pack, with breakage reported |

Our benchmark scene of record becomes `vibe-town_suite_bombard` and `_quick`, ported from the
tables above (`engine/stress-oracle/crates/stress-rubble/examples/stress_bench.rs`).
- **One known difference:** we have no destructible cars, and no idle or drive phases before
  the bombard. The town settles for 30 frames, then the 90 bombard frames are recorded.

## 5. First baseline on the ported scene (2026-10-18)

Run with `--config blast --authored-fatal --pack-masses --method blast_model=on`, the resident
quasi-static frame (`STRESS_RESIDENT=1`) and a budget of 1 × 16. The 60 quick-tier frames after
the 30 settle frames are recorded.

| Run | Median | p95 | Worst | Bonds broken |
|---|---|---|---|---|
| ours, run 1 | 16.8 ms | 134.2 ms | 276.7 ms | 2,920 |
| ours, run 2 | 16.6 ms | 120.4 ms | 222.1 ms | 513 |
| vibe-land quick tier | 11.38 ms | 15.32 ms | 35.5 ms | 638 |

**The two identical runs disagree, so the baseline is not deterministic.**
- Per-frame hashes, from `STRESS_GPU_SPLIT_HASH=1` and `STRESS_BENCH_TRACE=1`, locate it.
- After the first cannonball's split (frame 30, the full sync-and-rebuild path) the two runs agree on:
  - rubble's bodies;
  - the contact ingest;
  - the frame digest.
- Frame 31's first static solve then gives different results.
- It is the same signature as the in-place split's race: the first static solve after a split.
- Fixing it comes before any further performance work. A benchmark that does not repeat
  cannot gate anything.

**Also measured:** the `tiered` (impact-island) path on our older bombard ran for more than
40 minutes without finishing 90 frames. As built today, the impact-island path is far from
real time on Vibe Town.
