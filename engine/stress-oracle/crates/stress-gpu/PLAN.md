# stress-gpu plan

The goal is a GPU stress solver with the same model and features as `stress-ref`. It must match the reference's accuracy, run in real time at massive scale, and use f32 throughout.

The kernels are written in Slang. They run on Metal natively and on WebGPU through WGSL.

CUDA is a target through PhysX GPU: a GPU physics scene pairs with the GPU stress solver, and a CPU scene with the CPU solver. The integration point is the one the previous GPU stress solver used. `Nv::Blast::ExtStressGpuSolver` (`blast/include/extensions/stressgpu/NvBlastExtStressGpu.h`) is created by `PxgDestructionRuntime` and advanced from `PxgSimulationController::advanceDestruction`. The same Slang kernels compile to CUDA there. They share PhysX's CUDA context and streams, read contact buffers in place, and commit fractures with device kernels. The scene does not use the direct GPU API (`PxSceneFlag::eENABLE_DIRECT_GPU_API` stays off), and sleeping stays enabled. No CUDA machine is reachable from this checkout, so that backend is validated on one later. The Metal/WebGPU host here defines the kernels and their accuracy gates.

The design follows the colleague's speed-of-light analysis ("GPU Stress Solver: Speed-of-Light Analysis v2"). It is adapted where Metal and WebGPU differ from an RTX 4090. Each adaptation is listed under "Departures from the analysis" with its reason.

## Accuracy contract

| Gate | Check | Pass |
| --- | --- | --- |
| P1 kernel parity | Every GPU kernel against the f64 reference function, from identical random inputs (intact, cracked, crushed, sliding, rebar, buckling) | Within a bound derived from f32 rounding of that function's operations |
| P2 trajectory | Same scene, same substep, same substep count: GPU against `ReferenceSolver`/`World` in f64 | State within the f32 propagation bound; identical events (kind, bond) |
| P3 oracle | `stress-ref compare` on the GPU's `stress-observation/1`, against analytic values and goldens | Every gate the reference passes |
| P4 determinism | Repeated runs on the same device | Bit-identical state and events. There are no float atomics. |
| P5 behaviour | The showcase assertions | All pass |

**Two step modes:**

- **Reference step:** the reference's own `substep_dt`. Use it for P2 and P3 parity.
- **True step (real time):** each island's step is set from λ_max of M⁻¹K, its damping ratio and a safety factor of 0.9. This is a step change, judged by the reference's own refinement at the GPU's step: the reference can run at any dt, so the comparison stays like for like.

## Architecture on Metal and WebGPU

- **Island kernel.** One threadgroup per island, where an island is a cluster or a contact-coupled group of clusters. It runs every substep of a segment inside one dispatch, with threadgroup barriers between phases. That removes the 1.5–3.5 µs a dispatch costs per substep. The reference's per-substep cluster reductions become threadgroup reductions: net load, rigid acceleration, drift removal.
- **One shader, one binding layout.** `shaders/world.slang` holds every kernel with shared bindings and parameters. Islands that touch nothing run a whole segment per dispatch. Islands in the contact pipeline, and impactors, advance one substep per round of three dispatches:
  1. `contact_forces`: chunk pairs, impactor candidates and travel checks, one thread each;
  2. `island_frame`: the island kernel, which gathers impactor, ground and pair contact loads inline;
  3. `impactor_integrate`.

  Two crush passes come first, only when an impactor has a crush cap.
- **Substep phases:**
  1. Bonds: evaluate the joint law and write the 9-float body-frame loads.
  2. Chunks: fixed-order CSR gather, external and frame loads, then the central-difference update by support type.
  3. Cluster: rigid integration and drift removal.
- **Deterministic.** There are no float atomics. Sum orders are fixed: CSR in bond order, and reductions on a fixed tree.
- **Hot and cold bonds.** An intact bond below the utilization threshold takes the linear path, which is exact when nothing can change. Damaged bonds and bonds near the threshold take the full joint law.
- **Segments.** The host advances in segments: frame ends, sample times, scripted-event times, and early stops when a bond disconnects (a split is due). Loads that vary over time come from per-substep tables (scripted loads, blasts) evaluated on the GPU.
- **Topology on the host first.** The host keeps a `ReferenceSolver` mirror of clusters, bonds and structures. Rare operations reuse the oracle's exact code on the mirror, after the GPU state is downloaded into it: `split_cluster`, removals, refinement, static solves. They move to the GPU (union-find, segmented reductions) once they are correct.
- **Big islands.** Metal has no cooperative launch and no grid-wide barrier. So a big island runs in one large threadgroup with device-memory state, or with a dispatch per substep, whichever measures faster. Cross-threadgroup spin barriers are a measured experiment only, because forward progress is not guaranteed.
- **Quiet islands are skipped** behind certificates, not timers: settled with unchanged inputs (class A), closed-form fatigue (class B) and energy bounds (class C).

## Milestones

Status as of 2026-10-10 (Apple M3 Max). Every accuracy gate runs in `cargo test --release -p stress-gpu`.

| # | Milestone | Gate | Status |
| --- | --- | --- | --- |
| G0 | Toolchain, intact explicit substep, native viewer | Displacement error under 1e-5 relative to f64 over 4,000 substeps | Done |
| G1 | Island kernel: all substeps of a segment in one dispatch; big islands over many threadgroups | P2 unchanged; substeps per second measured on growing scenes | Done: narrow (one group, whole segment) and wide (a round of 5 dispatches per substep) kernels; `stress-gpu-bench --full` |
| G2 | Full joint law (damage, crushing, contact patch, friction, rebar, DIF, fatigue, Weibull, buckling); events | P1 on random states; P2 on fracture without splits | Done: `joint_parity` (0 discrete mismatches in 20,000 cases) |
| G3 | Free clusters (rigid motion, frame loads, drift removal); splits through the host mirror | P2 on splitting scenes | Done: `solver_vs_reference`; halts that do not split resume on the GPU |
| G4 | World loads, supports, removal events, replacement loads, probes, observation output | P3 on every benchmark without contact | Done: `world_vs_reference` |
| G5 | Contacts: impactors, ground, chunk pairs; convex hull chunks | P3 on the impact benchmarks and showcases | Penalty contact and hulls done (`world_vs_reference`, `packs_vs_reference`); layer contact open |
| G6 | Solve modes: quasi-static, implicit and adaptive | P3 in every mode | Quasi-static and adaptive done, with the static solves on the GPU (`modes_vs_reference`, `statics_vs_reference`); implicit open; multi-group and multigrid statics for big islands open |
| G7 | True step; sleeping | Step-change bar; idle scenes nearly free | The true step is a scene setting (`--true-step`; `stress-gpu-stepcheck`). Adaptive sleeping with GPU wake checks and settled fatigue: 512 idle structures take 13 ms per frame. Certificates (classes A-C) open |
| G8 | Scale: scene packs, towns, GPU topology | Real time at measured bounds; gates still pass | All packs run. A fully active 24k-chunk island costs 139 us per substep at the true step; GPU connectivity open |

The accuracy gate (`tests/common`) holds each probe within 1e-3 of its range, or within twice the reference's own spread. That spread is the larger of timestep halving and +-1e-4 input perturbations, the size of the GPU's f32 drift. Broken bonds and fragments must be within 10%, or within twice the spread's deviation.

## Test tiers

Every reference run is cached under `target/stress-gate-cache` (keyed by the scene and stress-ref's source fingerprint); a tier costs only its GPU runs. Run from `destruction/stress-oracle`.

| tier | command | what it covers | wall time (M3 Max, shared GPU, warm) | when |
|---|---|---|---|---|
| micro | `stress-gpu-gate --micro` | 25 tiny scenes from t = 0 through the production path (`GpuWorld`, `GpuSolver`, `GpuCoupledSolver`), each for one semantic; knob runs; GPU snapshots; fuzz regressions | ~12 s | after every edit |
| quick | `stress-gpu-gate --quick` | 13 representative gate cases at full duration | ~35 s | while iterating |
| full | `stress-gpu-gate` | 47 gate cases (catalogue, solve modes, packs) | ~5 min | before a commit |
| fuzz | `stress-gpu-gate --fuzz N [--seed S]` | N seeded scenes (grid, material, supports, impactor, solve mode) | ~55 s for 30 | after kernel or scheduling changes |

Missing references: `stress-gpu-gate --build --micro` or `--build --fuzz N` (CPU only, seconds). `cargo test --release -p stress-gpu` runs the micro tier too (`tests/micro.rs`; cases without a reference are skipped with a message).

**What a micro case checks** (`src/micro.rs`, `micro_cases`), beside the gate:
- events as sorted (kind, bond) multisets equal to the reference's, times within one substep or twice the reference ensemble's scatter;
- the fragment partition;
- chunk velocities (momentum exchange) for collision cases;
- energy closure `E0 + W_ext = mechanical + dissipated` where contacts hold no energy at the end;
- path coverage: the bits the case exists for (`src/coverage.rs`, `COV_*` in `shaders/common.slang`) must have been set by the GPU or the host, so a case cannot pass without running its path. `--micro` prints the matrix of paths by case. `SETTLED_WIDE` is uncovered by construction: `GpuSolver::build` never makes a settled island wide, so `wide_wake` and the settled branches of `wide_chunks`/`wide_rigid` cannot run.

The micro references perturb the scripted loads as well as gravity and velocities (`micro::perturbed`), so load-driven scenes have an ensemble too.

**Knob runs (metamorphic, no reference).** `STRESS_GPU_BATCH_ROUNDS`, `STRESS_GPU_SYNC_EACH_STEP`, `STRESS_GPU_PLAN_STEPS` and `STRESS_GPU_WIDE_CHUNKS` change only scheduling (`micro::Knobs`). Batch boundaries and sync cadence must reproduce the case's own run bit for bit (state hash, probes, events). Only the f32 per-sync energy sums may move, within 1e-4 relative. Wide islands and plan horizons re-associate f32 sums: probes within 1e-5 of their range, equal counts and (kind, bond) events, times within one substep. A scene whose own reference scatters beyond that is judged by the gate instead. Fuzz scenes run batch=3 (bitwise) and wide=2 (gate).

**Snapshots.** Every micro and fuzz run's outcome is recorded per case, knobs and GPU adapter under `gpu-snapshots/`: probes, values, events, state hash and coverage, as bits. The GPU is deterministic, so `--micro` reports any change. An intended change (a kernel, a schedule) is accepted with `stress-gpu-gate --accept-snapshots`, which runs the quick gate in-process first and writes nothing unless it passes.

**Adding a micro case.** Build the scene in `micro_cases` from the stress-ref builders, a few chunks and a duration just past the event. Name its semantic, its `Expect` (coverage bits, minimum events, velocity and energy checks) and the knob runs that force its rare paths. Configuration that is not a scene field (`active_time`, `wake_threshold`) goes in a `Tweak`, applied to both solvers. Then run `--build --micro` and `--micro --explain --only NAME` (events, coverage, differences) to check that the reference shows the semantic. A failing fuzz seed is printed ready to paste into `gate::FUZZ_REGRESSIONS`.

## Departures from the analysis

- **No cooperative launch or grid barrier on Metal or WebGPU.** That rules out the analysis's multi-SM neighbour-flag partitions as a correctness-safe design. Big islands are measured on single-threadgroup and dispatch-per-substep designs first.
- **Threadgroup memory is 32 KB on Apple GPUs, and 16 KB by default on WebGPU.** On-chip state is reserved for small islands. Larger ones rely on the device cache, which is large on Apple silicon.
- **Fast math is disabled in the parity builds** (Metal compiles with fast math by default), and the production build is measured against them.
- **The reference step stays available** so every parity gate compares like for like. The true step is a separate, validated mode.
