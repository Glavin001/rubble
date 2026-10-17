# Stress solver: improvement tracker

One place to see what has been done to the destruction stress solver, what was tried and
rejected (and why), what is built but switched off, and what is still planned. Details,
numbers and reproduction steps live in the linked documents; this file is the index.

Update it whenever an item is adopted, rejected or added. Each item lists its layer:

- **GPU**: the GPU stress solver (`stress-oracle/crates/stress-gpu`), the production path.
- **Coupling**: rubble as the rigid-body engine (`stress-oracle/crates/stress-rubble`,
  `GpuWorld::with_engine`).
- **Oracle**: the CPU f64 reference solver (`stress-oracle/crates/stress-ref`), which the GPU
  solver is gated against.

Status as of 2026-10-17 (combined branch `integrate/raw`, PR #32), updated the same day by
branch `claude/stress-top3-roi-8a4275` (items 1, 2 and 11; measurements in
[`../stress-oracle/findings/substep-latency.md`](../stress-oracle/findings/substep-latency.md)).

Sources:
- [`stress-fracture.md`](stress-fracture.md): overview and scale results.
- [`stress-fracture-benchmark.md`](stress-fracture-benchmark.md): oracle vs PhysX GPU vs
  rubble.
- [`stress-fracture-recommendations.md`](stress-fracture-recommendations.md): done and
  decisions.
- [`stress-fracture-gpu-resident.md`](stress-fracture-gpu-resident.md): coupling stages.
- [`stress-fracture-blast-ladder.md`](stress-fracture-blast-ladder.md): quasi-static mode
  and the feature ladder.
- [`../stress-oracle/REALTIME_ROADMAP.md`](../stress-oracle/REALTIME_ROADMAP.md),
  [`../stress-oracle/IMPROVEMENTS.md`](../stress-oracle/IMPROVEMENTS.md),
  [`../stress-oracle/DECISIONS.md`](../stress-oracle/DECISIONS.md).
- [`../stress-oracle/findings/`](../stress-oracle/findings/): `multirate.md`,
  `multi-time-step.md`, `avbd-stress.md`, `modal-prewindow.md`, `rewind-scaling.md`,
  `rubble-engine.md`, `stress-literature-review.md`.

## Where the time goes (why this list is ordered as it is)

- An awake building costs its stress substeps times the per-substep latency. A rammed
  house-1story runs about 2,900 substeps per 1/60 s frame at about 44 µs each (three
  dependent kernels). On an M3 Max that is about 150–270 ms a frame on the GPU path.
  Rigid-body physics in rubble is about 1 ms.
- Idle and settled structures are already cheap: idle house 1.5–2.3 ms, idle 64-house town
  2.4 ms steady state.
- Many-house destruction is host-bound. In 256 houses with four rammed, GPU waits are under
  a fifth of the 514 ms frame.
- So the levers are: fewer substeps, cheaper substeps (latency), and moving host work onto
  the GPU.
- **What a substep's latency is made of** (2026-10-17, `findings/substep-latency.md`): a
  dependent instruction costs 1.6–3 ns on M3 Max and a dispatch about 1–2.5 µs. A substep
  is now two dependent dispatches whose kernels execute thousands of instructions on their
  critical path (the joint law alone about 4.7k). Real time for a rammed house at its
  2,200–2,900 substeps per frame would need about 7 µs per substep; lossless kernel work
  reaches about 45 µs.

Frame times (M3 Max, mean; the three-way benchmark):

| Scene | CPU oracle | PhysX GPU | rubble GPU |
|---|---|---|---|
| Idle house | 131 ms | 6.3 ms | 2.3 ms |
| House hit by a car | 41 ms | 41 ms | 38 ms |
| House rammed | 4,706 ms | 166 ms | 162 ms |
| 16 houses, one rammed | — | 458 ms | 247 ms |
| 256 houses, four rammed | — | — | 514 ms |

## Production configuration (what runs today)

`stress_rubble::scenes::realtime` (used by `destruction-view`, `stress_bench --solver
rubble-gpu` and `destruction_video`):

- established methods;
- certified true step;
- contact safety 0.35 and stress safety 0.5;
- adaptive solve mode;
- impact islands routed before the step;
- no frame rewinds (`redo_fractured_frames = false`);
- quiet-cluster exclusion on.

Optional switches are listed under "Built, off by default" below.

## Done: adopted and merged

### Speed, lossless (bit-identical)

| Item | Layer | Effect |
|---|---|---|
| GPU-resident coupling, Stage 1: no `Scene` clones, shared structures, faster `pack`, cached engine shapes and step bounds, buffers rewritten in place | GPU, coupling | Idle 64-house town 210 → 80 ms |
| Quiet clusters left out of the GPU tables (Tier 0) | GPU | 80 → 25 ms; 16 houses 46–69 → 18–22 ms. `STRESS_GPU_QUIET_EXCLUDE=0` turns it off |
| Reused rubble body read-backs; routing bounds without building boxes | Coupling | 25 → 6–8 ms. Idle 64-house town now 2.4 ms per steady frame |
| Reused per-cluster routing data and step bounds for quiet clusters; parallel true steps; segment snapshots without quiet structures | GPU | 256 houses with four rammed 815 → 514 ms |
| Certified damped steps on all cores | GPU | 64-house setup 55 → 15 s |
| Deterministic convex hull order | Oracle | Coupled runs reproduce bit for bit across processes |
| Exact one-substep frame average | GPU | Idle house 43 → 21 ms |
| Skip solves whose inputs repeat bit for bit | GPU | House ram 436 → 42 ms; b5_v20 880 → 372 ms |
| No rebuild without damage | GPU | Lossless |
| Hulls cooked once per distinct shape (`stress-rubble` cache, box fallback when rubble rejects a hull) | Coupling | Faster setup; no silent hull loss |
| Adaptive dispatch in rubble (P0, DECISIONS 8ev): colours dispatched = last count + 4, then a sparse tail | Coupling (engine) | Fewer empty dispatches per step. Supersedes the stress session's colour trim (697e41a), which was dropped in the merge |
| Contact neighbour (skin) list in place of `chunk_pairs` every substep | Oracle | 182/182 identical; pair search 2–6× faster; CPU 1–10 % |
| Oracle runtime series (18 bit-exact commits) | Oracle | 22 bench cells 531.6 → 209.8 s (2.5×) |
| Contact pipeline in two dispatches per substep (`forces_and_bonds`, `islands_sums_impactors`; travel checks at the previous substep's end, stop recorded as a substep) | GPU | house_ram 50.1 → 45.3 µs per substep; bit-identical on every `stress_bench` set (33 scenes, re-plans included). `STRESS_GPU_FUSE=0` restores three |
| The same with settled contact islands (a waking island runs its joint law inline; a law-free kernel variant when none is settled) | GPU | town16_one_ram worst frame 346 → 254 ms; house_ram 46.6 → 44.8 µs per substep |
| Segment-mode bonds over 512 threads (`island_frame_bonds_wide`; chunks and reductions on the first 256) | GPU | house-1story segment +4.5 %, brick-building +2.7 %; bit-identical |
| Impactors' substep counter written in place (no blocking read-back per batch) | GPU | Fracture frames' host time; A/B under load: house_ram worst 854 → 746 ms |
| Static-solve loads uploaded contiguously; the scratch cleared on the GPU with only its non-zero regions written; quiet clusters left out in the tiered quasi-static mode (`static_stable`) | GPU, coupling | Fractured downtown (24k chunks) idle, tiered: 24.0 → 20.1 s per 6 frames (calibrated), median 72 → 34.5 ms (authored); bit-identical (guard 24/24 with the quasi-static runs) |

### Accuracy and model

| Item | Layer | Effect |
|---|---|---|
| Impact islands routed before the step (GPU coupled path) | GPU, coupling | Exit-speed error 64–69 % → 3 % (ladder). Fixes punch-through |
| Routing before the step, verify-and-redo (oracle) / promotion (rubble) | Oracle, coupling | One pass to the oracle's outcome; 0 redone frames for 1–8 walls at 40–160 m/s |
| Crack contact | Model | Broken-bond error 1.19 → 0.85 (×2^x), and cheaper (2,801 → 1,819 ms) |
| Softening | Model | Broken-bond error 1.79 → 1.19 |
| Rubble as the stress oracle's rigid-body engine (`stress-rubble`) | Coupling | Matches standalone (35.05 m/s, 6 fragments, 24 bonds); showcases 7/7 |
| Certified true step at safety 0.25 with separate stress and contact safety | GPU | 1.7× fewer substeps, converged outcome |
| Contact pipeline only for engine-owned bodies | Coupling | 460 → 365 ms |
| Grid broadphase keeps static pairs below the querier's index (8352765) | Coupling (engine) | Engine no longer drops contacts above 512 colliders. Exposes the b8 over-breaking listed under Open |
| Oracle method switches made default: `layer_contact`, `exact_frame_transfer`, `contact_crushing`, `tangent_statics`, `patch_after_damage` (DECISIONS 53), `lattice_contact_step`, `implicit_contact_steps`/`_windows`, `corotational_bonds` (DECISIONS 61) | Oracle | See IMPROVEMENTS.md ledger. The GPU honours the established set (see Open: proven methods on the GPU) |

### Tools

- `destruction-view`: live GPU pipeline. Draws true convex hulls since 2026-10-17; `B`
  toggles boxes.
- `stress-viz view`: oracle and rubble, interactive, with `--feature`.
- `destruction_video`.
- `stress_bench` with the fidelity ladder (`--config blast…realtime`).
- Blast head-to-head harness.
- Contact-skin shadow test.

## Built, off by default (measured; waiting on a decision)

| Item | Layer | Measured | Why off |
|---|---|---|---|
| Speculative contacts instead of CCD substeps (`STRESS_RUBBLE_SPECULATIVE=1`) | Coupling | Every outcome within run-to-run spread; free-wall ram 88 → 68 ms; b5 at 30–40 m/s 126 → 116 ms; at least PhysX GPU speed everywhere but the floor drop | Changes how fast contacts resolve. **Recommended to adopt** |
| Multirate integration (`sim.methods.multirate`, AVI bond rates at power-of-two ratios) | Oracle | 34/35 catalogue scenes pass; ×4.0–4.8 fewer bond evaluations on the house packs; rammed-house substep ×2.5 faster | Step change: adoption is a decision. Not on the GPU yet (see Open) |
| Quasi-static solve budget (`STRESS_GPU_STATICS_NEWTON=N`) | GPU | Villa 1.9 → 0.65 s/frame, same outcome | Lossy: house-2story rammed breaks 7 bonds instead of 34. Would need to be an explicit engine-wide setting |
| Stress safety 0.9 instead of 0.5 | GPU | About 45 % fewer substeps | Measured accuracy cost (house ram 2.62 vs 3.04): keep 0.5 |
| `true_step` in the oracle | Oracle | −9 % CPU | s_arch fragments 8 → 14 |
| **Tiered coupling** (`--config tiered`, `GpuWorld::set_tiered`): quasi-static mode with explicit impact islands; only a scene impactor's contact closing faster than `STRESS_TIERED_SPEED` (5 m/s) seeds an island; promotion renews only on fractures under the engine's loads | Coupling | Fractured high-rise 10F (vibe-land, 1,096 chunks): idle median 39 → 4 ms, rammed median 84 → 8 ms, peak 172 → 94 ms. house_ram 1 s: explicit frames until the island ends (0.73 s), then 10–13 ms frames | A model change (Blast's model outside impacts). Island frames keep the explicit cost (140–250 ms on house_ram). The router's impact speed is uncalibrated |
| **Quasi-static impacts** (`--config qs`, your choice 2026-10-17): realtime's features, quasi-static everywhere, impacts included (Blast's model) with budgeted static solves (`STRESS_QS_BUDGET`) | Coupling | house_ram median 7.5 ms, worst 54 ms (4 × 64); fractured high-rise rammed median 6.9 ms, worst after setup 17 ms (2 × 32); town16_one_ram median 20.8 ms; fractured downtown (authored) rammed median 69 ms | No punch-through: the 400 kg ram leaves at 4.91 m/s and breaks no bond (explicit: 3.10 m/s, 188 bonds) |
| **Authored pack strengths** (`scene_pack::ImportOptions::authored_strengths`, `stress_bench --authored`) | Import | The vibe-land packs collapse at rest with the calibrated materials (downtown: 100+ cascade passes per frame, 14 s frames); with the packs' own elastic limits they stand | Like-for-like with vibe-land's Blast; the calibrated import stays the default |
| **Strict step mode** (`GpuWorld::set_min_stress_step`, `STRESS_BENCH_STRICT_STEP=s`) | Coupling | Fails setup when a structure's certified step is below s, naming the chunks that set it: house-1story 7.55 µs (2.5 m timber posts of 16 kg), fractured high-rise 14.1 µs (thin concrete walls) | Your choice over sliver elimination: fix assets at the source |
| Static-solve budget (`SolverConfig::static_budget`, `STRESS_TIERED_BUDGET=newton,cg`), applied after setup | GPU | Tiered house_ram aftermath frames 425 → 13 ms (4 × 64) or 10 ms (2 × 32), broken bonds unchanged (198). Blast-equivalent rung: peak 238 → 43 ms at 4 Newton steps, same outcome | Unconverged solves keep their start state: lossy against converged statics |

## Rejected (tried and measured)

| Item | Layer | Result | Rationale |
|---|---|---|---|
| Blast-equivalent quasi-static configuration as a cheap fallback | GPU | Costs more than real-time on the accuracy set (2,246 vs 1,621 ms, sum of medians); exit-speed error 69 %; 2–3× over-fracture; no punch-through | Not cheaper and wrong wherever impacts matter. The per-frame floor, not the physics, is what separates us from Blast |
| AVBD (backward Euler) as the stress solver | Oracle prototype | b5 at 15–30 m/s becomes clean punch-through at a 25 µs step; b4 overshoot 1.56 vs 1.94; within 1 % needs about 900 steps per period | Numerical damping changes outcomes |
| Block Gauss–Seidel on a Newmark step | Oracle prototype | 9–24 sweeps per step, 8–32× more work on b5 and the free wall; diverges at 100 µs | Nowhere a clear win |
| AVBD hard (augmented-Lagrangian) contact | Oracle prototype | Diverges (1,960 bonds broken) or acts as rigid contact (13.8 vs 35.9 m/s) | Wrong physics |
| Damage frozen during the iteration | Oracle prototype | 40 m/s punch-through becomes a push-over | Wrong outcome |
| Chebyshev acceleration | Oracle prototype | 3–5× on smooth problems; 93 vs 78 sweeps on impact; unstable at 25 µs | No gain for impacts |
| Explicit hot zone with Newmark exterior (Gravouil–Combescure) | Oracle prototype (`mts`) | b5 and the free wall become fully hot within about 1 ms (cap 1.1–1.3×); house at best about 1.3× on CPU, worse on GPU; b5 at 40 m/s fails at every ratio | Wrong where it matters, small gain where it works |
| Hot zone with a cracking exterior; linear Craig–Bampton exterior; textbook GC coupling | Oracle prototype | Diverges on b5; 598–762 vs 154–159 broken bonds; weakly unstable (spectral radius 1.001–1.005) | Wrong or unstable |
| Blast-style correction passes after the step | Coupling | Ram 10.9–15.6 m/s against 35.0; 23–141 fragments against 6; 0.6–60 s per impact frame | Never reaches the right outcome; routing before the step does, in one pass |
| Modal pre-window before impact | Oracle prototype (`modal`) | Accurate (17/17, 6/6), but 2–23× slower than the substeps it replaces; 170 s / 0.74 GB precompute | Replaces only 0.1–2 % of substeps |
| Stiffness or mass scaling | GPU | Large substep cuts | Shifts the impulsive/quasi-static boundary; not recommended |
| Kernel trials: 512-thread island groups, blocked bond-load gather, device fences narrowed to group barriers | GPU | −5 % / +2 % (not bit-identical), 0 %, 0 % | No lossless gain |
| One-threadgroup intact kernel | GPU | 12.4 vs 9.4 µs | Slower |
| Static condensation on partitions | Oracle, GPU | No explicit-step gain (×1.00–1.25) | Revisit after sliver elimination (×5.2–7.7 combined) |
| `frame_contact_step` | Oracle | 7–20× more substeps | Not a lever |
| Pulse windows | Oracle | b3 wave error 0.2 → 7.1 % | Inaccurate |
| Tangent-first solves; fallback without line search; per-evaluate patch caching | Oracle | house_settlement 161 → 214; none faster | Slower or no gain |
| Oracle micro-optimisations: intact-patch skip, multiply-rotate hasher, branchless patch loop, depth table, pooled bond sets | Oracle | +11 %, —, 128 vs 100 ns, +7.7 %, noise | Slower or within noise |
| Colour trim in the coupling (697e41a) | Coupling | Floor drop 4.4 → 2.6 ms per rubble step | Superseded by adaptive dispatch (P0); dropped in the merge |
| Emit-solver-pairs fork experiments (early SolverPairs, workgroup-ordered awake list, lazy partner load) | Coupling (engine) | Within noise | Rejected in engine DECISIONS 8eq |
| **Intact-only kernel with escalation (was item 2)** | GPU | Contact pipeline: the bond dispatch costs 16.0 µs with the full law (no damage evolution) and 16.5 µs with a bare spring; the impactor integration sets it. Segment mode: damage evolution, the only part an intact bond can skip without changing results, is 4 % of the law (36.0 → 34.6 µs) | The ×2.5–6 compared the full law with a kernel that skips required state (rate filter, fatigue, friction offsets, rebar, utilization). No lossless gain where production spends its time |
| Indirect command buffer / persistent-kernel replay (part of item 1) | GPU | Three trivial dependent dispatches cost 7–8 µs of a 45 µs substep. An ICB replays only host encoding (5–6 ms per frame); a persistent kernel needs a grid barrier Metal does not guarantee (no forward progress between threadgroups) | Launch is not the latency; kernel critical paths are. The two-dispatch merge took the launch share that was available |
| Submitting a batch in 512-dispatch slices (overlap encoding with the GPU) | GPU | Peak 273.5 vs 275.8 ms | Within noise |
| Drift moments summed in the chunk update (one chunk pass fewer) | GPU | 44.8 → 45.1 µs per substep | Barriers and dependent math set the island kernel, not loads |
| The waking island's joint law as a `[noinline]` call | GPU | Spill 1,328 → 1,680 B, threadgroup memory 19.5 KB | Worse; replaced by a kernel variant |
| Segment-mode bonds over 1024 threads | GPU | house-1story 24.5k → 21.0k substeps/s | Register spills at 1,024 threads; 512 kept |
| Stress safety 0.9 in the tiered coupling | GPU | Substeps −14 % (129k → 111k); broken bonds 198 → 273, fragments 39 → 60 | Contact sets most island steps: poor trade |

## Open: planned or proposed

Ordered by expected leverage for real-time destruction. "Decision" means the next step is
your call; the rest need building.

### Fewer or cheaper substeps

1. **Substep latency: kernel critical paths.** Replay was the wrong lever (see Rejected);
   the two-dispatch pipeline is done. What remains is the instructions on each kernel's
   critical path: the island kernel (about 23 µs: barriers, rigid frame, travel checks) and
   the forces dispatch (about 14 µs). Needs GPU counters (an Xcode capture) to aim; each
   lossless trial so far gave 0–4 %.
2. *(Rejected, see above: the intact-only kernel.)*
3. **Sliver elimination**, in the assets or as a solver switch. ×1.6–4.5 fewer substeps
   (×4.51 on house-1story). Prove it in the oracle first. With item 2 it brings an awake
   house to about 2–4 ms (roadmap estimate).
4. **Multirate on the GPU.** The GPU design is written (`findings/multirate.md`); not built.
   The cost model gives ×1.03–1.3 on one house, about ×1.7–2 on 100 houses and ×7–13 on a
   huge structure. It multiplies with item 1.
5. **Per-island time step.** ×1.0 on single-island scenes, ×1.09–1.37 on multi-island
   scenes. Not built.
6. **Tighter certified step.** The damping term (largest β × ω_max) gives 3–6× too many
   substeps on houses. Open in the oracle.
7. Per-chunk certified bounds for multirate rates (Gershgorin is about 1.6× loose).
8. Calibrated critical-speed / impact router (P–I maps, SDOF). Research.
9. Implicit contacts only, inside explicit windows: the house's step is set by contact at
   1.3 µs, while the bonds allow 3.5 µs. A lead from the AVBD study.
10. Smoothed-aggregation multigrid, for large quasi-static islands and any GPU implicit
    mode.

### Coupling with rubble

11. **GPU-resident coupling, Stage 3:** routing, contact planning and splits on the GPU,
    with no CPU mirror inside a frame. Expected 3.5–5× on many-house destruction. It is
    also the prerequisite for a browser version. **Scoped 2026-10-17** (house_ram fracture
    frame 1: 234 ms, of which host 60–75 ms: 21 host splits 19 ms, 61 batches' encode
    21 ms and readbacks 13 ms, downloads 10 ms; 41 % of its 3,759 encoded rounds launch
    empty after halts). First stage, in this order:
    1. splits on the GPU from the labels the device connectivity check already computes
       (label propagation): new clusters' chunk lists and mass properties by reductions,
       the host only re-indexing; gate: digests within the GPU's run-to-run spread, as
       `stress-fracture-gpu-resident.md` stage 3 states (f32 mass properties);
    2. halts that need no host work resume in the same submission (indirect dispatch
       sizes zeroed per halted island instead of ending the batch);
    3. contact planning per segment on the GPU (swept boxes through rubble's broadphase).

    **Stages 2–3 under way (2026-10-17)**, phases in `stress-fracture-gpu-resident.md`.
    Phases 1 and 2a are done: shared rubble buffers, the collider table, and the contact
    ingest on the GPU (`coupling.slang`). The ingest matches the host's manifolds to f32
    rounding (`stress-rubble/tests/ingest_gpu.rs`). It is off by default and the guard is
    bit-identical.

    **The resident quasi-static frame** (`GpuWorld::with_resident_engine`,
    `STRESS_RESIDENT=1`) is now complete: phases 2b–5 for the quasi-static coupling.
    - rubble steps without read-backs.
    - The loads are formed on the GPU.
    - The static cascade runs under GPU control.
    - The frame needs one status read.
    - The repeat gate is lossless.

    It gives the fast frame's outcomes; the digests match except for a mirror-only −0.0.
    Vibe Town idle: 35 → **4.2 ms** median.

    For island frames, phase 6a (the contact plan on the GPU) is done and checked against
    the host plan. Phases 6b–6e (tables, snapshots, routing and splits on the GPU) remain;
    they are the impact mode's host overhead, about 20 % of an island frame.
12a. **Tiered router calibration.** The tiered coupling's impact speed (5 m/s) and its
    island membership (any overlap joins a component) are uncalibrated. On house_ram the
    island lasts 0.73 s because fragments touching the ram close at 5–9 m/s. Item 8's
    critical-speed router is the principled version.
12d. **Quasi-static parity with Blast** (your priority, 2026-10-17). PhysX GPU + Blast on
    this M3 Max (vibe-land perf suite, quick, runtime profile): Vibe Town idle median
    10.2 ms, p95 15.3 ms; bombard median 11.4, p95 15.3, worst 35.5 ms. rubble, Vibe Town
    (52k chunks, 281 clusters, `--authored --config qs`, budget 2 × 32): 141 ms per frame
    before this work; 119 ms after the lossless cuts; **34–45 ms with the fast
    quasi-static frame** (`STRESS_QS_FAST=1`: solves and damage commit stay on the GPU, no
    rebuild, download or sync per frame; f32 commit, opt-in). What remains is the static
    solve itself, ~18–22 ms: one threadgroup per island of Newton + PCG, so the
    thousand-chunk terrain and building islands run long serial chains, and only ~82 of
    278 clusters converge within 2 × 32 (warm start, `STRESS_QS_WARM=1`, does not change
    that). Next: a wide (multi-group) static solve or multigrid (item 10), then the
    engine's per-frame `prepare` (2–20 ms). Smaller scenes are already real time in `qs`:
    house_ram median 2.8 ms, p95 7.8; fractured high-rise rammed median 4.4–7.5, p95 11–14.
    **What Blast's GPU solve is, and a matched configuration (2026-10-17).** The full
    study, with citations, is the side-session note `blast-stress-solver-research.md`.
    - **Blast's model** (vibe-land's runtime profile is the native PhysX destruction
      stage running Blast's CUDA kernels via CuMetal):
      - forces: the least-work forces of a linear network of isotropic bond springs,
        weighted `sqrt(E A / L)` with floors A ≥ 1e-4 and L ≥ 0.05; rotational stiffness
        is `k l^2`, one `l` per graph;
      - loads: gravity, centrifugal, and contact impulse / dt at the chunk (no torque);
      - solver: f32 PCG with a Chebyshev block-Jacobi preconditioner, 16 iterations a
        tick, warm-started, tolerance 1e-3; unconverged answers are published and graded
        (the cap binds every Vibe Town tick);
      - stress: force / remaining area, with the bending gain capped at 3;
      - fracture: at the fatal limit outright, erosion at 2/s between the elastic and
        fatal limits.
    - **Matched configuration** (measurement switches, all off by default):
      - `--config blast` (the base material model, quasi-static);
      - `--authored-fatal` (the pack's fatal limits as strengths);
      - `--pack-masses` (the pack's node masses and mass-0 supports);
      - `--method blast_model=on` (Blast's isotropic stiffness and capped section gains).
    - **Each switch's effect on Vibe Town idle, resident:**

      | Configuration | What happens |
      |---|---|
      | elastic limits (`--authored`) | 196 of 278 clusters have no equilibrium (Newton fails at residuals of 0.07 to 2), so they never fracture |
      | fatal limits | 204 of 278 converge |
      | fatal limits, `blast` features | 3,251 bonds break: the town collapses |
      | plus Blast's stiffness | 13,779 bonds break |
      | plus the capped stress gain | 273 of 278 converge, 969 bonds break at the start |
      | plus the pack's masses (the full match) | every cluster converges; the town stands; 493 bonds break as the first frame settles (Blast: none) |

      The remaining gap is the converged answer against Blast's unconverged 16 iterations,
      the L1 against L2 bending norm, and the damage law.
    - **Cost of the matched configuration: median ~55–60 ms.** 129 clusters are at exact
      fixed points and the repeat gate skips them. The other 155 take one Newton step a
      frame: `island_statics` certifies convergence at f32's floor only after a step
      shows the residual has stopped falling, and that step re-rounds the state. Skipping
      it would change results at the rounding level, which is your call. The other lever
      is the static solve's per-iteration latency on the large islands (one threadgroup
      per island).
12c. **Quasi-static frames at scale** (fractured downtown, authored, 34 ms): two full layout builds per frame (~6.5 ms each: pack 1.8, terms 1.5, uploads 2.8), the tail's solves of the 7 fatigue-active buildings, and the write-back; then item 11.
12b. **Static solves in the tiered coupling:** the budget is a per-frame cost bound
    (Blast's choice); a warm-started multigrid (item 10) would converge within it.
12. Stage 2: one submission per frame with shared buffers. About 5 % on destruction
    frames.
13. Use rubble's pipelined `step_submit`, which takes the previous step's counts on the
    GPU. About 1 ms per substep for fast bodies.
14. **Decision:** adopt speculative contacts by default (see "Built, off by default").
15. Port the contact skin list to `stress-gpu`: a sweep-based plan and event-driven
    re-plans instead of `PLAN_STEPS = 4`.

### Fidelity: the GPU solver against the oracle

16. **Decision:** fix contact-crushing pass-through. Once a contact face crushes through
    its first layer it pushes with zero force, and fragments slide through walls (on
    s_house_car, 30–60 pairs overlap by more than 5 mm). It changes oracle results, so
    goldens would be regenerated.
17. Over-breaking in the contact-to-load path: b8_sudden breaks 36 bonds against the
    oracle's 11, exposed by the broadphase fix 8352765.
18. Fast punches: both GPU solvers break far fewer bonds than the oracle on b5 at
    30–40 m/s (91 vs 599, 72 vs 437).
19. Bond dashpots move b5_v20 away from the oracle (513 → 350 bonds).
20. Proven oracle methods on the GPU: layer contact, contact crushing, lattice contact
    step, co-rotational bonds, Rayleigh damping. The GPU has penalty contact, small-strain
    bonds and dashpots.
21. Implicit mode on the GPU (currently rejected), and HHT-α.

### Sleeping and startup

22. **Decision:** let coupled rigid bodies sleep, freezing their loads into the solve.
23. Safe sleeping: settling and energy certificates (shadow first), skipping frames of
    settled structures, closed-form fatigue while asleep, and the 93 ms startup frame.
24. Debris release certificates; the simple test isn't conservative.

### Assets

25. Multi-story packs are over strength at rest. Needs a design check (as in Shardwright)
    or recalibrated materials.
26. Cross-platform determinism of Shardwright bakes (unchecked across OS and architecture).

### Bugs and robustness

- **A travel-check stop races the contact kernels of its substep** (found 2026-10-17).
  When STOP_NOW lands inside `contact_forces`, pairs and candidates already started there
  have advanced their contact history for a substep that is then re-planned and run
  again (crush passes likewise). Deterministic in practice (the pairs win), so digests
  repeat; the two-dispatch pipeline keeps the behaviour for bit-identity. Not running
  the stopped substep's contact work is more correct but changes re-planning scenes:
  a fidelity decision.

27. `destruction-view` panicked on non-convex fragment hulls (house_ram, town). The
    `stress-rubble` box fallback may cover it; unverified.
28. GPU contact-state buffer overflow on collapse (a 4.87 GB crash).
29. rubble grid overflow on house + car without impact islands.
30. The oracle's convex hull isn't robust to f32-coplanar input.
31. Handover energy fails on "pair" (proven) and "split" in the rubble coupling. Also:
    rest-overlap `C0` offsets, contact warm-up sink (1.6 mm), append-only bodies, CCD by
    substepping, gravity along z only, friction fixed at 0.5.

### Platform and tooling

32. Stress solver in the browser (wasm). Nothing built; needs item 11.
33. Ladder configurations (`--config`) in `destruction-view` and `stress-viz view`.
34. CUDA host and RTX 4090 qualification (roadmap phase 8).

## Doc inconsistencies to resolve

- `patch_after_damage` is "awaiting review" in the IMPROVEMENTS.md ledger and roadmap A8,
  but default on in IMPROVEMENTS.md's Open table and DECISIONS 53.
- `corotational_bonds` says "kept off" in its IMPROVEMENTS.md entry; its heading and
  DECISIONS 61 say default on.
- `scaled_step_bound` says "running" in its entry; the protocol lists it as proven and on.
