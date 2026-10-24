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
| `pair_point` with one sample and one penetration call site | GPU | Rammed high-rise 57.0 → 55.4 µs per substep, house_ram 41.6 → 41.2; bit-identical. The same rewrite of `impactor_contact` was **rejected**: the default Metal build compiles it wrongly (the catalogue fails; b6_spall breaks 0 of 288 bonds), while the no-contraction build is correct. See `findings/substep-latency.md` |
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
| **Load path** (profile `load_path`, the fracture world; `STRESS_FRACTURE_LOAD=load_path`, `STRESS_LOAD_PATH_SWEEPS`) | GPU | Warm-started block Jacobi with Chebyshev's limit recurrence, 16 sweeps a frame: house-1story idle 2.2 ms median, nothing breaks; a 250 kg weight on a cantilever plate's tip breaks its root in frame 3, as the explicit reference does (threshold: nothing) | New option, your decision to adopt. Physical bonds (`blast_model` off). Slow modes converge over frames (about 190 frames of 16 sweeps to 1e-3 on a 20-chunk cantilever) |
| **Local impact** (profile `impact_local`, the fracture world; `STRESS_FRACTURE_STRIKE=analytic`, `STRESS_IMPACT_FRACTURE_WORK`) | GPU | Strikes in 2–3 ms frames; the ball holed or stopped as the explicit reference (threshold lets it pass at full speed through a shattered wall); exit speed 25 against the reference's 50 m/s on a 0.2 m wall at 250 m/s, plug in one piece against 18 | New option, your decision to adopt. Open: the plug's breakup |
| **In-step contact fracture** (`Profile::contact_fracture`, on in `load_path`; `STRESS_CONTACT_FRACTURE`, `STRESS_CONTACT_SWEEPS`) | GPU | Within rubble's step, where its broad phase ends: strikes from the step's candidate pairs, the strike's load solved by the load path (256 sweeps), bonds broken by their strength, the structure split before the narrow phase. Agrees with the explicit reference on through/stopped at 15, 120 and 250 m/s; not at 40 m/s; over-breaks 4–40 times | New option, your decision to adopt. Open: the strike's load model (rigid-structure Hertz peak, quasi-static) |
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
   the two-dispatch pipeline is done.
   - **Measured (2026-10-18), rammed high-rise (one wide island and one debris
     island):**
     - `contact_forces` 26 µs, of which 6 chunk pairs take about 12 µs and 4 impactor
       candidates about 10 µs. That cost sits outside their sample loops: re-running a
       loop adds about 1 µs.
     - `islands_and_wide_chunks` 30 µs, set by the one-chunk debris island's
       whole-substep path; the wide island alone takes 9–13 µs.
     - `pair_point` with one call site per helper: −3 %, bit-identical (adopted).
   - **Next:** an Xcode GPU capture (per-instruction stalls) for the pair and candidate
     setup, and the debris island's substep path. A restructured kernel must be
     bit-identical in the default build: `STRESS_GPU_METAL_CONTRACT=off` only separates
     fused rounding from semantic change, as the `impactor_contact` miscompile showed.
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
    - **Your decision (2026-10-17): floor-certify converged repeats.**
      - An island's converged solve whose commit grew no damage is certified converged
        without a solve, when its loads repeat bit for bit and its state is untouched.
        Its commit still runs, and afterwards it can be a fixed point.
        `STRESS_QS_CERTIFY=0` turns this off.
      - Matched Vibe Town idle: **55–60 → 9.7 ms median**, with 263 of 284 clusters at
        fixed points. Broken bonds are unchanged (493) and here even the digest is
        unchanged.
    - **Blast's per-tick behaviour as configuration** (`SolverConfig`, off by default):
      - `static_unconverged` (`STRESS_QS_UNCONVERGED=keep|commit`): keep an unconverged
        iterate (Blast's warm start), or keep it and commit its damage (Blast's
        `ALLOW_UNCONVERGED`);
      - `static_passes` (`STRESS_QS_PASSES=2`): at most two evaluations a frame.

      Used with `STRESS_QS_BUDGET=1,32` and `STRESS_GPU_STATICS_TOL=1e-3`.
    - **Vibe Town bombard** (`vibe-town_bombard`): vibe-land's town-bombard as close as
      the pack allows. Cannonballs (10,650 kg at 60 m/s) and meteors (2 m at 140 m/s)
      are fired at four houses and the cinema, one every 10 frames.
      - Matched configuration with Blast's per-tick behaviour, after the warm-up:
        **median 21–37 ms, split frames 130–225 ms**. 3,157 bonds break, against
        Blast's 638.
      - Blast's numbers for comparison (its whole server tick): median 11.4 ms, p95 15.3,
        worst 35.5.
      - Split frames were 1.3–2.5 s until a quadratic contact-material lookup in `build`
        became a hash map (lossless).
      - A split frame is now: the mirror sync (12–31 ms), the rebuild (~50 ms), and the
        new clusters' first solves (~100 ms). The repeat gate now carries over
        rebuilds; this is lossless.
      - Next: splits on the GPU (phase 6e), and the static solve's latency on the
        struck clusters.
    - **Where a bombard frame goes now** (GPU timestamps: `STRESS_GPU_TIMING=1` for the
      cascade, `RUBBLE_GPU_TIMING=1` per phase of rubble's step; wall clock per phase:
      `STRESS_RESIDENT_PHASES=1`):
      - **Barrier fix.** The Metal group barriers fenced texture and imageblock memory
        too. Dropping those (bit-identical) took the static solve from 38 to 20 ms a frame.
      - **The stress cascade: ~5–6 ms a frame.** The static solves are 3.2 ms of GPU time
        (92 islands, 1 × 32 CG) and the commit 0.25 ms.
      - **rubble: ~17 ms.** The fast projectiles' CCD needs 3 substeps of 5–6 ms each.
        A substep is broadphase ~0.55 ms, narrowphase ~0.55 ms, and 10 primal
        iterations of ~0.33 ms each. Those are 8 colours of indirect dispatches, so
        dispatch latency dominates; the dual passes are ~0.01 ms.
      - **Speculative contacts** (`STRESS_RUBBLE_SPECULATIVE=1`, item 14, one step)
        overflow rubble's candidate capacity here. Its contact range is one global offset
        (~2.3 m for a 140 m/s meteor), so it needs per-body margins before it can stand
        in for the CCD substeps.
      - **rubble's iterations** (`STRESS_RUBBLE_ITERATIONS`, measurement only).
        vibe-land's PhysX runs 4 position iterations; rubble runs 10. At 4 the bombard
        median falls 20.9 → 15.8 ms, but broken bonds rise 3,157 → 5,655 and fragments
        887 → 1,668: softer contacts let the impacts go deeper. Not adopted.
      - **Accuracy at a matched iteration budget** (2026-10-17, your request: match
        Blast's configuration, then its accuracy).
        - **Configuration.** Blast's CPU `ExtStressSolver` defaults to
          `maxSolverIterationsPerFrame` = 25; vibe-land's native GPU stage runs 16. Our
          equivalent is a budget of one Newton step (the matched model is linear until a
          bond breaks) × N CG iterations, from rest.
        - **Measure.** Each solver's cold-start answer against its own converged answer,
          stressed with Blast's formula: peak-fibre and shear Pearson, median ratio,
          share within 2×.
        - **Tools.** Ours: `stress-gpu/examples/blast_forces_gpu --blast-inputs
          --blast-model --budget 1,N`, the GPU iterate kept. Blast:
          `blast_ladder_bench forces PACK 3000 N`, first frame against the converged
          one.

        | pack | Blast 16 it | Blast 25 it | ours 1 × 10 | ours 1 × 16 | ours 1 × 25 |
        |---|---|---|---|---|---|
        | house-1story (peak fibre: Pearson / median ratio) | 0.83 / 0.37 | 0.90 / 0.70 | 0.88 / 0.92 | 0.93 / 0.95 | 0.99 / 0.95 |
        | brick-building | 0.94 / 0.39 | 0.96 / 0.73 | 0.90 / 0.71 | 0.94 / 0.85 | 0.97 / 1.02 |

        At the same iteration count our answer is closer to its converged one than
        Blast's is: the block-Jacobi PCG on the elastic network against Blast's CGNR.
        Ours at 10 iterations is about Blast's at 25. So a 1 × 16 budget already exceeds
        Blast's per-frame accuracy, and 1 × 25 is well above it.
        - **The converged answers still differ** (house-1story, `blast_model` and Blast's
          inputs: Pearson 0.14, median ratio 0.44, 41 % within 2×; brick-building agreed
          at 0.98 already with the inputs alone). On the hull packs Blast's CPU answer is
          not in equilibrium: the 2026-10-16 support check bounds its support load at
          0.6–10× the weight. So it is not a reference ours can match there. vibe-land's
          native GPU stage solves the same equations with another scheme; its captured
          forces are the reference to try next.
      - **Blast GPU solver strategies, read from its source** (stressgpu
        `detail/README.md`), mapped to ours:
        - per-component blocks;
        - f32, warm start and a fixed cap with unconverged answers published;
        - a Chebyshev block-Jacobi preconditioner;
        - rigid-mode projection;
        - a settled / idle skip;
        - captured, GPU-owned execution.

        The resident QS frame has each of these, or a finer-grained equivalent: the
        repeat gate, fixed points and certification. The one not adopted is the
        whole-GPU cooperative kernel with a V-cycle for components over 8,192 nodes;
        our largest island is ~1,800 chunks and the solve is 3–4 ms. Its cascade costs
        ~5 ms per bombard frame. What separated us from Blast's tick was rubble.
      - **Look-ahead CCD** (`STRESS_RUBBLE_SPECULATIVE_CCD=1`, your decision: opt-in,
        measured). rubble's `FLAG_SPEC_VELOCITY` contacts with a reach cap
        (`STRESS_RUBBLE_SPEC_REACH`, default 3 m; `WgpuSim::set_spec_reach_max`, host
        word 31) replace the fast projectiles' CCD substeps.
        - Bombard median **17.9 → 9.1 ms**, below Blast's 11.4.
        - The outcome differs: broken bonds 2,650 → 5,071, fragments 865 → 1,668. Each
          impact is resolved in one step instead of 2–3, so it is not adopted.
        - Guard 24/24 with it off.
      - **Bombard frame on a quiet GPU (2026-10-18, `STRESS_RESIDENT_PHASES=1`,
        `RUBBLE_GPU_TIMING=1`): about 15 ms.**
        - 2 rubble substeps of about 5 ms each (CCD for the projectiles). Each is:
          - broadphase 0.85 ms and narrowphase 0.55 ms;
          - 10 primal iterations of about 0.28 ms, each 7 colours plus 4 headroom
            dispatches (about 25 µs per dispatch).
        - The stress cascade, about 3 ms.
        - The CCD substep count's body read-back (0.2–0.4 ms) now rides on the frame's
          one status read (`ResidentEngine::resident_status` returns ranges); the next
          frame's `bodies_now` uses it. Resident tests pass and the guard is 24/24.
        - **Found and contained: in-place splits made the bombard nondeterministic.**
          - Identical runs gave different digests (at 3 s: 3,343 against 6,317 broken
            bonds), on Metal and on WGSL alike.
          - With `STRESS_QS_SPLIT_IN_PLACE=0` the runs repeat exactly, and they differ
            from in-place (1,827 against 1,859 broken bonds at 0.5 s).
          - Tables uploaded by a split hash identically across runs
            (`STRESS_GPU_SPLIT_HASH=1`). The GPU state diverges in the frame after: a race
            that the in-place layout exposes, not yet found.
          - **In-place splits are now opt-in** (`=1`). Bombard over 3 s with them off
            repeats bit for bit: median 14.8–15.0 ms, but rebuild-split spikes give a p95
            of 107–336 ms and a worst of 181–414 ms.
          - Fixing the race is the route to the p95.
        - **The Blast reference is vibe-land's standard tier, not the quick one.** The
          11.4 / 15.3 / 35.5 ms figures come from the quick tier: 2 events, one breaking.
          The standard bombard tier has 5 events, 6,707 broken bonds and 1,569 awake
          bodies: median 11.5, p95 37.9, worst 48.2 ms.
          - vibe-land runs PhysX with no sweep CCD (its destruction stage refuses scene
            CCD; speculative CCD is off), one 60 Hz step, PGS at 4 position / 1 velocity
            iterations, stabilization and damping. Thin walls tunnel at 60 m/s.
          - Audit: side-session note `physx-blast-techniques-audit.md`.
        - Under Unreal Editor's GPU load, the same runs measure 20–42 ms.
        - **The decision that remains is yours:** match vibe-land's PhysX (one-step CCD,
          4 iterations; about 9 ms, but 2× the breakage), or stay lossless. Lossless
          means cheaper rubble substeps, and splits on the GPU (6e) for the spikes.
      - **Splits are the spikes (p95 ~155 ms, now ~55 with splits in place).** Each one downloads the mirror (12–30 ms)
        and rebuilds the tables of all 52k chunks (~50 ms) for a few changed clusters.
        The new fragments' first solves come on top. Next: re-lay out only the split
        islands (phase 6e).
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
