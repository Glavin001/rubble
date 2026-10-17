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

Status as of 2026-10-17 (combined branch `integrate/raw`, PR #32).

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

## Open: planned or proposed

Ordered by expected leverage for real-time destruction. "Decision" means the next step is
your call; the rest need building.

### Fewer or cheaper substeps

1. **Low-latency substep replay.** Record a frame's substeps once and replay them on the
   GPU (an indirect command buffer on Metal, or a persistent kernel). The per-substep
   latency drops from about 24 µs toward 2–5 µs. This is the main lever for a single
   building. Not built.
2. **Intact-only kernel with escalation to the full law.** Measured ×2.5–6 per substep
   (house-1story 24.9 → 9.4 µs). The kernel exists for benchmarks; it isn't in the coupled
   solver.
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
    also the prerequisite for a browser version.
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
