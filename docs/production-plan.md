# Rubble Production Plan v2 — a GPU-resident AVBD rigid-body engine

**Status:** approved plan of record (2026-07-05, branch `claude/rubble-physics-production-plan-jmj42u`).
This document supersedes the roadmap portions of
[`avbd-competitive-analysis.md`](./avbd-competitive-analysis.md) (its §§3–7 recommendations) and
[`performance-plan.md`](./performance-plan.md) (its milestones); both remain valuable as the underlying
line-by-line evidence. [`TODO.md`](../TODO.md) tracks spec-level feature status and now defers to the
milestones below for sequencing.

## 0. Context

**Task:** re-examine the earlier competitive analysis (branch `claude/happy-fermat-8sb3le`, merged into
this branch), re-verify the current code, research the algorithm landscape, and produce the definitive
plan to make rubble's **rigid-body physics engine production-ready**: bleeding-edge algorithms, fastest
runtimes, scalable to thousands+ of bodies, stable/reliable collisions. First-principles rewrites are
explicitly welcome; every decision below records rationale, trade-offs, benefits, and effort
(S ≤1 day, M 1–3 days, L 1–2 wk, XL 2–4+ wk of focused work).

**Confirmed scope:** native (Rust/wgpu) + web (WebGPU/WASM) equally; v1 = *rigid contacts only, rock
solid* (joints/soft deferred to roadmap); 2D kept but rebuilt on the shared core.

---

## 1. Research findings (what the evidence says)

### 1.1 The AVBD paper is the right algorithm — and rubble deviates from it

[AVBD, SIGGRAPH 2025 (Giles/Diaz/Yuksel; Roblox + Utah)] — read in full for this plan:
- **Performance bar:** 110,000 boxes piled+smashed = **3.5 ms solve / 9.8 ms with collision detection** at 4 iterations; **510,000 bodies = 10.3/17.6 ms** at 3–4 iterations (RTX 4090, dt=1/60, fp32). Sequential impulse/XPBD/VBD are 4–50× slower or collapse in the same scenes.
- **No substepping.** Single implicit step, few iterations, warm-started duals. (PhysX-5 TGS and Box2D-v3 "TGS Soft" won the *impulse-solver* comparison via substeps — Catto's Solver2D — but that family is sequential per-constraint and needs constraint-level coloring on GPU; VBD/AVBD color *bodies* (far fewer colors) and parallelize per-body, which is why it holds 500k bodies. For a GPU-first engine AVBD is the correct family.)
- Key machinery, in paper order: augmented Lagrangian per constraint (λ update `λ ← clamp(kC+λ, λmin, λmax)`); stiffness ramp `k ← k + β|C|` **only while force is inside bounds**; α-regularization `C(x) = C*(x) − α·C*(x_prev)` (α=0.95) to stop explosive error correction; warm-start `λ←αγλ, k←max(k_start, γk)` (γ=0.99); **Taylor-cached constraints** — freeze `C0` and Jacobians at step start, drop the 2nd-order term for contacts (cheap iterations); SPD quasi-Newton Hessian via diagonal column-norm approximation of the geometric term (matters for joints; contacts drop it anyway); **LDLᵀ** solve; friction = 3-row constraint with lagged cone `‖λ_t‖ ≤ μλ_n`, **static/dynamic μ switch**, **tangential anchor freeze** when sticking, and **stiffness rescaling (Eq. 14)** which alone took sliding friction from 4 iterations to 1; incremental Jacobi-greedy **body** coloring where color conflicts are tolerated (double-buffered positions degrade conflicting bodies to Jacobi safely).
- Stated limitations we design around: force propagation is bounded by iteration count (long chains take frames); BDF1 dissipation; **collision detection costs more than the solve** at scale (6.3–7.2 ms of the totals above).

**Rubble today** (verified line-by-line): the P0 stability core from the previous analysis genuinely landed (α-reg with frozen C0 + recovery-rate clamp, mass-scaled dual cap, trust region 0.35R/0.5rad, lagged cone — `crates/rubble3d/src/gpu/avbd_solve_wgsl.rs`), but vs the paper it still: runs α=0.3 (paper 0.95 — low α is compensating for weak convergence), has **no m/dt² penalty floor** (flat k_start=1e4), **no split normal/tangential k caps** (single 1e6 vs refs' 1e10/1e6), uses **Gaussian elimination that returns zero on a small pivot** (drops the whole body update; paper/refs use preconditioned LDLᵀ with pivot clamping), re-derives constraint geometry from live anchors every iteration (paper caches Taylor J/C0 — cheaper and calmer), has no stiffness rescaling, no static/dynamic μ, no velocity clamp, and no substep option. The 3D pyramid energy injection and 2D resting drift known-failures are the visible symptoms.

### 1.2 Both reference repos agree on the execution model rubble lacks

- **avbd-metal** (Swift+Metal; near-faithful GPU port + extensions): zero blocking readbacks (async stats only, 2 frames in flight), indirect dispatch everywhere, GPU CSR adjacency (count→scan→scatter), **persistent incremental GPU coloring with compacted per-color lists + GPU-written dispatch args**, CAS open-addressing pair map for O(1) warm-start (capacity 2·maxPairs, linear probe), m/dt² penalty floor at narrowphase, speculative margins (`approach·dt`, capped), **encoder split at stage boundaries measured 2.5–3×**, single-workgroup persistent solve kernel for small scenes, 8-lane SIMD-split primal for high-degree bodies, mass-scaled λ cap `min(1e6, max(10, 1e5·minMass))`, velocity clamp 100 m/s. Its multi-workgroup persistent kernel **deadlocked** (no forward-progress guarantee) — a hard warning we heed: WGSL has no device-scope barrier, so only the single-workgroup variant is portable.
- **webphysics** (TS+WebGPU): proves the whole model works in the browser. Adds: **AABB fattening = 0.01 margin + |v|·(1/60) velocity horizon**; SAT **hysteresis** (previous-axis preference, edge-win margin, keep/stale bands — all relative to slop=0.005); 9-bit structured feature keys; 4-slot per-pair manifolds with **tangent-basis rotation of carried λ + cone re-clamp**; double-buffered **async LBVH rebuild** (traverse last completed tree); fixed-timestep accumulator with catch-up clamp; +1e-4 diagonal regularization; LDLᵀ pivot floor 1e-6. Its main weaknesses (our openings): per-color dispatch over **all** bodies (no compaction — the scalability killer), per-iteration world-inertia refresh, top-K pair pruning that silently drops contacts, sleeping removed, box+sphere only, no tests, Chrome-only.
- Neither has: restitution, sleeping/islands, CCD beyond speculation, 2D, native+web portability, real test suites, GPU queries, collision filtering (webphysics has groups/masks; avbd-metal doesn't), determinism.

### 1.3 Landscape check (is anything newer/better?)

- **VBD** (SIGGRAPH 2024) is the base method; AVBD is its constrained/rigid completion. No 2026 successor supersedes it for massively-parallel game-style rigids.
- **NVIDIA Newton 1.0** (Mar 2026, Linux Foundation; NVIDIA+DeepMind+Disney): rigid solvers are MuJoCo-Warp and **Kamino** (NCP/optimization, kinematic loops) — robotics-accuracy tools that scale via *thousands of independent environments*, not one coupled 100k-body pile; **uses VBD for deformables**. Validates the VBD family; not a better fit for our target.
- **OGC** (SIGGRAPH 2025): penetration-free contact for codimensional objects (cloth). avbd-metal uses it for cloth only and had to disable its barrier for rigid contact. Roadmap item for soft bodies, not v1 rigid.
- **PhysX 5 GPU / Box2D v3**: TGS substep solvers; excellent CPU/console engines; their GPU paths are constraint-partitioned Gauss-Seidel — heavier machinery for the same outcome AVBD reaches with body-coloring. No reason to switch families.
- **Broadphase SOTA**: our Karras LBVH (GPU Morton+radix+build+refit+traverse) is the paper's own choice. **H-PLOC (HPG 2024)** builds better trees in a single kernel 1.1–3.6× faster — noted as a v1.x upgrade, not a blocker.
- **Subgroups**: shipped in WebGPU spec (CR Jan 2025) + Chrome; wgpu supports natively. Use with feature-detection + scalar fallback.
- **GPU determinism**: same-adapter run-to-run determinism is attainable (no float atomics; sort atomically-appended lists before consumption; index-based tie-breaks). Cross-platform bit-determinism is not realistic (FMA/driver variance) — declare it a non-goal, offer the CPU mirror solver (D12) for bit-stable needs.

### 1.4 Current-code audit verdict (what the rewrite must fix)

Hot path per `World::step()` today (`crates/rubble3d/src/lib.rs:903`, `gpu/mod.rs:1392/1429`): full CPU re-compaction and **re-upload of all bodies/props/shapes every frame**; **~4 blocking counter readbacks + 2 full contact-buffer downloads + 1 full body-state download every frame**; CPU adjacency rebuild + re-upload; GPU Luby coloring with **33 submits** (fixed 32 rounds, no early-out) and a CPU sort on readback; **one queue submit per dispatch** (`run_pass`, mod.rs:2873) except the batched solve loop; O(new×prev) warm-start scan; O(bodies×contacts)/round coloring; compounds solved by a **CPU sphere-cloud approximation** (falls through floors — known failure); constant feature IDs for capsule/hull/sphere pairs (breaks warm-start); no AABB fattening; queries/events on CPU; 64 MB+ contact buffers at default `max_bodies` regardless of scene; `rubble-shaders` (rust-gpu) is **dead code** (not a workspace member, build.rs commented out, zero callers) that pins nightly; multi-GPU crate is standalone and unwired; 2D is a divergent parallel copy.

**Worth keeping** (genuinely good): the test culture (parry oracle, invariant harness, honest known-failure registry, heavy/fault-injection lanes, CI on software Vulkan + SwiftShader-WebGPU), broad 3D+2D shape matrix with manifold reduction, the fully-GPU LBVH that already exists, free-motion fast path, generational-handle World API shape, wgpu portability, and the landed P0 stability math.

---

## 2. Architecture decisions (ADRs)

**D1. Keep AVBD; implement it completely and faithfully, then extend.**
Rationale: SOTA evidence (§1.1), both references, and rubble's own partial success. The instability is implementation gaps, not the algorithm.
Trade-offs: AVBD is inelastic by construction (restitution needs an explicit pass — D10) and propagation is iteration-bounded (long chains; joints later will want more iterations or substeps).
Effort: the fidelity work is M–L (D6). Benefit: paper-level stability at 4–10 iterations instead of 12–20.

**D2. Rewrite the execution model to GPU-resident, zero hot-path readbacks. This is the centerpiece.**
The step becomes: one (or few) command encoder(s) recording predict → broadphase → narrowphase → warm-start → adjacency → coloring → N×(primal per color + dual) → restitution/velocity finalize → sleep/events bookkeeping, **all sized by GPU-written indirect dispatch args**. Body/contact/shape state lives on the GPU across frames. User mutations (add/remove/teleport/impulses) go into a **mutation queue buffer** applied by a small kernel at step start; render/query state returns via an **async snapshot ring** (2–3 frames deep, never blocking), plus an optional blocking `sync()` for tests.
Rationale: this is the single property both references share and the paper assumes; rubble's frame time is readback/submit-bound long before ALU matters.
Trade-offs: async snapshots mean render transforms lag ≤1 frame (industry-standard; imperceptible); API becomes "command + query" instead of direct mutable state (D11); big one-time engineering cost; determinism needs explicit care (D13).
Effort: **XL** (the bulk of the plan). Benefit: structural parity with the references; CPU cost per step collapses to command recording (~O(iterations+stages) dispatches, not O(bodies)); unlocks 10k–100k+ bodies.

**D3. Contact pipeline on GPU end-to-end.**
- **CSR adjacency** on GPU (clear→count→scan→scatter; avbd-metal `adj_*`). Replaces the per-frame CPU build.
- **Persistent incremental body coloring** on GPU (seed with last frame's colors; greedy vs smaller-index neighbors; fixed ≤8–20 passes with a `changed` early-out flag; cap `MAX_COLORS=64`, overflow color = Jacobi-tolerated), then **compacted per-color body lists** via count→scan→scatter writing **per-color indirect dispatch args** (avbd-metal's `color_count/scan/scatter`). Explicitly reject webphysics's colors×all-bodies dispatch.
- **Primal color-safety**: within a color, solve in place (Gauss-Seidel across colors — better convergence, what avbd-metal does); conflicts only exist in the overflow color, which we treat as Jacobi (paper's double-buffer argument makes this safe).
- **Warm-start via CAS open-addressing pair map** keyed `(bodyA,bodyB)` → previous manifold slot (capacity nextPow2(2·maxPairs), linear probing; avbd-metal `pm_insert/pairMapFind`), then per-point matching *within* the pair: exact structured feature key for polytopes, proximity for round shapes, with **tangent-basis rotation of carried λ + cone re-clamp** (webphysics). Kills the O(new×prev) scan.
- **Split the contact record**: geometry (point/normal/anchors/feature/C0 — written by narrowphase each frame) vs solver state (λ/k/flags — persisted across frames via the pair map). Removes any reason to round-trip contacts.
Effort: **L–XL** combined. Benefit: contact costs become O(contacts) with small constants; no readbacks; warm-start correctness improves (feature IDs + basis rotation) which directly lifts stacking stability.

**D4. Broadphase: keep GPU LBVH, add the missing safety features; compounds become first-class leaves.**
- Add **AABB fattening = margin (0.01) + |v|·dt horizon** and a **speculative narrowphase margin** `min(approach·dt, cap)` (webphysics + avbd-metal) — fixes resting-pair buzzing and cheap anti-tunneling.
- **Compound shapes: expand children into broadphase leaves** (leaf = (body, child) pair), so narrowphase runs native child-shape tests. Deletes the CPU sphere-approximation path and its known failure. This also removes the last consumer of the hybrid CPU-Karras path — delete it.
- Never silently drop pairs (webphysics's top-K pruning and avbd-metal's saturation are documented failure modes): overflow sets a GPU flag → async detect → grow + log; size buffers from live body count, not `max_bodies`.
- v1.x upgrades (not v1): H-PLOC single-kernel build; refit-only frames between rebuilds; async double-buffered build (webphysics) if profiling shows build on the critical path.
Effort: **M** (fattening/speculative) + **L** (compound leaves) + deletion is S. Benefit: correctness (compounds), stability (pair persistence), tunneling resistance.

**D5. Narrowphase: stable feature IDs everywhere + SAT hysteresis.**
Extend the box-box structured key scheme (`axis_type|ref_axis|inc_axis|ordinal`) to capsule/hull/sphere pairs (per-point ordinals; today they share one constant ID per pair — warm-start collisions, tracked failure). Add webphysics's hysteresis: previous-axis preference within tolerance, edge-win margin, keep/stale bands (all slop-relative). Fix the remaining pair-matrix known-failures against the parry oracle (capsule/hull/compound misses). Wire `penetration_slop`/`contact_offset` params (today hardcoded `CONTACT_MARGIN=0.01`).
Effort: **L**. Benefit: manifold stability under rotation = the difference between "stack settles" and "stack buzzes/topples"; closes 3 known failures.

**D6. Solver-core fidelity: finish the paper (can land before the D2 rewrite, on the current architecture).**
(a) **Preconditioned LDLᵀ 6×6** with Jacobi diagonal preconditioning and pivot clamp 1e-6 (+1e-4 diagonal regularization), replacing return-zero Gaussian elimination — port avbd-metal `solve6x6`. (b) **Penalty floor `k0 = clamp(min_mass/dt², k_start, k_max)`** at contact init, **split caps** normal 1e8–1e10 / tangential 1e6, warm-start `k ← max(k_start·floor, γk)`. (c) **Taylor-cached constraints**: freeze `C0` and Jacobians at step start (narrowphase already computes them); primal iterations become dot products (paper's own formulation; big inner-loop win; trust region bounds the linearization error). (d) **Friction quality**: stiffness rescaling (Eq. 14) for the clamped-force Hessian; static/dynamic μ pair with the cone test switching them; keep the existing sticking-anchor carry. (e) **Velocity clamp** at finalize (`maxSpeed ≈ 0.5·cell/dt` or 100 m/s default). (f) Raise **α to 0.95** once (a)–(d) restore convergence; expose α/β/γ in `SimConfig` with paper defaults (β=1e4-scale matching both refs' constraint scaling, γ=0.99–0.999). (g) Optional **substep loop** (`substeps` in config, broadphase once per frame, narrowphase+solve per substep — webphysics structure) for high-energy scenes; default 1.
Acceptance: the pyramid-stack and 2D resting-rect known-failures flip to passing; default iterations drop 12 → 8.
Effort: **L** total (each item S–M). Benefit: this is the stability core; also −30–50% solve ALU from (c).

**D7. Scale features neither reference has: GPU sleeping + islands.**
Per-body low-motion counters on GPU; island labels via iterative label-propagation over the contact graph (a few pointer-jumping passes piggybacking on adjacency); an island sleeps when all members are below thresholds for N frames; sleeping bodies are excluded from the **compacted active-body list** (stream compaction — the free-motion pass already points this direction) so predict/solve/narrowphase cost tracks *awake* bodies only; woken by new pair vs an awake body, or user mutation.
Rationale: settled piles are the common production case; both references pay full price forever (webphysics deleted sleep "while we improve performance").
Trade-offs: wake latency 1 frame; label propagation adds ~2 small passes/frame.
Effort: **L**. Benefit: order-of-magnitude wins on settled scenes; also reduces jitter (sleeping bodies are pinned).

**D8. Small-scene latency: single-workgroup persistent solve kernel.**
Port avbd-metal's `solve_persistent`: entire iterations×colors×dual loop in one workgroup with `workgroupBarrier()`, selected when active bodies ≤ workgroup size (≤256 on web limits). Do **not** attempt the multi-workgroup variant (deadlocked on Metal; WGSL has no device-scope barrier — impossible by construction).
Effort: **M**. Benefit: dominant win for the interactive ≤256-body case (dispatch overhead → one dispatch); great for the web demos.

**D9. Throughput extras (after profiling, in order):** subgroup 8-lane split primal for high-degree bodies (feature-detect; scalar fallback — WGSL subgroups now ship in Chrome + wgpu native); encoder/pass-split experiments on wgpu (avbd-metal's 2.5–3× is Metal-specific hazard behavior; on wgpu measure submit-granularity vs single-encoder — rubble today has the *opposite* pathology, one submit per dispatch); SoA hot/cold body layout (pos/quat hot; vel/props cold); f16 experiments — non-goal for v1.
Effort: **M each**, gated on the perf harness. Benefit: constant-factor wins; keep only what measures.

**D10. Restitution (differentiator #1).**
Cache pre-solve relative normal velocity per contact at narrowphase; after velocity finalize, run a contact-parallel restitution pass applying `Δv_n = −(1+e)·min(v_n⁻+threshold, 0)` style impulses (mass-weighted, λ-capped), Jolt/Box2D-style. Params already public (`restitution_threshold` — currently dead).
Trade-offs: adds one pass + 8 bytes/contact; naive velocity pass can re-penetrate (bounded by next step's contact, standard).
Effort: **M**. Benefit: bouncing that the paper and both references simply lack.

**D11. Production API (v1).**
GPU-resident `World` with: mutation queue (add/remove/set_pose/set_velocity/apply_impulse/kinematic targets); **collision filtering** (groups/masks + pair-exclusion — currently absent); per-body friction/restitution/damping/user-data; **GPU queries** — raycast + AABB overlap traversing the live LBVH (batch-first, async; blocking convenience wrappers for tests); events (started/ended) computed on GPU from pair-map diff, compacted, surfaced via the async snapshot (fix the fake-generation-0 handles); fixed-timestep accumulator helper with catch-up clamp (webphysics's lessons); world snapshot/restore (non-hot-path full download/upload — enables save/load + golden tests); explicit `max_bodies`-free growth (size to live counts).
Effort: **L** cumulative. Benefit: this is what "production-ready" means to users; also closes the wasm marshaling risks (remove-body aliasing) found in the test-gap audit.

**D12. CPU mirror solver (differentiator #2, testing anchor).**
A small scalar Rust implementation of the *same* AVBD math over the same contact layouts (avbd-metal's CPUSolver proved this invaluable). Used for GPU↔CPU parity tests, algorithm documentation, and debugging; not a runtime fallback.
Effort: **M–L**. Benefit: the strongest correctness instrument available; neither reference's testing matches an oracle-backed dual implementation.

**D13. Determinism (same-adapter) as a supported mode.**
Sort GPU-appended lists (pairs, contacts) by stable keys before consumption (radix sort exists); index-based tie-breaks in coloring (already); no float atomics (already); fixed iteration counts (already). Document cross-platform bit-determinism as a non-goal (FMA/driver variance) — the CPU mirror covers bit-stable needs.
Effort: **S–M** (mostly a sort + tests). Benefit: replayable sims; a differentiator neither reference attempts (both are explicitly non-deterministic).

**D14. Codebase restructure.**
- **Delete `rubble-shaders`** (dead rust-gpu crate; removes the nightly pin) and **move `multi_gpu.rs` out of the engine core** (park as an example/util crate; multi-GPU returns later as *multi-world* parallelism, one world per GPU — the form that actually scales).
- **Single WGSL source tree**: numbered shared modules (`00_types.wgsl`, `01_math.wgsl`, `02_ldlt.wgsl`, …) composed by a tiny in-repo concatenator (avbd-metal's model) + WGSL `override` constants for tunables; 2D/3D share scheduling/coloring/persistence/sort/scan kernels and differ only in narrowphase + the 3×3-vs-6×6 solve. Kills the 2D/3D drift (2D currently uses CPU greedy coloring and has no warm-start shader).
- New internal crate layout: `rubble-core` (scheduling, buffers, kernels, coloring, persistence map, LBVH), `rubble3d`/`rubble2d` as thin dimension bindings + narrowphase, keeping `rubble-math/-shapes*/-testkit/-scenes/-wasm/-viewer`.
Effort: **M** (deletions S; shader-tree refactor M; crate moves M). Benefit: one solver to maintain; stable toolchain; smaller build.

---

## 3. Milestones (each test-gated; sequenced to de-risk)

**M0 — Baseline + guards (S–M).** Perf harness: fixed scenes (256/1k/4k/16k bodies; stack; pyramid; pile) recording per-phase `StepTimingsMs` to committed JSON baselines + CI delta job (non-blocking). Wire the browser testkit lane (CB9: wasm `run_scenario` export + `testkit.spec.ts`) so native (lavapipe) and web (SwiftShader) can't diverge silently. Exit: baselines committed; browser lane green.

**M1 — Solver fidelity on the current architecture (L).** D6 a–g + D10 restitution + D12 CPU mirror (parity tests native-only initially). Exit gates: pyramid + resting-rect known-failures pass; CA3/CA4 stay green; GPU==CPU parity within f32 tolerance on the oracle scenarios; default iterations 8.

**M2 — GPU-resident rewrite (XL; the centerpiece).** D2 + D3 landed behind a feature flag, sub-staged: (a) persistent body/shape buffers + mutation queue + async snapshot ring; (b) indirect dispatch + GPU CSR adjacency; (c) persistent coloring + compacted color lists; (d) CAS pair map warm-start + split contact record; (e) flip default, delete the readback path. Exit gates: **zero hot-path readbacks** (debug assert), two-world determinism (CA1) green on native+web, full oracle ladder green, perf baselines show upload/readback lanes ≈ 0.

**M3 — Collision robustness (L).** D4 (fattening, speculative margins, compound leaves, no-silent-drop growth) + D5 (feature IDs, hysteresis, pair-matrix fixes). Exit: compound/capsule/hull known-failures pass vs parry oracle; resting stacks hold pairs across 1k-frame soak; CA5 (broadphase vs brute force under motion) green.

**M4 — Scale + latency (L).** D7 sleeping/islands + active-list compaction; D8 persistent small-scene kernel; D13 determinism mode. Exit: settled 4k-body pile step cost drops ≥5×; ≤256-body scenes step in ≤3 dispatches; determinism replay test green.

**M5 — Production API + perf polish (L).** D11 (filtering, GPU queries, events, snapshot/restore, accumulator) + D9 experiments (subgroups, encoder granularity, SoA) kept only where measured. Exit: API docs + examples; benchmark page (native + web) published in-repo vs the two references' demo scenes.

**M6 — 2D rebuild on the shared core (M–L).** Re-host 2D on `rubble-core` (shared coloring/persistence/scheduling; 3×3 solve; 2D narrowphase); add parry2d oracle (CA13). Exit: 2D known-failures (resting drift) pass; 2D/3D share >70% of kernel source.

**Roadmap (post-v1, recorded for direction):** joints/motors/limits (AVBD-native: same block solve + geometric-stiffness SPD projection — `m3_diagonalize`), soft bodies/cloth via OGC, H-PLOC broadphase, BDF2 integrator option, multi-world multi-GPU, large-world origin shifting.

## 4. Performance targets (definition of "strictly better")

- **Web (Chrome, mid-range dGPU):** ≥10k awake bodies at 60 fps; ≥50k at 30 fps. Beats webphysics's 12,288-body max demo config with headroom.
- **Native (4090-class):** 100k-body pile ≤ 10 ms/step at 4–8 iterations (paper-parity order of magnitude).
- **Zero** hot-path CPU↔GPU synchronization (asserted in debug builds); CPU step cost = command recording only (≤0.5 ms at 10k bodies).
- Settled scenes: cost tracks awake bodies (sleeping ≥5× win on the pile scene).
- Every number tracked by the M0 harness on both lanes; regressions fail CI.

## 5. Risks & mitigations

1. **M2 is a big-bang risk** → sub-staged behind a flag with the M0 net + CA1 determinism gating each sub-stage; old path deleted only at the end.
2. **wgpu/WebGPU limits** (bind-group counts, 256-thread workgroups on web, subgroups exposure on wasm) → budget bindings per stage (webphysics hit this), feature-detect + scalar fallbacks; CI runs SwiftShader to catch web-only breaks.
3. **fp32 conditioning at k=1e10** → Jacobi preconditioning + pivot clamps (the references' proven fix); split tangential cap 1e6.
4. **Persistent coloring degradation** (adversarial graphs) → overflow color is Jacobi-safe by design; cap colors; monitor color-count in the perf overlay.
5. **Async snapshot semantics change tests** → keep a blocking `sync()` used by the oracle harness; document frame-lag for render consumers.
6. **Scope creep toward joints/soft** → explicitly out of v1; the solver is built so they slot in (same block solve), but no v1 work.

## 6. Verification

- Existing net stays authoritative: parry oracle, invariant ladder, known-failure registry (each milestone converts specific known-failures to passing gates), heavy/chaos lanes, two-world determinism, wasm marshaling tests.
- New: GPU↔CPU mirror parity (M1+), zero-readback assertion (M2), browser testkit lane (M0), perf baselines with CI deltas (M0), determinism replay (M4), benchmark suite vs references (M5).
- Everything green on **both** lavapipe (native) and SwiftShader-WebGPU (browser) before any milestone merges.

## 7. Explicitly rejected alternatives (for the record)

- Switching to TGS-Soft/impulse substepping (CPU-optimal, GPU-hostile constraint coloring), XPBD (dual-method mass-ratio + energy issues; 26 substeps still collapsed the paper's 510k scene), NCP/Newton-style global solvers (robotics accuracy, wrong scale model), IPC/ABD (offline accuracy), OGC for rigid contacts (avbd-metal disabled it for rigids).
- Multi-workgroup persistent kernels (no forward-progress guarantee — deadlocks even on Metal; impossible in WGSL).
- webphysics's colors×all-bodies dispatch and top-K pair pruning (scalability/correctness traps).
- Cross-platform bit determinism (driver/FMA variance; CPU mirror covers it).
- rust-gpu dual-source shaders and in-core multi-GPU (dead weight; cut per D14).

## 8. Status & next actions

1. ✅ This plan committed as `docs/production-plan.md` with superseding pointers in
   `avbd-competitive-analysis.md`, `performance-plan.md`, and `TODO.md`.
2. Next: **M0** (perf harness + browser testkit lane — test-only, zero engine risk), then M1 onward in order.

## 9. Execution log

### M0 — Baseline + guards (landed)
- Perf harness `crates/rubble3d/tests/perf_baseline_tests.rs`: six deterministic scenes
  (stack/pyramid/grid 256–4k/pile 1k), per-phase `StepTimingsMs` medians, JSON report,
  committed baselines + non-blocking CI `perf` job (`.github/workflows/ci.yml`).
- Browser testkit lane (CB9) deliberately deferred to M2's async-readback work, per
  `test-coverage-gaps.md`'s own sequencing note.

### M1 — Solver fidelity (landed)
- **D6a** Jacobi-preconditioned LDLᵀ 6×6 with pivot floor (replaces return-zero Gaussian
  elimination). **D6b** `m/dt²` penalty floor at contact prep + split caps (normal 1e9 /
  tangential 1e6, configurable). **D6c** Taylor-cached constraints: new `contact_prep`
  kernel freezes C0 rows + anchors in the solve-start frame (`solve_start_states`
  snapshot); primal/dual evaluate `C = C0 + J·Δx` against frozen Jacobians. **D6d**
  Eq. 14 secant stiffness rescaling — *friction rows only*: applying it to the normal
  row (force clamped to zero ⇒ zero Hessian) removes the two-sided stiffness that damps
  the contact boundary and limit-cycles stacks (measured, then scoped correctly).
  **D6e** post-solve velocity clamps (`max_speed`, `max_ang_speed`). **D6f** α kept at
  0.3 — measured: with rubble's recovery-clamp formulation, α=0.95 parks resting
  contacts ~20× deeper and destabilizes stacks/pyramids; the paper's 0.95 belongs to its
  history formulation (revisit under M2 if the formulation changes). **D6g** `substeps`
  config (full detection+solve per substep; default 1).
- **D10** Restitution: per-body coefficient (`RigidBodyDesc::restitution`), pre-solve
  normal speed captured in contact prep, post-extract per-body Jacobi restitution pass.
- **D12** CPU mirror (`rubble3d::mirror`): LDLᵀ, contact rows, primal update, dual
  update as scalar Rust with unit tests; full-step GPU↔CPU parity test
  (`mirror_parity_tests.rs`) pins the whole solve chain on a sphere–plane scene.
- **Determinism (D13 brought forward — three real races found and fixed):**
  1. Luby coloring wrote colors in the same dispatch that read neighbors' colors →
     improper colorings run-to-run → primal races between adjacent bodies. Fixed with a
     mark/apply two-phase round.
  2. LBVH pair emission silently dropped a scheduling-dependent subset when demand
     exceeded `bodies*8`. Fixed with grow-and-retry off the unclamped attempt counter
     (native; wasm single-shot until M2).
  3. Contact solve order inherited atomicAdd emission order. Fixed with a total-order
     sort (pair, feature, then raw bytes).
  Plus one compute pass per solve color (pass-boundary hazard ordering between
  Gauss-Seidel colors). Result: `determinism_probe_tests.rs` — detection stack and
  1-iteration solve are bitwise deterministic across processes; one residual
  multithread-scheduling source at ≥2 iterations remains (~1e-10/step; pinned as an
  ignored repro with bisection notes; owned by the M2 rewrite).
- **Exit-gate status:** full 3D+2D+testkit ladder green (200+ tests). Pyramid
  known-failure *improved* (tail ~13.3 → ~12 vs limit 10, wild variance eliminated) but
  not flipped; kept registered with updated notes — the residual rides the remaining
  nondeterminism source + M3's manifold work. 2D resting-rect gate moved to M6 with the
  2D rebuild (per D14: the 2D solver is replaced by the shared core, not hand-patched).

### M0 baselines (release, lavapipe) — the measured M2 thesis
At 4k bodies / 22k contacts: 607 ms/step, of which `contact_fetch` (full contact
download + CPU merge/sort/adjacency) = 318 ms and the solve lane (incl. coloring
submits + CPU orchestration) = 266 ms. Upload enqueue is negligible; the round-trips
are everything. Committed in `crates/rubble3d/tests/perf/baselines.json`.

### M2 — first slice (submit granularity + clean-frame uploads)
- Clean-frame upload skip: `World` tracks `gpu_dirty`; unmutated frames call
  `refresh_frame_params` instead of re-uploading every body/prop/shape buffer (the
  GPU already holds bit-identical state — enabled by removing the vestigial
  body-state ping-pong swap, which also stabilizes bind-group caches).
- Luby coloring: all 32 mark/apply rounds record into ONE submit (per-round uniform
  buffers + bind groups cached across frames) — was 33 submits per recolor.
- Step front-end: predict + prev-step snapshot + solve-start snapshot + AABB fused
  into ONE submit — was 4.
- Next (the real M2 body): GPU CSR adjacency + recolor-every-frame to drop the
  coloring's contact dependency, then the D3 split contact record + GPU sort to kill
  `contact_fetch` entirely.
