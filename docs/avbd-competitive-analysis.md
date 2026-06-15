# AVBD Competitive Analysis & Actionable Roadmap

**Goal:** make rubble's AVBD solver *strictly better in every way* than two reference
implementations that are currently more stable and performant:

- **avbd-metal** — `https://github.com/LuckyIYI/avbd-metal` (Swift + Metal). A near‑faithful GPU port of
  Chris Giles' SIGGRAPH‑2025 AVBD demo, extended with cloth, tet‑FEM soft bodies, joints, OGC contact,
  and a robotics/world‑model layer. Apple/Metal only.
- **webphysics** — `https://github.com/jure/webphysics` (TypeScript + WebGPU via Three.js TSL). AVBD rigid +
  soft + cloth + joints + springs, LBVH broadphase, optional sleeping. Chrome/WebGPU only.

This document is the result of a line‑by‑line read of all three solvers. Every recommendation cites the
**reference technique** (`repo:file:line`) and the **rubble site to change** (`crates/...:line`).

> **Headline finding.** Both reference solvers are **single time‑step, ~10 iterations — they do *not*
> substep.** Rubble is also single‑step (20 iterations). So rubble's architecture is *not* the problem.
> The gap is that rubble is missing the AVBD *stabilization machinery* that both references share, and it
> pays heavy CPU↔GPU round‑trip costs that neither reference pays. Closing those two gaps is the bulk of
> the work.

---

## 1. Capability matrix (where each repo stands today)

| Axis | avbd-metal | webphysics | **rubble (today)** |
|---|---|---|---|
| Language / backend | Swift + Metal | TS + WebGPU (TSL) | **Rust + wgpu (WGSL)** |
| Portability | Apple only | Chrome only | **Native (Vk/Metal/DX12) + Web (WebGPU) + WASM** ✅ |
| 2D | ✗ | ✗ | **✅** |
| Multi‑GPU | ✗ | ✗ | **✅ (`MultiGpuContext`)** |
| Time stepping | 1 step × 10–25 iters | 1 step × 10 iters | 1 step × 20 iters |
| α‑regularized contact (`(1−α)C0 + JΔx`) | ✅ α=0.99 | ✅ α=0.95 | **✗ (full current penetration enforced)** |
| Bounded dual λ (mass‑scaled cap) | ✅ `min(λmax, 1e5·minMass)` | ✅ ±1e10 | **✗ (no lower bound)** |
| Penalty floor = `m/dt²` | ✅ | partial (k_min=1 + inertial diag) | **✗ (fixed `k_start=1e4`)** |
| Split normal/tangential penalty cap | ✅ 1e10 / 1e6 | 1e10 / 1e10 | single 1e6 |
| Trust‑region step caps | ✅ on (0.35·r, 0.5 rad) | present, off in main | **✗** |
| Preconditioned LDLᵀ + pivot clamp | ✅ Jacobi + pivot 1e‑6 | ✅ LDLᵀ + pivot 1e‑6 | **✗ (Gauss‑elim → returns *zero* on singular)** |
| Velocity safety clamp | ✅ cell‑tied | (relies on α) | **✗** |
| Adaptive warm‑start guess (`accelWeight`) | ✅ | ✅ | **✅ already present** |
| Quaternion double‑cover guard | ✅ | ✅ | **✅ already present** |
| Coupled 2D friction cone | ✅ | ✅ | **✅ (3D length‑clamp)** |
| Stable contact feature IDs (anti‑flicker) | ✅ feature+proximity | ✅ SAT hysteresis + edge margin | **✗ (documented "unstable feature IDs")** |
| Warm‑start λ basis rotation (tangent) | ✅ | ✅ | **✗** |
| Warm‑start match cost | O(1) CAS map | per‑pair manifold slots | **O(new × prev) brute force** |
| State residency | GPU‑resident | GPU‑resident | **CPU‑authoritative (full up/download each frame)** |
| Indirect dispatch (no count readback) | ✅ everywhere | ✅ contacts | **✗ (reads `contact_count`, downloads contacts every frame)** |
| Per‑body adjacency build | GPU CSR | GPU | **CPU, rebuilt every frame** |
| Coloring | GPU incremental, persistent, O(deg) | GPU incremental, persistent | GPU Luby, **O(bodies×contacts)/round**, order built on CPU |
| Single‑dispatch small‑scene solve | ✅ `solve_persistent` | ✗ | ✗ |
| SIMD lane‑split primal (8 thr/body) | ✅ | ✗ | ✗ |
| Restitution | ✗ | ✗ (dead code) | **✗ (param exists, unused)** |
| Sleeping / islands | ✗ | optional | ✗ |
| Joints / springs | ✅ | ✅ | **✗** |
| Cloth / tet‑FEM soft bodies | ✅ | ✅ | ✗ |
| CPU reference solver (for parity tests) | ✅ | ✗ | **✗** |
| Narrowphase shape coverage | broad | mostly box+sphere | **broad (box/capsule/hull/compound/plane), 2D+3D** ✅ |
| Test rigor | 16 suites | regression HTML | **✅ invariant + parry‑oracle + known‑failure tracking** |

**Rubble already wins** on portability, 2D+3D, multi‑GPU, shape coverage, and test rigor. To be *strictly
better in every way* it must (a) match the stability/perf core, and (b) add the missing features
(joints → springs → soft bodies). Items below are ordered to get the biggest stability/perf wins first.

---

## 2. Reference parameter cheat‑sheet (verified from source)

| Param | avbd-metal | webphysics | **rubble today** | Recommended rubble target |
|---|---|---|---|---|
| α (contact regularization) | 0.99 (`Scene.swift:210`) | 0.95 (`avbdState.ts:36`) | **0 (absent)** | **0.95** (tunable 0.9–0.99) |
| γ (penalty + λ decay) | 0.999 | 0.99 | 0.999 (`lib.rs:58`) | keep 0.99–0.999 |
| λ warm‑start scale | α·γ≈0.989 | α·γ≈0.94 | α·γ=0.99·0.999 (`mod.rs:53`) | α·γ |
| β (penalty ramp rate) | 10000 lin / 100 ang | 10000 | 1e4 (`lib.rs:56`) | keep ~1e4 |
| k_min / floor | `m/dt²` | 1.0 + inertial diag | **1e4 fixed** | **`max(1, minMass/dt²)`** |
| k_max normal | 1e10 | 1e10 | **1e6** | 1e8–1e10 |
| k_max tangential | **1e6** | 1e10 | 1e6 | keep 1e6 |
| λ cap (normal) | `min(λmax, max(10, 1e5·minMass))` | ±1e10 | **none** | **mass‑scaled cap** |
| iterations | 10 (16–25 stacks) | 10 (×2 sweeps) | 20 | 10–15 (after fixes) |
| trust region lin / ang | 0.35·r / 0.5 rad | (off) | **none** | **0.35·r / 0.5 rad** |
| μ combine | √(μₐμᵦ) | √ × material | √(μₐμᵦ) ✅ | keep |
| collision margin | 0.01 | 5e‑4 | 0.01 (`CONTACT_MARGIN`) | expose as param |
| restitution | none | none | dead param | **implement (differentiator)** |

---

## 3. P0 — Solver stability (fixes rubble's documented divergence/jitter)

These directly target the *known‑failure* scenarios rubble already tracks:
- 3D: *"AVBD solver injects energy into large 3D pyramid stacks; max_speed diverges over time"*
- 2D: *"resting rect develops large upward drift and spin"*, *"frictionless circle impacts create extra translational energy"*

### P0.1 — Add α‑regularized constraint targets `C_reg = (1−α)·C0 + J·Δx` ⭐ highest impact
**Why:** This *is* the "augmented/stabilized" half of AVBD. Both references only drive the constraint toward
`(1−α)·C0` per step (correcting 1–5 % of the *original* violation), letting the rest bleed out
exponentially. Rubble instead enforces the **full current penetration every iteration**
(`c_n = dot(normal, separation) + MARGIN`), which slams deep penetrations to zero in one step and injects
energy — exactly the documented stack divergence.
- Reference: avbd-metal `contactForceC` `40_solver.metal:902` (`C = C0*(1-alpha) + J·dq`); webphysics
  `avbdState.ts:2194` (`cRegN = (1-α)*c0 + dot(jAL,dPosA)+…`).
- Rubble sites: 3D primal `crates/rubble3d/src/gpu/avbd_solve_wgsl.rs:324`; 3D dual `:470`;
  2D primal `crates/rubble2d/src/gpu/avbd_solve_wgsl.rs:138`; 2D dual `:236`.
- **Change:**
  1. Freeze `C0 = (c0_n, c0_t1, c0_t2)` once when a contact is created / warm‑started (in narrowphase
     `narrowphase_wgsl.rs`), storing it in spare contact lanes (`normal.w`, `tangent.w`, and an anchor `.w`
     are free in `Contact3D`; `point.w`/`local_anchors` spare in `Contact2D`).
  2. Snapshot each body's **solve‑start pose** (avbd‑metal `initLin/initAng`, webphysics `initialPose`).
     Rubble already has `inertial_states` (the target) and `old_states` (pre‑predict). Add an explicit
     `solve_start_states` (= warm‑started `bodies` at iteration 0) so `Δx = x − x_start`.
  3. In primal/dual replace `c_n = dot(n, sep) + MARGIN` with
     `c_reg = dot(n, sep) − α·c0_n` (≡ `(1−α)·c0_n + J·Δx`, since `sep ≈ c0 + JΔx`). Same for tangentials.
- **Effort:** M. **Risk:** medium (touches the heart of the solve). **Validation:** the ignored
  `demo_pyramid_stays_supported_by_floor_without_exploding_3d` and `resting_rect_stays_quiet_on_floor_2d`
  should pass with `RUBBLE_RUN_KNOWN_FAILURES=1`. Expect to *reduce* iterations afterward.

### P0.2 — Bound the dual with a mass‑scaled cap ⭐
**Why:** avbd-metal's single biggest stability mechanism. Wedged/conflicting contacts otherwise ratchet
force toward infinity; a light body must not stockpile heavy forces. Rubble's normal force is clamped
`≤ 0` but has **no lower bound** → unbounded accumulation in stacks.
- Reference: avbd-metal `40_solver.metal:1575,1586` (`lamCap = min(λmax, max(10, 1e5·minMass)); F.x = max(F.x,-lamCap)`),
  friction cone `:908‑914`; webphysics `avbdState.ts:2791,2837‑2843`.
- Rubble sites: 3D `avbd_solve_wgsl.rs:327` (`f_n = min(k_n*c_n+lambda_n,0.0)`), dual `:473`;
  2D `:146`, dual `:247`.
- **Change:** compute `min_mass = min(mass_a, mass_b)` (statics → +∞), `lam_cap = min(MAX_PENALTY_FORCE, max(10.0, 1e5*min_mass))`,
  then `f_n = clamp(k_n*c_reg_n + lambda_n, -lam_cap, 0.0)` in both primal and dual.
- **Effort:** S. **Risk:** low. **Validation:** stack divergence test; mass‑ratio stress tests already in
  `physics_exhaustive_tests.rs`.

### P0.3 — Mass‑scaled penalty floor + split normal/tangential caps
**Why:** A contact should never be softer than the body's own inertial term `m/dt²`, and the tangential
penalty must stay ≤ 1e6 to preserve fp32 rolling‑mode conditioning even when the normal cap is much higher.
Rubble starts every contact at a flat `k_start=1e4` regardless of mass, and caps all axes at 1e6.
- Reference: avbd-metal `npPenaltyFloor` `30_narrowphase.metal:91‑101` (`k = max(PENALTY_MIN, minMass/dt²)`),
  tangential cap `00_common.metal:13‑16`.
- Rubble sites: `narrowphase_wgsl.rs:176,212` (`contacts[slot].penalty = vec4(k_start,…)`);
  `MAX_CONTACT_PENALTY` `mod.rs:57`.
- **Change:** initial `penalty.xyz = max(k_start, min_mass/dt²)` for normal, `min(that, 1e6)` for tangentials;
  raise normal `MAX_CONTACT_PENALTY` to ~1e8, keep tangential cap at 1e6 (carry a second constant or clamp
  per‑axis in the dual ramp at `avbd_solve_wgsl.rs:489,495‑496`).
- **Effort:** S. **Risk:** low.

### P0.4 — Trust‑region step caps in the primal
**Why:** Rubble applies the raw 6×6/3×3 solve delta directly (`pos - solution.lin`,
`small_angle_quat(-solution.ang)`), so one iteration can overshoot or tunnel a body through a thin contact.
avbd-metal caps linear to `0.35·boundingRadius` and angular to `0.5 rad` per iteration (its substitute for
line‑search).
- Reference: avbd-metal `40_solver.metal:1202‑1207`; webphysics `avbdState.ts:2455‑2464` (mechanism present).
- Rubble sites: 3D apply `avbd_solve_wgsl.rs:356‑357`; 2D `avbd_solve_wgsl.rs:168`.
- **Change:** pass per‑body bounding radius (already known from shape data) into the solve; clamp
  `|solution.lin|` to `0.35·r` and `|solution.ang|` to `0.5` before applying.
- **Effort:** S–M (need radius in the bind group). **Risk:** low. **Validation:** high‑velocity impact tests
  in `physics_exhaustive_tests.rs` / `gpu_performance_tests.rs`.

### P0.5 — Replace "return zero on singular" with a damped, pivot‑clamped solve
**Why:** `solve_6x6` returns `(0,0)` when any pivot `< 1e‑9` (`avbd_solve_wgsl.rs:190‑192`) — the body's
entire update is dropped that iteration, causing sticking/jitter and slow convergence. Both references use
**LDLᵀ with pivot flooring** (`max(pivot, 1e‑6)`) and Jacobi diagonal preconditioning so the solve is always
SPD and never explodes.
- Reference: avbd-metal `solve6x6` `00_common.metal:123‑181` (Jacobi precondition + pivot clamp);
  webphysics `avbdState.ts:2402‑2453`.
- Rubble sites: `avbd_solve_wgsl.rs:170‑223` (`solve_6x6`), 2D `solve_sym_3x3` `:54‑79`.
- **Change:** (a) add diagonal regularization `A[i][i] += 1e‑4` (matches webphysics `:1909`);
  (b) Jacobi‑precondition `S=diag(1/√A_ii)`; (c) on tiny pivot, clamp to `1e‑6` and continue instead of
  returning zero. Optionally port the in‑register LDLᵀ.
- **Effort:** M. **Risk:** medium. **Validation:** near‑degenerate stacking + coplanar box tests.

### P0.6 — Velocity safety clamp at extraction
**Why:** Defense‑in‑depth against tunneling: cap post‑solve speed so a body can't cross more than ~one
broadphase cell per step.
- Reference: avbd-metal `finalize_velocities` `40_solver.metal:1880‑1882`, `maxSpeed` `GPUSolver.swift:1366`.
- Rubble site: `crates/rubble3d/src/gpu/extract_velocity_wgsl.rs:59‑67` (and 2D analog).
- **Change:** `if (len² > maxSpeed²) v *= maxSpeed/len;` with `maxSpeed = max(30, 0.5·cell/dt)`.
- **Effort:** S. **Risk:** low.

---

## 4. P1 — Performance (eliminate CPU round‑trips; go GPU‑resident)

Neither reference reads anything back from the GPU in the hot path. Rubble, by contrast, every `step()`:
uploads all body state, **reads `contact_count`**, **downloads all contacts**, rebuilds adjacency on the CPU,
re‑uploads it, downloads contacts again after the solve, and downloads body states
(`mod.rs:1393,1396,1441,1451,1465,1476`; `build_body_contact_adjacency` `:1265`). This dominates frame time
at scale and serializes the pipeline.

### P1.1 — Build per‑body contact adjacency (CSR) on GPU
- Reference: avbd-metal CSR clear→count→scan→scatter `GPUSolver.swift:2058‑2094`; webphysics
  `buildPairBodyListsKernel` `contactGeneration.ts:2036‑2135`.
- Rubble site: replace CPU `build_body_contact_adjacency` (`mod.rs:2893`) + per‑frame upload (`:1265‑1272`).
- **Effort:** M. **Risk:** medium. **Impact:** removes a per‑frame CPU pass + upload.

### P1.2 — Indirect dispatch off GPU‑side contact/pair counts (kill `contact_count` readback)
- Reference: avbd-metal `bp_finalize_pairs`/`color_scan` write `dispatchArgs` (`20_broadphase.metal:202`,
  `40_solver.metal:406`); webphysics `buildContactDispatchArgsKernel` `avbdState.ts:554‑572`.
- Rubble site: `contact_count.read` at `mod.rs:1393,1441`; solve dispatch loop `:1286‑1310`.
- **Effort:** M. **Risk:** medium. **Impact:** removes the mid‑frame stall.

### P1.3 — Build the colored body order on GPU (radix sort by color)
**Why:** rubble computes the color → body‑order mapping on the **CPU** after reading back `body_colors`
(`coloring_wgsl.rs:111‑114`), forcing a readback whenever the contact graph changes.
- Reference: avbd-metal `color_count`/`color_scan`/`color_scatter` `40_solver.metal:365‑424` (counting sort,
  fully on GPU). Rubble already ships `GpuRadixSort` (`rubble-primitives`) — wire it in.
- **Effort:** M. **Risk:** low–medium.

### P1.4 — Replace O(bodies×contacts) Luby step with O(degree) using the CSR adjacency
**Why:** `coloring_wgsl.rs:81` scans **all** contacts for **every** body each round → O(N·M) per round ×
multiple rounds. Use the P1.1 adjacency so each body only scans its own contacts; keep colors persistent
across frames (incremental recolor) like both references.
- Reference: avbd-metal `color_iterate` `40_solver.metal:295` (early‑out when nothing changed, colors persist);
  webphysics `greedyBodyColorsKernel` `avbdState.ts:1438‑1548` (reuses previous color).
- **Effort:** M. **Risk:** medium.

### P1.5 — O(1) warm‑start matching via a GPU persistence map (drop the O(new×prev) scan)
**Why:** `warmstart_wgsl.rs:65` linearly scans every previous contact for every new contact — O(N²).
- Reference: avbd-metal CAS open‑addressing `pm_insert`/`pairMapFind` `20_broadphase.metal:246‑275`;
  webphysics per‑pair manifold‑slot snapshot `contactGeneration.ts:1402‑1452`.
- **Effort:** M–L. **Risk:** medium. **Impact:** the warm‑start pass stops being quadratic.

### P1.6 — `solve_persistent`: collapse the iteration loop into one dispatch for small scenes
**Why:** rubble issues `20 × (per‑color primal + dual)` dispatches per frame (`mod.rs:1286‑1310`); at ~40 µs
launch latency each, dispatch overhead dominates small scenes. avbd-metal runs the entire
`iterations × colors × dual` loop in **one** threadgroup using device‑memory barriers.
- Reference: avbd-metal `solve_persistent` `40_solver.metal:1667`, dispatched `GPUSolver.swift:2198`.
- **Effort:** L. **Risk:** medium (WGSL workgroup‑barrier semantics; single‑workgroup cap). **Impact:** large at
  low body counts (the common interactive case).

### P1.7 — Split command encoders at stage boundaries
**Why:** avbd-metal measured **2.5–3× faster** splitting encoders per stage vs one mega‑encoder
(`GPUSolver.swift:1785‑1787`); rubble wraps the whole solve batch in one encoder (`mod.rs:1280‑1311`).
- **Effort:** S. **Risk:** low.

### P1.8 — (Advanced) 8‑lane cooperative‑split primal with subgroup reductions
**Why:** rubble runs **one thread per body**; bodies with many contacts (a box on dense ground) serialize a
long gather, leaving the GPU at low occupancy. avbd-metal uses 8 threads/body + `simd_shuffle_xor` reduction
and applies it to rigids too.
- Reference: `primal_particles_split` `40_solver.metal:1263‑1335`.
- Rubble: requires `wgpu` subgroup ops (`subgroupAdd`/shuffle); gate behind a feature/limit check.
- **Effort:** L. **Risk:** medium‑high (portability of subgroup ops across wgpu backends). Do last.

---

## 5. P2 — Contact robustness & feature stability

Targets rubble's known failures: *"unstable feature ids"*, *"narrowphase paths miss contacts"*, and weak
warm‑start carry.

### P2.1 — Stable SAT feature IDs + anti‑flicker hysteresis
**Why:** rubble's `feature_id` is derived from the winning SAT axis indices
(`narrowphase_wgsl.rs:694,824‑827`), which flip frame‑to‑frame, breaking warm‑start matching
(key = `(min_body,max_body,feature_id)` in `warmstart_wgsl.rs:57‑70`).
- Reference: webphysics packs a 9‑bit key `ordinal|incAxis|refAxis|featureType` (`contactGeneration.ts:1702‑1714`)
  and **keeps the previous winning axis** when within a tolerance band (`:603‑759`), plus an edge‑win margin;
  avbd-metal matches boxes by exact feature ID and round shapes by proximity (`30_narrowphase.metal:817‑838,998‑1010`).
- **Change:** adopt the structured key + previous‑feature hysteresis so resting contacts keep a stable ID.
- **Effort:** M. **Risk:** medium. **Validation:** `no_duplicate_feature_ids_per_pair`,
  `pair_matrix_contacts_match_geometry_3d`.

### P2.2 — Warm‑start: nearest‑point match + tangent‑basis rotation of carried λ
**Why:** rubble carries λ only on an exact key match and never rotates the tangential impulse into the new
contact basis, so friction state is lost or misdirected when the basis turns.
- Reference: webphysics `contactGeneration.ts:1558‑1579` (rotate `shadow.t1/t2` old→world→new basis, then cone
  clamp); avbd-metal `30_narrowphase.metal:824‑827`.
- Rubble site: `warmstart_wgsl.rs:70‑84`.
- **Effort:** M. **Risk:** low–medium.

### P2.3 — Fatten broadphase AABBs (constant margin + velocity horizon)
**Why:** prevents resting stacks from dropping/re‑adding pairs ("buzzing") and adds speculative coverage.
- Reference: webphysics `broadPhase.ts:356‑357` (`aabbMargin=0.01`, `aabbVelocityHorizon=1/60` → `velPad=|v|·h`).
- Rubble site: AABB compute in the LBVH path (`mod.rs` AABB dispatch / `rubble-primitives/gpu_lbvh.rs`).
- **Effort:** S. **Risk:** low.

### P2.4 — Close the missing‑contact narrowphase paths
**Why:** ignored tests show capsule/hull/compound paths miss contacts in 3D and capsule‑capsule misses in 2D.
- Rubble sites: the relevant tests are `pair_matrix_tests.rs:1132,1252,1583` (3D) and
  `pair_matrix_tests.rs:41` (2D capsule). Use the existing **parry oracle** (`tests/support/parry_oracle.rs`)
  as ground truth.
- **Effort:** M per pair. **Risk:** low. Pure correctness — needed for "strictly better" narrowphase.

---

## 6. P3 — Feature parity (both references have these; rubble does not)

### P3.1 — Joints & springs (distance, ball‑socket, hinge + motor/limits)
- Reference: avbd-metal `stampJoint`/`dual_joint_one` (`40_solver.metal:598,1499`, SPD‑diagonalized geometric
  stiffness `m3_diagonalize`); webphysics `jointRecord.ts`, `springRecord.ts`, joint stamping `avbdState.ts:2040‑2065`.
- Rubble: add a `Constraint` buffer parallel to `Contact`, stamp into the same per‑body 6×6, share the dual
  ramp. The block solver is already general (it's just more rows). This is the **next big feature** and unlocks
  ragdolls, vehicles, the wrecking‑ball demos both references show off.
- **Effort:** L. **Risk:** medium.

### P3.2 — Sleeping / islands
- Reference: webphysics optional sleep (post‑process). avbd-metal leans on adaptive warm‑start instead.
- Rubble: per‑body low‑velocity counter → skip solve when an island is quiescent. Pairs naturally with the
  GPU adjacency (P1.1) to find islands.
- **Effort:** M. **Risk:** low–medium. **Impact:** big perf win on settled scenes + reduces resting jitter.

### P3.3 — Soft bodies (tet‑FEM, Stable Neo‑Hookean) and cloth (quadratic bending)
- Reference: avbd-metal `stampTet`/`stampMembrane`/`stampBend` (`40_solver.metal:811,863,887`),
  `25_cloth.metal`; webphysics `softBodyLattice.ts`, `springClothPatch.ts`.
- Rubble: largest scope. Required to be *strictly better than avbd‑metal* (which makes soft/cloth first‑class).
  Stage after joints; reuse the same colored block‑descent loop (vertices are just 3‑DOF bodies).
- **Effort:** XL. **Risk:** high. Roadmap item, not near‑term.

---

## 7. P4 — Differentiators (be *better*, not just at parity)

1. **Restitution done right (neither reference has it).** The param already exists
   (`restitution_threshold`, `lib.rs:43,61`) but the solver never reads it. Add a Jolt/Bullet‑style restitution:
   apply a post‑solve normal‑velocity bias above the threshold (a small velocity pass after `extract_velocity`),
   or bake a target separation into the contact. Gives rubble bouncing that both references lack.
2. **CPU reference solver + GPU↔CPU parity tests.** avbd-metal's clean `CPUSolver.swift` is its correctness
   anchor; rubble has *no* CPU reference (only WGSL). A small `rubble`‑side CPU AVBD over the same `Contact`
   layout would let invariant tests assert GPU==CPU within tolerance — a testing axis stronger than either repo.
3. **Determinism mode.** With GPU‑resident state + fixed iteration counts (both references already fixed), rubble
   can offer cross‑backend deterministic stepping — valuable for networked/robotics use and beyond either repo.
4. **Keep the portability moat.** Maintain native + WebGPU + WASM + multi‑GPU as first‑class while closing the
   gaps; this is the durable advantage neither Metal‑only nor Chrome‑only reference can match.

---

## 8. Suggested sequencing (milestones)

**M1 — Stability core (P0):** α‑regularization → dual caps → penalty floor/caps → trust region → damped LDLᵀ →
velocity clamp. Re‑enable the ignored stack/resting known‑failure tests as the acceptance gate. *Expectation:*
divergence gone; iteration count drops 20 → ~12.

**M2 — GPU‑resident hot path (P1.1–P1.5, P1.7):** CSR adjacency on GPU, indirect dispatch, GPU body‑order,
O(degree) coloring, persistence‑map warm‑start, encoder splitting. *Expectation:* per‑frame CPU time and
upload/download bandwidth collapse; large‑scene scaling improves markedly.

**M3 — Contact robustness (P2):** stable feature IDs + hysteresis, warm‑start basis rotation, fat AABBs, fix the
missing narrowphase pairs against the parry oracle.

**M4 — Throughput (P1.6, P1.8):** persistent single‑dispatch small‑scene solver; subgroup lane‑split primal.

**M5 — Features (P3) + differentiators (P4):** joints/springs, sleeping, restitution, CPU reference solver; then
soft bodies/cloth as the long‑horizon roadmap to surpass avbd‑metal's feature set.

---

## 9. Quick‑win shortlist (highest impact / lowest risk, do first)

1. **P0.2** mass‑scaled dual cap — a few lines, directly attacks stack divergence.
2. **P0.1** α‑regularization — the core AVBD stability fix.
3. **P0.3** mass‑scaled penalty floor + split caps.
4. **P0.4** trust‑region step caps.
5. **P0.5** damped pivot‑clamped solve (stop returning zero).
6. **P1.7** encoder splitting (2.5–3× reported, trivial change).

> Sources: rubble read at `crates/rubble{2d,3d}/src/gpu/*`; avbd-metal at
> `Sources/AVBDCore/{Shaders,CPU,GPU}/*`; webphysics at `src/physics/gpu/*`, `src/physics/PhysicsEngine.ts`.
</content>
</invoke>
