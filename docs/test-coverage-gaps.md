# Rigid-Body Test-Coverage Gaps (core Rust + web E2E)

Scope: **rigid-body collision correctness only** (joints / soft bodies / cloth are out of scope).
Purpose: (1) catalogue what is *not* verified about *correct* rigid-body physics, in both the core Rust
engine and the end-to-end web/wasm path; (2) define the safety net that must exist before the performance
work in [`performance-plan.md`](./performance-plan.md) so a perf refactor cannot silently break physics.

Sources: full reads of `crates/rubble3d/tests/**`, `crates/rubble2d/tests/**`, `crates/rubble-testkit/src/**`,
`crates/rubble-wasm/src/lib.rs`, `web/src/{2d,3d}/main.ts`, `web/tests/*.spec.ts`, `.github/workflows/ci.yml`.

---

## TL;DR

- The engine has **two test tiers**. (1) An excellent **oracle ladder** (`rubble-testkit` + `narrowphase_tests`,
  `inertia_tests`, `metamorphic_tests`, `ccd_tests`, `fault_detection_tests`) with parry-backed, f32-derived
  tolerances and a fault-injection "test-the-tests" matrix. (2) A **legacy/2D tier** that is overwhelmingly
  finiteness/divergence guards with loose fitted tolerances (`< 2.0`, `is_finite`, `y > -10`).
- The oracle ladder is **narrow**: ≤4 bodies, contact manifolds cross-checked against parry **only statically**
  (0 solver iterations) and **only for the single deepest contact's normal+depth**, and determinism tested by
  replaying **one** `World`. The exact dimensions a perf refactor touches are blind.
- **2D has no oracle at all** (no `parry2d`, no analytic ladder) — every 2D test is a divergence guard.
- The **web/wasm path verifies essentially no physics correctness** — E2E asserts "loads / ticks / falls
  roughly down / not NaN within ~30 steps of a *random scatter* scene." Tolerances are 2–3 m on ~1 m bodies.
- `rubble-wasm` has **zero Rust tests**; its hand-written marshaling (transform packing, shape-offset
  bookkeeping, `remove_body`, `None→origin` substitution, quaternion order) is unverified.
- **There is no perf-regression / timing guard anywhere** (only `debug_assert!` sanity on a timer).
- **`rubble-testkit` is documented as dual-target native+wasm, but the wasm runner is `#[cfg(not(wasm32))]`
  — the browser lane was designed and never built.** Building it is the single highest-leverage action: it
  closes web parity, determinism, and per-scenario correctness at once, reusing the native oracles verbatim.

---

## Part A — Core Rust gaps

### A.0 Oracle assessment (what the good tier actually guarantees)
- Real oracle = `crates/rubble-testkit/`: `oracle.rs` (analytic endpoints — discrete symplectic-Euler
  ballistic recurrence, restitution ceiling, settled-at-rest, rest height, kinematic-follows-path),
  `invariants.rs` (**per-tick** NaN/Inf, quaternion unit-norm, energy non-increase ≈ `8·N_iter·eps`,
  linear-momentum ≈ `2·N_iter·eps`, floor non-penetration), independent parry checks
  (`inertia_tests.rs` vs `parry3d::mass_properties` ~1e-7; `narrowphase_tests.rs` vs `parry3d::query::contact`,
  normal cos 5°, depth ≤ 0.025), and `faults.rs` proving the tolerances have teeth.
- **What it does NOT assert:** manifold point *count* or *location*; per-contact normals across a multi-point
  manifold; manifold during *dynamic* stepping; angular-momentum on more than one (quarantined) scenario;
  friction-impulse direction/magnitude; anything above 4 bodies; anything without a CPU readback.
- **Registered gaps (known-broken, quarantined):** angular integration drift (ω drifts ~3.3e-3 over 180 zero-torque
  steps, *identical for sphere and box* → systematic integrator bias, not gyroscopics); `static_friction_holds`
  (box sinks ~0.15 m under tangential load); `torque_free_box_angular_momentum` (L drifts ~14%);
  `deep_overlap_separates` (0.6 m overlap → ~70 m/s, no recovery clamp); `convex_cube_rests` (hull gains ~2× energy,
  flies to y≈−243); `compound_box_rests` (explodes ~3213 m/s); `narrowphase` convex_sphere normal ~60° off.
  Legacy known-failures: 3D pyramid energy injection, capsule/hull/compound pair-matrix misses, 2D friction
  inverted, 2D resting drift.

### A.1 P0 — a perf refactor can silently break these today
> These are the safety net. Each targets a dimension the perf plan explicitly changes
> (broadphase / coloring / warm-start / GPU-residency / indirect dispatch / readback removal).

**CA1. Two-world & scheduling-order determinism.** Today only *same-World replay* is bit-exact
(`metamorphic_tests.rs:115`). Indirect dispatch, atomic pair-list append, GPU coloring reorder, and
`compact_states` reindexing are textbook run-to-run nondeterminism sources that same-World replay cannot see.
*Add:* `determinism_two_worlds_identical_seed` (3D+2D) — build **two** `World`s from identical descs (mixed
scene, ~12 contacts), step 240×, assert `worst_delta < 1e-6`; plus a **reversed-insertion-order** variant
asserting identical contact set + per-body final state.

**CA2. Contact-manifold count & location (the biggest blind spot).** The only manifold oracle runs at **0
solver iterations** and checks only the **deepest** contact's normal+depth. A broadphase/coloring/readback
change that drops one of a box's 4 floor contacts fails *no* test but tips boxes over in production. parry
already returns `point1/point2` (`parry_oracle.rs:135`) but they're discarded.
*Add:* `box_on_plane_manifold_matches_parry` — flat box, assert **exactly 4** points each within 1e-2 m of a
parry manifold point, normals within cos 5°; edge case → 2 points; vertex → 1 point; plus a **dynamic** variant
(60 steps, full solver) asserting count/normal hold mid-settle.

**CA3. Warm-start is not proven correctness-neutral.** Tests check lambda bookkeeping and "warm settles
≤1.2× cold" — not that warm-start converges to the **same** fixed point as cold. A warm-start match/decay/feature-
hash refactor could bias the converged rest state and still pass.
*Add:* `warmstart_converges_to_cold_rest_state` — 4-box stack with `warmstart_decay=0.999` vs `0.0`, 600 steps,
assert final positions within `2·penetration_slop` and energy within the derived bound.

**CA4. No correctness at scale (only finiteness >16 bodies).** `scale_256_bodies` etc. assert `is_finite`.
The headline use of a perf refactor is *more bodies* — exactly where the no-oracle hole is widest (color-bucket
overflow, pair-buffer overflow, AABB tie handling).
*Add:* `grid_rest_heights_correct_64` — 8×8 grid of boxes dropped 0.1 m, 300 steps, assert **every** box rests at
`y=0.5 ± (slop+offset)`, `drift < 0.05`. Reuses `RestHeight`/`LateralDriftBounded` at N=64. Sphere variant too.

**CA5. Broadphase pair-set vs brute force, under motion.** Current coverage is one static hand-placed 4-AABB
case. "Changing broadphase" is the first perf milestone; a BVH/sweep that misses or duplicates a pair (or feeds
coloring a different order) fails nothing.
*Add:* `broadphase_matches_bruteforce_under_motion` (3D+2D) — 40 seeded moving bodies, 120 steps, every tick
assert the engine pair set is a superset of the tight CPU O(n²) AABB-overlap set and contains every truly-overlapping
pair.

### A.2 P1 — real physics gaps, regression-prone (several already known-broken)
- **CA6. Angular integration** is provably biased and only one (quarantined) scenario quantifies it. Add a
  *passing ceiling* test `angular_momentum_drift_within_current_envelope` (L-drift < 20%) so a worse
  `extract_velocity` refactor fails even while the gap is open. Add a non-principal-axis spin case.
- **CA7. Restitution value never asserted** — only the `≤ e²h` ceiling. Add `inelastic_drop_rebound_near_zero`
  (rebound < 0.1·drop_height) so an energy-injecting normal-impulse refactor trips a *tight* bound, not the loose
  energy guard.
- **CA8. Kinetic friction has no quantitative oracle.** Add `kinetic_friction_stopping_distance` (box at 5 m/s on
  μ=0.5 floor stops within ±15% of `v₀²/(2μg)`) and `friction_cone_not_exceeded` (tangential impulse ≤ μ·normal).
- **CA9. Capsule / convex-hull / compound** dynamic correctness is quarantined; add gap-registered
  `*_rests_at_correct_height` + `*_separates_with_correct_axis` (vs parry) so a fix is *detected* and failure
  severity is *bounded* (e.g., deep-overlap separation-speed ceiling < 15 m/s for gap #4).
- **CA10. CCD/tunneling floor** is characterization-only. Add `tunneling_floor_holds_to_30ms` (no tunnel up to
  30 m/s) so reducing iterations or `contact_offset` can't silently lower the safe envelope.

### A.3 P2 — hygiene / cheap, lower probability
- **CA11. Stale-handle rejection** after `remove_body` (assert `get_position(stale)==None`, no slot aliasing,
  generation bump) — matters when `remove_body`/`compact_states` is reworked for GPU-residency.
- **CA12. dt-convergence** — run ballistic drop at dt and dt/2, assert each matches the discrete oracle for its
  own dt (guards the predict shader under any substepping refactor).
- **CA13. 2D has no oracle.** Add `parry2d` dev-dep; port `parry_oracle`/`narrowphase_tests`/`inertia_tests` +
  an analytic ladder to 2D. Minimum: `narrowphase_matches_parry2d` (circle/box/capsule/polygon normal+depth) and
  `inertia_matches_parry2d`. **Without this, any 2D perf refactor is untested for correctness.**
- **CA14. Demo metrics unused** — `demo_repro_tests` already collects full momentum/energy metrics but asserts
  only height/speed. Add momentum/energy-non-increase over the settled window (free).

---

## Part B — Web / wasm E2E gaps

### B.0 What the web path actually verifies
`rubble-wasm` exposes `PhysicsWorld{2D,3D}` to JS. State crosses the boundary as flat `f32` arrays — 3D
`[x,y,z,qx,qy,qz,qw]` per body via `get_transforms()` / `copy_transforms_into()` (`lib.rs:524-576`); 2D positions
+ angles. The 3 Playwright specs assert, in CI under headless Chromium/**SwiftShader-WebGPU**: "no console error",
"bodyCount ≥ N", "step counter increments", ">30% of bodies moved down by >0.01 after 30 steps", "y > −2.0"
(2 m tolerance on 1 m bodies), "quaternion length ∈ (0.9,1.1)" (magnitude only → **order-blind**),
"array length % 7 == 0", "no NaN" (3D only). **Zero assertions of a precise physics invariant.** The default E2E
world is the `?bodies=50` **JS-RNG random-scatter** path — *not* a deterministic scene and *not* `load_scene`.

### B.1 Marshaling correctness risks (pure Rust logic, untested)
- **Removed body renders at the world origin.** `remove_body` never pops `handles`/`shape_*`; `get_transforms`
  substitutes `None → (Vec3::ZERO, Quat::IDENTITY)` (`lib.rs:527-528,564-565`). `body_count()` (alive) and
  `handle_count()` (total) diverge while JS indexes by slot. **No test calls `remove_body`.**
- **Quaternion order is load-bearing and unverified** — only unit-length is checked, which is invariant under
  xyzw↔wxyz transposition.
- **NaN/Inf cross the boundary unguarded** (raw float push); only a JS-side post-hoc check on one 3D scene.
- **Shape-size offset bookkeeping is hand-maintained**; hull/compound/plane push an offset but **no sizes**
  (`lib.rs:670-677`) — a mixed-shape round-trip is never asserted.
- **JS sees an index, not the `{index, generation}` handle** — stale-handle detection is impossible from JS.

### B.2 Web-only logic invisible to all Rust tests
JS PRNGs (`0x3d5eed`/`0x2d5eed`) define the E2E worlds; `?bodies=N` imperative spawn **bypasses scenes**;
WebGPU→WebGL render fallback (physics still wgpu, but *nothing asserts the GPU solver ran*); transform→matrix
compose incl. **approximate** capsule scale; `MAX_INSTANCES=768` drop-but-still-simulate index alignment;
world `free()`/recreate on scene switch; 2D physics↔screen transform + `ctx.rotate(-angle)` sign flip; 3D has a
`frameInFlight` guard, 2D does not. One `step()` per rAF frame — no fixed-timestep/interpolation.

### B.3 P0 — web correctness gaps that would let real bugs ship
**CB1. No native↔web parity test.** Native and wasm share scene builders but use **different step paths**
(`step()` vs `step_async()`); a divergence (upload order/readback/dt) is silent. *Add:* commit a native
golden trajectory for fixed scenes ("Ground"/"Stack"/"Pyramid", 60 steps, no RNG), and `web/tests/parity.spec.ts`
that loads the same scene, steps 60, and asserts per-body position within an f32-derived tolerance + quaternion
dot ≥ 1−ε (assert resting heights & X-drift, not just "fell").

**CB2. No determinism test on the WebGPU path.** Native determinism runs lavapipe only; SwiftShader-WebGPU is
never replayed. (Note: the `Scatter` scene uses **unseeded `rand::rng()`** — inherently non-reproducible yet
exercised — seed it or exclude it.) *Add:* `web/tests/determinism.spec.ts` — fixed scene, step 30, snapshot;
reload; assert tight-ε equality.

**CB3. `rubble-wasm` has zero Rust tests.** *Add:* `crates/rubble-wasm/tests/marshal.rs` (native target — the
methods run natively): assert shape-type/offset/size indexing for a mixed scene; `get_transforms().len()==handle_count()*7`;
after `remove_body(1)`, `body_count()==handle_count()-1` and the slot reads exactly `[0,0,0,0,0,0,1]` (pins the
origin-aliasing behavior); `copy_transforms_into` rejects a wrong-length buffer.

**CB4. wasm step ≠ proven == native step (and the async path is never tested even natively).** `step()` is sync
natively but `step_async()` on wasm32, so a native wasm-crate test exercises the *sync* path only. *Add:* a native
test comparing `World::step()` vs the `step_async` body under `pollster::block_on` for a fixed scene; fold the
browser side into CB1.

**CB5. Nothing proves the GPU solver actually ran** (the step counter increments even if `step()` were a no-op).
*Add:* in parity/determinism specs, assert a resting stack holds analytic spacing (only the real solver produces
it) and that a (to-be-exposed) solve-phase timing is non-zero.

### B.4 P1 — important web correctness coverage
- **CB6. In-browser stacking/resting** is never asserted (heavy scenes are switched to but only error-checked).
  Add `web/tests/stacking.spec.ts` — load "Stack", 120 steps, assert analytic rest heights + a small post-settle
  jitter bound.
- **CB7. Energy/momentum/velocity are unrepresentable web-side** — wasm exposes only position+rotation; native
  `get_velocity`/`last_contacts`/testkit metrics are not bridged. Needs a `get_velocities()` export; meanwhile a
  **position-only ballistic oracle** spec (port the `oracle.rs` recurrence to JS) validates the integrator through
  the boundary.
- **CB8. WebGPU-vs-Vulkan parity** — CI runs SwiftShader-WebGPU (web) vs lavapipe-Vulkan (native), never
  cross-checked; CB1's golden makes this explicit.
- **CB9. Build the dual-target `rubble-testkit` browser lane (highest leverage).** The harness is documented as
  native+wasm so "the native lavapipe lane and the browser Chrome+SwiftShader lane can never silently disagree
  about what 'correct' means" — but `runner`/`run_*` are `#[cfg(not(wasm32))]`. *Add:* a `rubble-wasm` export
  `run_scenario(name) -> JsValue` driving `rubble-testkit` scenarios through `PhysicsWorld{3D,2D}`, plus
  `web/tests/testkit.spec.ts` asserting `report.violations` is empty per scenario. **This single piece of plumbing
  closes CB1/CB2/CB4/CB6/CB7 for every scenario the native lane already covers**, reusing the existing oracles.

### B.5 P2 — web hygiene
- **CB10.** `web/package.json` `test:webgpu` references a **non-existent** `tests/webgpu-narrowphase.spec.ts` —
  create it (assert specific 2-body overlap contact outcomes) or fix the script.
- **CB11.** 2D has no NaN test and no ground-contact acceptance test — add both.
- **CB12.** Scene-switch `free()`/recreate leakage (A→B→A body-count/position identity) untested.

---

## Part C — The safety net to build *first* (gates the performance work)

A perf refactor changes broadphase, coloring, warm-start, GPU-residency, indirect dispatch, and removes CPU
readbacks. The minimum set that makes those changes *safe to land* (highest signal, ordered):

1. **CB9 — wire the wasm testkit runner + `testkit.spec.ts`.** Reuses 19 existing oracle-backed scenarios; gives
   native↔web parity + WebGPU determinism + per-scenario correctness in one move. Foundational for everything web.
2. **CA1 + CB2 — two-world / scheduling-order / WebGPU determinism.** Directly guards indirect-dispatch & atomic
   broadphase & coloring-reorder nondeterminism.
3. **CA2 — manifold count/location vs parry (static + dynamic).** Guards the contact path that GPU-residency and
   readback removal touch most.
4. **CA5 — broadphase vs brute-force under motion.** Guards the first perf milestone (broadphase) explicitly.
5. **CA4 + CA3 — scale correctness + warm-start correctness-neutrality.** Guards "more bodies" and the warm-start
   persistence-map rewrite.
6. **CB3/CB4 — `rubble-wasm` marshaling + sync/async equivalence.** Guards the boundary the perf work will also
   touch (e.g. async readback for rendering).
7. **A perf-regression guard** (see performance-plan.md Milestone 0) — there is none today.

Lower priority but valuable in parallel: CA6–CA10 (real physics gaps), CA13 (2D oracle), CB6/CB10/CB11.

Every item above is a *test addition only* — no engine change — and most reuse existing oracles (`rubble-testkit`,
`parry_oracle`, the discrete ballistic recurrence). That is the point: build the net cheaply, then refactor for
speed against it.
</content>
