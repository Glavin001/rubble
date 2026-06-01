# Rubble physics engine — gap report

Produced by the `rubble-testkit` detection harness. This is the deliverable: an
explicit, categorized account of what the engine gets right (with the oracle that
proves it), what it does **not**, and characteristics worth knowing. The goal of
this pass was to *detect* bugs and map gaps — not to fix them.

Run it (software Vulkan, CI-equivalent):

```bash
sudo apt-get install -y mesa-vulkan-drivers
export WGPU_BACKEND=vulkan
export VK_ICD_FILENAMES=$(find /usr/share/vulkan/icd.d -name 'lvp_icd*' | head -1)

cargo test -p rubble3d --test harness_tests      -- --nocapture   # scenario ladder + per-tick invariants
cargo test -p rubble3d --test inertia_tests      -- --nocapture   # inertia vs parry (independent)
cargo test -p rubble3d --test metamorphic_tests  -- --nocapture   # determinism / invariances
cargo test -p rubble3d --test fault_detection_tests -- --ignored --nocapture   # "test the tests"
RUBBLE_RUN_KNOWN_FAILURES=1 cargo test -p rubble3d --test harness_tests        # surface every gap
```

Every tolerance is **derived** (f32 round-off or a config limit), never fitted; a
failure prints `tick / body / measured / bound` and a failing run dumps its full
trajectory. The fault matrix below proves the tolerances are tight enough to
catch real bugs, not merely loose enough to pass.

## Confidence — what PASSES, and the oracle that proves it

| Area | Scenario / test | Oracle | Result |
|---|---|---|---|
| Integration | free_fall, projectile | discrete symplectic-Euler recurrence | exact to derived f32 bound |
| Angular energy | zero_g_box_energy_conserved | torque-free KE conservation | pass |
| Pairwise collision | two_body, newtons_cradle, unequal_mass | momentum conservation + "struck body moves" | pass |
| No energy creation | drop_bounce | rebound ≤ drop height | pass |
| Resting | resting_box, box_rests_at_correct_height | non-penetration + exact rest height + settle | pass |
| Stacking | stack_two_boxes, stack_four_boxes | non-penetration + settle + energy | pass |
| **Inertia tensor** | inertia_tests | **parry3d mass-properties (independent)** | sphere/box/**capsule** match to **~1e-7** |
| **Determinism** | metamorphic_tests | same input twice | **bit-identical (0.0)** |
| **Mass-independence** | metamorphic_tests | free fall vs 1000× mass | **bit-identical (0.0)** |
| **Permutation invariance** | metamorphic_tests | reversed insertion order | **bit-identical (0.0)** — solving is order-independent despite graph coloring |
| **Narrowphase (collision detection)** | narrowphase_tests | **parry3d, all shape pairs incl. rotated** | normal axis + penetration depth match **exactly** for sphere/box/capsule/plane (8 axis-aligned + 2 rotated): the detection layer is solid; gaps are in the solver/integration |

## Engine gaps DETECTED (registry: `gaps.rs`)

1. **Zero-torque angular-velocity drift** — `AngularIntegration`. `zero_g_sphere_constant_spin`: ω drifts ~3.3e-3 over 180 steps, *identically for a sphere and a box*. A sphere cannot precess, so this is a systematic quaternion integration/extraction bias (not gyroscopics); it exceeds the f32 random-walk floor by ~17×. Corrects the existing known-failure's "gyroscopic" attribution.
2. **Angular-momentum non-conservation** — `AngularIntegration`. `torque_free_box_angular_momentum`: L drifts **~14% of |L₀|** over 180 zero-torque steps (energy stays bounded). A torque-free body must conserve L exactly — the conservation-law quantification of gap #1.
3. **Penetration under tangential load** — `Solver`. `static_friction_holds`: a resting box sinks **~0.15 m** into the floor under tilted gravity (normal+tangential); straight-gravity resting does not. The normal constraint is under-resolved when a large tangential load is present.
4. **Explosive deep-penetration recovery** — `Solver`. `deep_overlap_separates`: two spheres started 0.6 m overlapping separate at **~70 m/s** (vs a 15 m/s sanity bound). No penetration-recovery velocity clamp — bodies spawned overlapping get launched.

## Characterizations — real, but expected / good-to-know (not "bugs")

- **Convex-hull inertia is a bbox approximation** (`inertia_tests`): for an octahedron the engine's inertia is ~3.3× the true value (0.48 vs 0.144). Affects rotational dynamics of convex hulls; primitives are exact.
- **Absolute-position f32 sensitivity / large-world precision** (`metamorphic_tests`): translating a scene 13 m from the origin changes the result by **~6 mm for free motion** and **~2 m for a contact scene** (chaos-amplified). The equations are translation-invariant; this is f32 precision in absolute-position math (notably finite-difference velocity extraction), and it grows with distance from the origin. Relevant for large worlds — keep scenes near the origin or use a local frame.

## Suite blind spot (from the fault-detection matrix)

The matrix injects known bugs and confirms the suite catches them:

| Fault | Caught by |
|---|---|
| NegateGravity / ZeroGravity / ScaleGravity(1.5) | free_fall, projectile (+ contact scenes) |
| DropSolverIterations→1 | two_body, drop_bounce, stack |
| **ZeroFriction** | **NOT DETECTED** |

4/5 caught by multiple scenarios. **`ZeroFriction` is a blind spot**: the only
friction scenario is quarantined for gap #3, and no *passing* scenario loads
friction tangentially. Friction is under-tested until gap #3 is fixed or a
gentle-incline-that-holds scenario is added.

## Methodology findings (the tests were wrong before the engine was)

Surfaced and fixed with derivations, not looser numbers — this is what keeps the
tolerances honest:

- **Velocity tolerance** must scale as `pos·ε/dt` (velocity is finite-differenced
  from f32 positions), not a flat constant.
- **"Constant spin" is invalid for anisotropic bodies** (a tumbling box precesses);
  applied only to isotropic spheres, with energy/angular-momentum used for boxes.
  This is *why* gap #1 is credible — the sphere isolates the numerical drift from
  real precession.
- **Momentum tolerance** coefficient set to the f32 round-off floor a 5:1
  mass-ratio collision actually lands at (~5e-5 relative).

## Not yet covered (next, in confidence order)

- **Convex-hull narrowphase pairs** — the narrowphase matrix covers
  sphere/box/capsule/plane; convex-hull pairs are not yet validated against parry
  (and convex inertia is already a known bbox approximation).
- **A passing friction scenario** to close the ZeroFriction blind spot.
- **Convergence testing** for rotational accuracy — does gap #1/#2 shrink at the
  integrator's order as dt→0? Distinguishes "acceptable order-of-accuracy" from a
  biased integrator.
- **2D coverage** (`rubble2d`) — the harness is structured to extend.
- **Browser lane** — same scenarios + checks inside wasm under Chrome+SwiftShader,
  to catch browser-only (toolchain / binding / async / device-limit) divergences.
