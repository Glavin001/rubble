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
cargo test -p rubble3d --test narrowphase_tests  -- --nocapture   # contact geometry vs parry (incl. convex)
cargo test -p rubble3d --test metamorphic_tests  -- --nocapture   # determinism / invariances
cargo test -p rubble3d --test ccd_tests          -- --nocapture   # tunneling / CCD characterization
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
| Friction (tangential hold) | gentle_incline_friction_holds | μ>tanθ ⇒ no slide | pass (penetration covered by gap #3) |
| **Inertia tensor** | inertia_tests | **parry3d mass-properties (independent)** | sphere/box/**capsule** match to **~1e-7** |
| **Determinism** | metamorphic_tests | same input twice | **bit-identical (0.0)** |
| **Mass-independence** | metamorphic_tests | free fall vs 1000× mass | **bit-identical (0.0)** |
| **Permutation invariance** | metamorphic_tests | reversed insertion order | **bit-identical (0.0)** — solving is order-independent despite graph coloring |
| **Narrowphase (collision detection)** | narrowphase_tests | **parry3d, all shape pairs incl. rotated** | normal axis + depth match **exactly** for sphere/box/capsule/plane (8 axis-aligned + 2 rotated) and for convex-convex / convex-box. The primitive detection layer is solid. (Exception: convex-**sphere** is broken — gap #5.) |

## Engine gaps DETECTED (registry: `gaps.rs`)

1. **Zero-torque angular-velocity drift** — `AngularIntegration`. `zero_g_sphere_constant_spin`: ω drifts ~3.3e-3 over 180 steps, *identically for a sphere and a box*. A sphere cannot precess, so this is a systematic quaternion integration/extraction bias (not gyroscopics); it exceeds the f32 random-walk floor by ~17×. Corrects the existing known-failure's "gyroscopic" attribution.
2. **Angular-momentum non-conservation** — `AngularIntegration`. `torque_free_box_angular_momentum`: L drifts **~14% of |L₀|** over 180 zero-torque steps (energy stays bounded). A torque-free body must conserve L exactly — the conservation-law quantification of gap #1.
3. **Penetration under tangential load** — `Solver`. `static_friction_holds`: a resting box sinks **~0.15 m** into the floor under tilted gravity (normal+tangential); straight-gravity resting does not. The normal constraint is under-resolved when a large tangential load is present.
4. **Explosive deep-penetration recovery** — `Solver`. `deep_overlap_separates`: two spheres started 0.6 m overlapping separate at **~70 m/s** (vs a 15 m/s sanity bound). No penetration-recovery velocity clamp — bodies spawned overlapping get launched.
5. **Convex-hull vs sphere narrowphase is wrong** — `Narrowphase`. `narrowphase_tests` (convex_sphere): a cube-hull penetrating a sphere reports a normal ~60° off parry's and a depth of 0.92 m vs the correct 0.10 m. Convex-convex and convex-box are exact, so the bug is specific to the convex↔sphere path.
6. **Convex-hull resting dynamics are unstable** — `Solver`. `convex_cube_rests`: a convex cube dropped on the floor gains energy (~2× E0), reaches ~118 m/s and flies to y≈-243. The convex↔box *narrowphase* is exact, so this is the dynamic contact solve for convex hulls (perhaps compounded by the bbox-approximated convex inertia). Convex hulls aren't yet usable as resting dynamic bodies.
7. **Compound bodies explode on contact** — `Solver`. `compound_box_rests`: a two-box compound dropped on the floor reaches ~3213 m/s by tick 1 and ends at y≈-6444. Refines the prior "compound falls through floor" known-failure to a violent explosion; compound contact handling is broken for resting on a surface.

## Characterizations — real, but expected / good-to-know (not "bugs")

- **Convex-hull inertia is a bbox approximation** (`inertia_tests`): for an octahedron the engine's inertia is ~3.3× the true value (0.48 vs 0.144). Affects rotational dynamics of convex hulls; primitives are exact.
- **Absolute-position f32 sensitivity / large-world precision** (`metamorphic_tests`): translating a scene 13 m from the origin changes the result by **~6 mm for free motion** and **~2 m for a contact scene** (chaos-amplified). The equations are translation-invariant; this is f32 precision in absolute-position math (notably finite-difference velocity extraction), and it grows with distance from the origin. Relevant for large worlds — keep scenes near the origin or use a local frame.
- **No continuous collision detection (CCD)** (`ccd_tests`): a sphere is correctly stopped by a thin solid wall up to ~40 m/s but **tunnels through at ~80 m/s** (≈0.67 m/step at 1/120s). Expected for a discrete-time engine without CCD; the correctness floor (slow bodies are always stopped) holds. Fast/small bodies vs thin geometry need CCD or substepping.

## Fault-detection matrix — does the suite catch injected bugs?

The matrix injects known bugs and confirms ≥1 *passing* scenario catches each
(proving the derived tolerances are tight enough to fail on a real bug):

| Fault | Caught by |
|---|---|
| NegateGravity | free_fall, projectile, drop_bounce, resting_box, gentle_incline, stack×2, box_rests |
| ZeroGravity | free_fall, projectile, box_rests |
| ScaleGravity(1.5) | free_fall, projectile, drop_bounce, resting_box, box_rests |
| DropSolverIterations→1 | two_body, drop_bounce, stack×2, newtons_cradle, unequal_mass |
| ZeroFriction | gentle_incline_friction_holds |

**All 5 faults are caught — no remaining blind spots.** A 1.5× gravity error, a
single solver iteration, or zeroed friction each trip ≥1 scenario, which is what
makes the *passing* results trustworthy rather than merely loose.

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

- **Kinematic bodies** — `set_body_kinematic` exists; add a test that a kinematic
  body follows its prescribed motion and is unaffected by collisions/gravity.
- **Convergence/order test** — does gap #1/#2 shrink at the integrator's order as
  dt→0? Distinguishes acceptable order-of-accuracy from a biased integrator.
- **2D coverage** (`rubble2d`) — the harness is structured to extend.
- **Convergence testing** for rotational accuracy — does gap #1/#2 shrink at the
  integrator's order as dt→0? Distinguishes "acceptable order-of-accuracy" from a
  biased integrator.
- **2D coverage** (`rubble2d`) — the harness is structured to extend.
- **Browser lane** — same scenarios + checks inside wasm under Chrome+SwiftShader,
  to catch browser-only (toolchain / binding / async / device-limit) divergences.
