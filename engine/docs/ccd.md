# Continuous collision (CCD), GPU impulses and projectiles

How Rubble keeps fast bodies from passing through thin geometry between
steps, all inside the GPU step (`slang/kernels/ccd_step.slang`, scheduled by
`engine_exec::rigid_plan::encode_step` and by the fracture lab's
`engine-fracture/src/plan.rs`). It implements the CCD audit's rows 1–7.

## What a host does

| Want | Call | Notes |
|---|---|---|
| Look-ahead contacts for every awake body | `AvbdParams::flags \|= FLAG_SPEC_VELOCITY` | Off in the engine default, on in the fracture lab (see row 6). A body's contacts reach as far as it moves this step, capped at 0.5 m per collider unless host word 31 says otherwise |
| A body that must never tunnel | `BodyState::flags \|= BODY_CCD` (64) | Uncapped reach, the fast list, a time-of-impact (TOI) contact and a TOI start pose. For projectiles, rockets, thrown debris, vehicles at speed |
| An impulse at a point, a set velocity, or linear + angular | `WgpuSim::apply_impulse`, `set_kinematic_velocity`, `apply_linear_angular_impulse`, `queue_impulses` | Applied on the GPU at step start (`apply_impulses`), summed per body in record order, so the result does not depend on thread timing |
| A kinematic body (platform, elevator, door) | `flags & 3 == 2` and a velocity | Moved by its velocity in `predict`; dynamic bodies ride and are pushed; it wakes what it touches |
| Bullets that pierce | `WgpuSim::spawn_projectile(&ProjectileDesc)` | Segment casts, not bodies: energy cost per metre of collider, impulses on the bodies hit, hit and stop events (`poll_motion_events`) |
| The fracture lab's balls | `Planner::set_projectile_ccd(true)`, the lab's "CCD balls" checkbox | Shots carry `BODY_CCD` from then on |

## The step

```
apply_impulses (host records)
build_aabbs                 swept boxes; CCD and kinematic reach uncapped
broadphase_clear
ccd_register                one slot per BODY_CCD body (scratch, cap 1024)
broadphase_insert           sweeps longer than a cell go to the fast list
broadphase_query            grid entries only
broadphase_fast             each fast collider walks the cells along its sweep
projectile_step             segment casts; apply_impulses (late pass) for hits
write_work_args / carry_rest_pairs
ccd_toi                     GJK ray cast of each CCD pair along its relative motion
ccd_pick                    per body: the earliest TOI, then the lowest candidate
ccd_mark                    tags the winning pair; the narrowphase skips it
narrowphase / convex
ccd_emit                    the winner's manifold at the TOI pose
... graph, colouring ...
predict_bodies              a CCD body starts from its TOI pose
solve
```

### Row 1: look-ahead that scales

`colliderReach` caps a body's reach (0.5 m per collider) so one fast body
cannot inflate every neighbour's pairs. CCD and kinematic bodies are exempt.
The bound used to place a box in the hierarchical grid includes its sweep
(`_pad0 = boundR + max(reach, |sweep|)`), so a fast box sits at a level whose
27-cell stencil covers it.

A sweep longer than a grid cell would still miss pairs in the 27-cell stencil,
so `broadphase_insert` puts such a collider on the **fast list** instead of the
grid. `broadphase_fast` gives each one a workgroup: it walks every cell within
reach of its sweep segment (a tube, not the whole swept box) on every level and
table (uniform, split, hierarchical, the lab's rest and awake tables), plus the
plane, large and other-fast lists. It owns every pair it is part of, so nothing
is found twice. `tests/broadphase_oracle.rs` checks the candidate pairs against
a CPU brute force: 0 missed, 0 extra, 0 duplicate on both grids, while the
control (fast list off) misses pairs on the uniform grid and overflows the
hierarchical grid's oversized list.

On the uniform grid an oversized awake box (larger than a cell) is owned by the
larger box, which walks its own extent; the smaller one skips it.

### Rows 2–3: TOI start pose and TOI contacts

`ccd_toi` casts each CCD body's collider along its motion relative to each
candidate (start rotations, as Jolt's `LinearCast`): a GJK ray cast (van den
Bergen) in the pair's local frame. A cast starts one step back, so a pair
already touching and closing gets TOI 0. It ends when the distance stops
shrinking: a hit if the gap is within 5 mm (`GJK_TOUCH`), a miss otherwise.

Each body keeps the earliest TOI by an atomic minimum on its bits, then the
lowest candidate index among equal TOIs, so the winner is deterministic.
`ccd_mark` tags the winner, the narrowphase leaves it alone, and `ccd_emit`
builds its manifold at the TOI pose. The rows hold the gap along the path, so
AVBD solves it as an ordinary contact: real two-way forces, visible to
`last_contacts` and the stress solver. `predict_bodies` starts the body at its
TOI fraction of the motion, so it arrives at the surface rather than inside it.
A TOI that would not matter (the body stops short anyway) is skipped.

### Row 4: GPU impulses

The scratch buffer holds up to 1024 impulse records (8 words: body, mode and
key, impulse, point). `apply_impulses` runs one thread per body that has
records and sums them in key order. Host records come first in a step;
projectile hits append records, applied in a late pass in the same step.
Queuing more than the cap panics on the host.

### Row 5: projectiles

`projectile_step` advances each projectile along its arc in up to four
segments. Each segment cast walks the grid (`scene_query.slang` `sceneWalk`)
and orders hits along the path. A hit costs `penetration cost × thickness`
of kinetic energy (the exit point is a second cast from beyond the collider),
pushes the body with an impulse, and logs an event. A projectile stops when its
energy runs out and continues from its exit point otherwise. A character's
inner capsule is a collider like any other, so its hit event names the
character's collider. (A `BODY_CCD` body, such as a rocket, never passes a
character: `tests/character.rs`.)

### Row 6: on by default (evaluated, not adopted)

The fracture lab has look-ahead on (`LAB_FLAGS`). With `AvbdParams::default()`
switched to `FLAG_SPEC_VELOCITY`, every engine-webgpu suite passes (lib 63,
vehicles 32, columns, gables, ring, joints, CCD, characters, queries). The
default is still off because:

- `MetalWorld`'s legacy schedule then diverges from the shared schedule by
  8.4 m in `joints_wgpu_parity` (0 m with look-ahead off), so parity would
  need work on that host first.
- stress-rubble's look-ahead (`STRESS_RUBBLE_SPECULATIVE_CCD=1`) roughly
  doubles broken bonds in the bombard and is an open decision
  (docs/stress-qs-baseline-plan.md).

The tunnelling suite (`tests/ccd_tunnel.rs`) is the other half of row 6: each
case runs with look-ahead off, look-ahead on, and `BODY_CCD` (which turns
look-ahead on for the scene by itself).

### Row 7: content rules

- Thin pieces: a body thinner than its speed × dt tunnels without CCD. Flag
  fast movers `BODY_CCD` rather than raising the step rate.
- 120 Hz halves the distance per step but is no substitute for CCD at
  projectile speeds (the 250 m/s cases tunnel at 120 Hz without CCD).
- `stress-rubble` keeps its whole-world CCD substeps by default; TOI CCD for its
  impactors is opt-in (`set_toi_ccd`, see "In the demos").

## Limits

| Limit | Value | Over it |
|---|---|---|
| CCD bodies per step | 1024 | `OVERFLOW_CCD_BODIES` (4096) |
| Fast list | 1024 colliders, 262,144 cells per level walked | `OVERFLOW_FAST_LIST` (1024) |
| Impulse records | 1024 | host panic / `OVERFLOW_IMPULSES` (8192) |
| GJK iterations | 32 | not an overflow: counted in `ccd_fallbacks()`, and the pair keeps its ordinary look-ahead contact |
| Large dynamics on the uniform grid | 16 | `OVERFLOW_LARGE_DYNAMICS` (2048) |

Manifold reuse (`step_reuse_manifolds`, the lab's reuse mode) is off while CCD
bodies exist: a reused step would keep last step's TOI contacts.

## Tests

| Test | Shows |
|---|---|
| `engine-webgpu/tests/ccd_tunnel.rs` | 60 cases (60/140/250 m/s; 0.05/0.2/1 m slabs and brick walls; 60/120 Hz; both grids) hold; the controls tunnel. A debris shower through a slab: 0 through with CCD vs 51 (60 Hz) and 25 (120 Hz) without |
| `tests/broadphase_oracle.rs` | Candidate pairs equal a CPU brute force for sweeps of 0.5–10 cells |
| `tests/kinematic.rs` | Kinematic platforms carry and push; sleeping bodies wake |
| `tests/queries.rs` | Ray and shape casts match a full scan; bullets pierce five walls with exact energy accounting; impulses are exact and replay bit-identically |
| `stress-rubble/tests/engine_world.rs` | With TOI CCD on, a 250 m/s CCD impactor stops at a 0.1 m wall with one step per frame |
| `tests/static_pairs.rs` | A box lands on a lower-index static pedestal (on main it fell through: the grid query dropped dynamic–static pairs whose static had the lower index) |
| `engine-viewer/tests/fracture_regression.rs` | A 300 m/s lab ball tunnels through the arch without CCD and knocks the crown out with it |
| `engine-webgpu/motion-check.mjs` | The same in headless Chrome (`motion.html` and the fracture lab) |

## In the demos

| Page | CCD | Switch |
|---|---|---|
| `motion.html` | CCD gallery, character rockets, bullets | per lane, per body |
| `demo.html` | the smash balls of the built scenes and the snapshot scenes' F projectile carry `BODY_CCD`; `demo.html`'s own schedule runs the CCD passes only in a scene that has one | `?ccd=0` off. The `?check=ring\|gables` regressions keep it off unless `?ccd=1` (their native baselines run without it); the ring check passes with it too |
| `vehicle.html` | cannonballs (B) | `?ccd=0` off |
| `fracture/` | Rubble bond modes: the "CCD balls" checkbox (on). Stress-solver modes: whole-world substeps for their balls (below) | checkbox |

stress-rubble keeps whole-world CCD substeps by default: `BODY_CCD` turns on
look-ahead contact semantics for the whole scene, and the stress coupling's breakage
was measured against substeps (with `BODY_CCD`, `punch_gpu`'s ram left the wall at
6.5 m/s instead of 28, reference 35.1). `RubbleEngine::set_toi_ccd(true)` or
`STRESS_RUBBLE_TOI_CCD=1` opts in.

## Browser demo

`crates/engine-webgpu/motion.html`: `?scene=ccd` (ball gallery, CCD on/off
lanes, a debris shower), `?scene=character`, `?scene=bullets`. The fracture
lab has a "CCD balls" checkbox next to Ball speed.
