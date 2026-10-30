# Deformable terrain

How rubble makes ground that a meteor, an explosion or a heavy body really reshapes, so
that the new crater, rim, ejecta blanket or debris pile changes how vehicles, characters
and debris move across it. Cosmetic surface effects (footprints, splashes) are a separate,
cheaper layer and are not covered here.

Decisions (2026-10-24): settled terrain is 2.5D (craters, rims, mounds, ruts; no caves or
tunnels); it runs live in the engine in real time; explosions and meteor strikes first.

## Representation

Material is in exactly one of three states, and moves between them conservatively.

| State | Representation | Why |
|---|---|---|
| At rest (almost all terrain, almost all the time) | **Layered heightfield**: a grid of columns, each a short stack of material layers (material, thickness, bulk density) over bedrock. | A cheap, exact collider for 10k+ rigid bodies. Layers keep stratigraphy: a crater wall shows overturned strata and an ejecta blanket carries buried material. |
| Moving (a local *active region* around an event) | **GPU MLS-MPM** on a sparse block grid with **Drucker–Prager elastoplasticity**: friction angle and cohesion for soil, tensile strength and damage softening for rock. | The standard continuum method for large-deformation granular and soil flow (Klár et al. 2016; Wolper et al. 2019). Overhangs, the ejecta curtain, uplift, slope failure and repose come from the constitutive law. Grid transfers give two-way rigid coupling (Hu et al. 2018). |
| In flight (material leaving the active grid) | **Ballistic parcels** with an exact volume. | Free to fly; a parcel deposits its volume where it lands, or re-enters an active region. |

A depth-averaged (shallow-water) granular flow on the heightfield was considered and
rejected as the moving representation: it cannot represent the overturned flap, the ejecta
curtain or material driven down under the cavity, and needs uncalibrated forcing and lift
closures to imitate them.

## Event sources

- **Heavy bodies** (boulders, fracture chunks, vehicles; m/s to hundreds of m/s): no
  closure. The rigid body is coupled to the MPM grid; penetration, splash and crater are
  computed.
- **Explosions and hypervelocity impacts** (km/s): the contact and compression stage needs
  an equation of state, melt and vapour, and is not resolvable in real time. It is
  replaced by a point-source initial condition: a Maxwell Z-model radial excavation
  velocity field (v ∝ r^−Z, Z ≈ 2.7) scaled by the Holsapple–Schmidt π-scaling coupling
  parameter, with depth of burial for charges. MPM then computes excavation, ejection, rim
  uplift and collapse.

The crater shape is never prescribed. Scaling laws enter in two places only: sizing the
active region (with a margin, checked by an insensitivity test), and the initial velocity
of a km/s source.

## Lifecycle

Forward only; nothing is re-simulated (AGENTS.md).

1. **Request.** A hit above the activation threshold, or an explosion or impact API call,
   requests an active region sized from π-scaling with a margin so the bed does not limit
   the cavity.
2. **Activate** (end of the frame before the step). Heightfield columns in the footprint
   become particles layer by layer; the cells are marked active, so the heightfield
   collider ignores them; bedrock below the region is the MPM bed boundary.
3. **Step** (inside the rigid step). MPM substeps at its CFL limit with rigid poses
   interpolated, and returns reaction impulses to the bodies. The sparse grid grows by
   blocks as material flows.
4. **Eject.** Isolated, low-density particles become parcels; landing parcels deposit.
5. **Bake.** When the region's kinetic energy stays below ε for a dwell time, particles
   are binned back to layered columns (volume exact), the cells revert to heightfield and
   the particles are freed. Baked ground is the same material: a later hit reactivates it.

Everything above runs as Slang kernels on the frame's command stream. The host only
decides when to open or close a region from GPU-written flags, enforces the limits and
schedules the phases.

## Scale

The active region's cell size is a fraction of the predicted transient radius (an explicit
cells-per-radius setting): about 10–15 m cells for a 1 km crater, centimetres for a
grenade. The heightfield keeps its own resolution; the bake integrates volume from the MPM
grid onto its columns. Long modification stages (tens of seconds) stay affordable through
block sleeping: a block whose particles are at rest sleeps and wakes on incoming flux or
contact. That changes results only if the wake test misses something, so it is behind a
flag and measured before it becomes a default.

## Limits

Nothing degrades silently. All limits are engine-wide configuration (`TerrainConfig`) and
fail with a clear error when exceeded: open areas, particles, grid blocks, parcels,
particles per cell, minimum cell size, heightfield size. Elastic stiffness defaults to
physical values; softening it to need fewer substeps is a lossy trade that is measured and
reported, not adopted.

## Milestones

| | Scope |
|---|---|
| T1 | Layered heightfield collider: `ShapeType::HeightField`, convex-vs-heightfield narrowphase on the GPU, `create_height_field_shape` and height edits in the public API, a terrain demo with many bodies. |
| T2 | Granular MPM core, no rigid coupling: `engine-terrain` host crate, `terrain_mpm.slang` (sparse P2G/G2P, Drucker–Prager return mapping, bed boundary, eject and bake). |
| T3 | Two-way rigid coupling; contact/hit events (`public-api.md` P4) as the activation trigger. Explicit staggered coupling first; resting stability of heavy bodies on sand decides whether AVBD needs an implicit coupling block. |
| T4 | Explosion and impact sources, ejecta parcels, an impact demo at 1× with a cross-section view and a vehicle driving into the settled crater. |
| T5 | Cohesive soil, damage-softening rock, slope failure, a Bekker–Wong rut tier, an Unreal sample. |

## Where it stands (2026-10-25)

All five milestones are built; the sections below record each, with the
measurements behind them.
- **T1.** Heightfield collider (convex, CCD, fracture and soft paths), editable.
- **T2.** GPU MLS-MPM sand: Mohr–Coulomb and Drucker–Prager; activation and
  bake in whole volume quanta.
- **T3.** Two-way rigid coupling, a momentum-conserving inelastic exchange
  stable at any mass ratio. Strikes open regions before they land: a GPU
  look-ahead, chosen over hit events, which arrive after the rigid step.
- **T4.** π-scaled impact and explosion sources: a Z-model field bounded at the
  transient radius. Ejecta parcels; the World and C APIs; `impact.html` in the
  browser, sharing the Rust command plan.
- **T5.** Cohesive soil, damage-softening rock, vehicle ruts (Bekker), and the
  slope-failure check.
- **Multiple impacts.** Several areas in one sparse domain: far hits open
  their own, near ones grow them, sources superpose, each area bakes on its
  own ("Multiple impacts" below).
- **Wheels on open sand.** Two-way: a wheel on an open area's live surface
  presses its contact force into the sand, which ruts, berms and carries the
  car ("Wheels on open sand" below).
- **Rendering.** A render-only GPU pass rebuilds the ground over every open
  area from its particles every frame, and lists what flies as debris. A
  patch per area keeps the wheels on the live sand.
  `showcase.html` is a 120 Hz chase-camera run past three strikes; see
  "Rendering and showcase".

**Defaults** (`TerrainConfig::default()`):
- physical stiffness (E = 50 MPa);
- absorbing ground with a geostatic start;
- the hybrid bake: tent at home for unmoved sand, P2G's B-spline for moved
  sand, in the unactivated ground's frame;
- a π-scaled region's bed 1.5 R deep (`bed_depth`; "Performance").

**Profiles** (`TerrainProfile`, `TerrainConfig::profile`): `Physical` (the
default, what the validations run at) and `Game` (8 cells/R, bed 1.25 R,
Courant 0.45, no growth on the clock's speed bound), which `impact.html`
uses by default ("Performance").

**Validated:**
- a resting region is the identity;
- the slope-failure check;
- a dropped ball against Uehara's penetration scaling;
- the light-box coupling;
- P-wave absorption;
- demo-scale settling;
- the crater against π-scaling in radius (1.25 R), shape (d/D 0.17) and the
  volume ledger;
- no axis ridge or window-edge ridge;
- the volume ledger through forced closes with ejecta in flight.

**Open:**
- **Crater volume.** About 2.6× π-scaling's (margin 0.6), a constant factor
  over two decades of energy (slope 0.836 against 0.830). The field now
  conserves the impactor's momentum (2026-10-29), which took 2.79× to 2.62×.
  What remains is the source counting the excavation push twice (crater-scale
  ground leaves 2–5× faster than Housen & Holsapple's launch law), plus a
  bowl deeper than Holsapple's Kd/Kr ("Volume cause found"). Calibrating the
  source to the volume was decided against.
- **Wall give.** Constant-E sand at depth gives metres at the absorbing walls.
  Hardin-type pressure-dependent stiffness would fix it at the root, but it
  changes validated results.
- **Coupling.** Bodies that rotate fast within a frame (poses are the frame's
  start).
- **Hosts.** An Unreal sample project. The browser has no strike look-ahead and
  no ruts yet. Native Metal runs the terrain plan (`MetalSim`); CUDA compiles
  it but has no executor ("Native backends").
- **Real time** comes from scale. Centimetre-cell events run far slower than
  1×, by the user's choice of physical stiffness.

## T1 status (2026-10-24)

The heightfield collider is in:
- `ShapeType::HeightField` (`engine-schema/src/heightfield.rs`, packed into the hull
  arena; `slang/physics/heightfield.slang`). Pairs take the convex stages
  (`convex_classify`), so no binding is added.
- Contacts for spheres (exact closest point), capsules (end caps, the axis at every
  grid-line and diagonal crossing, ridge edges) and boxes/hulls (vertices, terrain
  vertices inside, ridge edge pairs), reduced to four points with one normal.
- Ray, sphere and capsule casts, exact box/hull sweeps (`scene_query`), the character
  controller, CCD (time of impact, TOI manifold, post-solve sweep check).
- `World::create_height_field_shape`, `World::set_height_field_region` (wakes bodies
  that slept over the edit), `shape_mesh`; C ABI `rbCreateHeightFieldShape`,
  `rbHeightField_SetRegion` (Unreal axes mapped exactly); the browser host
  (`height_field_arena`, `world.setHullRecords`) and `terrain.html`.

Engine fixes found on the way, both affecting capsules on any ground:
- contact persistence pinned a capsule's anchors while static friction held (only
  spheres were exempt), so a rolling capsule sank through planes;
- the solver took capsules' lever arms at the current rotation (spheres use the
  step-start one), so small, fast-rolling capsules sank (0.75 m at 60 Hz on a 20°
  plane; now under 1 cm).

Known limits:
- One normal per manifold (the solver's model): a body spanning a sharp crease gets
  the depth-weighted mean of its contacts' normals.
- A terrain sample inside a box or hull takes the least-deep hull face as its
  contact. On flat ground beside a resting box's side (inside it by less than
  the rest depth), that is the side face, and its horizontal normal tilts the
  shared one: at μ = 0.05 the box slid 7.8 mm in 2 s, at μ = 0.8 not at all.
  Restricting the face to ones the terrain supports there made boxes sink or
  slide on uneven terrain and under CCD, so it is not done.
- Not in the fracture lab's own pipeline (`fracture_step`, its own hull buffer),
  in `MetalWorld` or on CUDA. `MetalSim` takes heightfields since 2026-10-25
  ("Native backends").
- Rolling spheres and capsules lose speed as their spin grows, on planes too (an
  existing solver effect, not the heightfield's).
- `char_move` compiles in about 8 s in Chrome on Metal, against about 5.6 s with
  the heightfield routines stubbed out: the capsule clearance is inlined at each
  contact and cast call site.
- A projectile's path through terrain is its centre line's run under the surface
  plus its radius at each end; a shot that does not come back out over the field
  stops in it.

## T2 kernel design (2026-10-24)

From a survey of open GPU MPM codes: Taichi's MLS-MPM examples, taichi_elements,
Claymore, Genesis, NVIDIA Newton's implicit MPM, dimforge's wgsparkl and sparkl,
EA's pbmpm and WebGPU-Ocean. Read to understand the methods; no code or outputs
are taken from them.

- **Discretisation.** Explicit MLS-MPM: quadratic B-splines, the APIC affine matrix
  C, and the stress term fused into P2G as −dt·V·(4/dx²)·τ (Hu et al. 2018).
- **Sand.** A frictional return map in Hencky (log) strain with the log-volume
  correction (Tampubolon et al. 2017). The yield surface is Mohr–Coulomb by default
  (2026-10-24, after the column-collapse validation below); Klár et al.'s (2016)
  Drucker–Prager cone stays available. Both:
  - ε = ln max(|σ|, 1e-4) − c + log Jp/3, with cohesion c (0 for dry sand), so
    compression first consumes the volume a particle gained when it separated, and
    tr = Σε (ε̂ = ε − tr/3 is traceless).
  - tr ≥ 0: project to the tip, and Jp keeps the volume.
  - Otherwise, Mohr–Coulomb: principal Kirchhoff stresses τᵢ = 2μεᵢ + λ tr ε
    (ε₁ ≥ ε₂ ≥ ε₃) satisfy (τ₁ − τ₃) + (τ₁ + τ₃) sin φ ≤ 0. Flow is non-dilatant
    (ψ = 0), so the return is closed form onto the main plane (Δγ = f/4μ) and a 2×2
    solve onto an edge.
  - Or the cone: δγ = ‖ε̂‖ + (3λ + 2μ)/(2μ)·tr·α, with α = √(2/3)·2 sin φ/(3 − sin φ).
    Project onto the cone when δγ > 0. It equals Mohr–Coulomb on the compression
    meridian only; in triaxial extension a 30° cone acts like about 46°.
  - Stress τ = U·diag(2μ ln σ + λ Σ ln σ)·Uᵀ, with no division by small σ.
  - Klár's friction-angle hardening comes later, behind a flag.
- **SVD.** A 3×3 quaternion Jacobi (McAdams et al. 2011) with a fixed number of
  sweeps (6), then a 2×2 SVD of R's trailing block, which resolves nearly rank-1 F
  (σ₂ ≲ √ε·σ₁) that FᵀF cannot in f32. It is sign-corrected (det U = det V = +1; σ₃ may be negative) and lives in
  its own function. Its NaN guards must not rely on IEEE behaviour that fast-math
  removes; engine-metal parity is checked.
- **Sparse grid.** Blocks of 4³ cells in a fixed-capacity GPU hash
  (`TerrainConfig::max_blocks`), rebuilt every substep from the blocks the particles'
  stencils reach. Overflow sets a flag and the step fails with a `TerrainError`;
  blocks are never dropped.
- **Binning.** A deterministic counting sort of particles by (block, cell), with
  particle id breaking ties (prefix sums, as in the rigid solver's stable colour
  packing). P2G then sums in a fixed order, so runs and backends (WGSL, Metal,
  SPIR-V) agree.
- **P2G is a gather, with no grid atomics.**
  - One workgroup per active block, one thread per node.
  - A shared tile holds the block and its halo. Each node accumulates the particles
    whose stencil covers it, in sorted order, and is written once.
  - The tile fits WebGPU's default 16 KiB of workgroup memory, checked when the
    pipeline is created.
  - Fixed-point grid atomics, used by the WebGPU ports, were rejected: their scale
    trades range against resolution, and overflow wraps silently.
- **Grid update.**
  - Nodes with mass above a threshold get normalised velocity plus gravity.
  - The bed is a separating Coulomb contact on grid velocity against the heightfield
    under the region: the normal approach is removed and the tangential part is
    clamped by the friction cone (the nodal form of Newton and Genesis).
  - Or, with `RegionBoundary::Absorbing`, the bed and side walls of an
    activated region are the absorbing ground: dashpots and springs on the
    geostatic traction ("Absorbing ground" below). The rigid bed stays
    available.
  - Material reaching the region's box raises a region-overflow flag (an error that
    asks for a larger margin) until ejection to parcels exists (T4).
- **G2P.** Velocity and C come from the grid, then x += dt·v and F ← (I + dt·C)F. The
  next P2G applies the return map to F.
- **Time step.**
  - Each substep satisfies CFL on the P-wave speed plus the flow:
    dt = k·dx/(c + |v|max), with c = √((λ + 2μ)/ρ).
  - The host schedules substeps from the material's wave speed; the GPU checks the
    actual |v|max. A violation, or more than a configured maximum of substeps,
    fails with an error. Nothing clamps dt or velocity.
  - **Cost of physical stiffness.** Sand's Young's modulus (10–100 MPa) gives
    c ≈ 100–250 m/s, so dx = 0.1 m needs dt ≈ 0.1–0.3 ms: about 50–170 substeps per
    60 Hz frame. Softening E would be a lossy trade, so it is only measured and
    reported, not adopted. An implicit step (Newton's approach, after Daviet &
    Bertails-Descoubes 2016) is the alternative for large steps.
- **Volume is integer.** Each particle carries its volume as a whole number of quanta.
  Activation, bake and the ledger sum quanta with u32 atomics, so conservation is
  exact and independent of order.
- **Activation.**
  - Inside the region, the heightfield is lowered to the region's bed (the bottom
    of its box). The material between the bed and the old surface becomes
    particles: particles_per_cell_axis³ per cell, at deterministic offsets.
  - The bed stays a heightfield, so rigid bodies over the region stand on it until
    two-way coupling (T3).
- **Bake.** When the region's kinetic energy stays below ε for a dwell time, each
  particle's quanta are splatted bilinearly onto the four heightfield samples
  around it, in whole quanta, and the particles are freed ("The splat bake"
  below).

## T2 and T3 status (2026-10-24)

**T2 (granular MPM core)** is in `slang/kernels/terrain_mpm.slang`, the host
`engine-webgpu/src/gpu_terrain.rs` and `engine-terrain`:
- transfers, sand, CFL substeps;
- indirect dispatch over the blocks and particles in use;
- GPU activation from the heightfield, and bake back into it, in whole volume quanta.

Tests (`tests/terrain_mpm.rs`, `tests/mpm_material.rs`, `engine-terrain`):
- a uniform motion is carried exactly;
- free fall;
- linear momentum is conserved to 1e-6 of the mass;
- runs are bit-identical;
- material stops on the bed;
- limits fail by name;
- activate-then-bake with no motion is the identity to 1e-5 m;
- volume is kept through flow and bake;
- the GPU material matches its CPU mirror.

Column collapse against the laboratory fits (`tests/terrain_collapse.rs`, ignored):
- **Heights** agree in every case.
- **Runout with Mohr–Coulomb:**
  - a = 3: on Lube's and Warnett's fits (2.74 against 2.75–2.77);
  - a = 1: 70% of Lube's;
  - a = 0.5: 60%, a quasi-static cone at φ. The Drucker–Prager cone ran every case
    short.
- **The open gap** below a ≈ 1.7 is how strongly basal friction stops the thin
  spreading front. Lowering the bed friction to 0.35 closes it, but the
  laboratory runouts are nearly independent of friction (Lube et al. 2005). It is
  documented, not tuned.

**T3 (two-way rigid coupling)** is `WgpuSim::step_with_terrain`, explicit and
staggered, all on the GPU:
1. Once per frame, the rigid colliders overlapping the region are listed in its block
   cells. Their bounds are padded by their motion over the frame, and
   `TerrainConfig::max_coupled` and 8 per cell are limits that error.
2. Each substep holds every grid node inside a collider (signed distance; boxes and
   hulls by their face planes) to the body's surface velocity, as a separating
   Coulomb contact, and records the node's momentum change.
3. One workgroup per body sums the reaction in a fixed order. It moves the body's
   proxy velocity by that reaction (over its mass and inertia) plus gravity.
4. The frame's total becomes one `LINEAR_ANGULAR` impulse record, which the rigid
   step applies at its start.

Without the proxy, a body held at its frame-start velocity drove the sand for the
whole frame and was thrown up 0.45 m; with it, a 54 kg box set on 27 cm of sand
rests 2.6 mm in, at 0.2 mm/s. A box striking a sand block conserves the pair's
momentum to 2e-5. The design question of T3 (does resting stability need an
implicit AVBD coupling block?) is answered no for this case.

**The contact exchange (fixed 2026-10-25).** Step 2 first held each node to an
immovable body, then pushed the body with the whole reaction. That overshoots
once the sand in contact outweighs about twice the body: a 4 kg box was thrown
up at 136 m/s.

Now each substep runs in this order:
1. The grid records the holding impulse.
2. `mpm_couple_gather` sums it with the contact nodes' mass m.
3. Body and nodes (`mpm_couple_apply`, before G2P) take the share Me/(Me + m).

For translation the share gives exactly their common velocity (perfectly
inelastic, momentum conserved), stable at any mass ratio. Me is the effective
mass at the farthest reach of the body's bounds, a lower bound over its contacts,
so spin cannot overshoot either. A 1 kg box now rests on sand
(`a_light_box_rests_on_sand_whatever_the_mass_ratio`).

**Bodies sunk into the sand at bake time (the user's rule, 2026-10-25).** A
heightfield cannot hold a buried body. Before the rule, the bake left sand above
a sunk ball's centre and the ball fell through the ground.

Now `mpm_bake_bin` (`mpmBakeColumn`) moves any particle lying above part of a
coupled dynamic body to just outside the body's footprint, straight away from
its centre. The body is left in a pit as deep as it went, and the volume is
kept. The dropped ball rests in its pit to within 1.3 cm of its depth in the
sand. With the splat bake, the moved particle is splatted at its new place,
which lies one sample spacing past the footprint's half diagonal (half a
spacing before), so the four samples it reaches stay outside it.

**Activation by strikes: a GPU look-ahead.** The user chose this over hit events,
which reach the host 1–3 frames after the rigid step has already stopped the
body on rigid ground.
- After each step, `terrain_watch.slang` marches every awake dynamic collider's
  ballistic path against the watched heightfields
  (`World::terrain_watch_shape`).
- It reports the most energetic strike that passes `TerrainDef::strike`'s
  speed and energy and lies far enough ahead to act on.
- The host opens a region before the next step. The region is sized by the
  scaling of spheres dropped into dry sand (Uehara et al. 2003, penetration and
  crater diameter) plus the region margin, and held until the strike has passed.
- Strikes seen too late are counted (`terrain_missed_strikes`).
- A region also stays open while any dynamic body coupled to it still moves.

Result: a 0.2 m steel ball dropped 1 m opens its region while 0.31 m above the
sand and penetrates 0.115 m, against Uehara's 0.194 m (0.59×, at 6 cells per
body).

Not yet:
- coupling for bodies that rotate fast within a frame (poses are the frame's start);
- performance (about 1,900 dispatches per 60 Hz frame at physical stiffness).

## T4 status (2026-10-24)

**Sources.** `engine-terrain::source` predicts the crater by Holsapple's
π-scaling (gravity and strength regimes, complex-crater rim) and builds the
Maxwell Z-model excavation field. Its magnitude is fitted to Housen &
Holsapple's (2011) launch speeds, with Z = 1/μ by default, which is the Z for
which a steady Z-flow reproduces point-source ejecta scaling. `mpm_source` adds
that field to every particle of the region on the GPU. From there the MPM
computes everything: excavation, ejection, rim uplift and collapse. The
prediction only sizes the region.

**Ejecta.** Material leaving the region's box becomes a ballistic parcel.
- A parcel that lands inside the box re-enters as a particle.
- One that lands outside deposits its volume quanta on the four heightfield
  samples around where it lands, split bilinearly in whole quanta (as the bake).
- One that leaves the field's footprint counts as escaped.
- `airborne` counts the parcels still in flight.
- The ledger closes exactly: initial volume = field + particles + airborne + escaped.

**Two bed bugs a resting region exposed.** Sand activated with no source must
stay put. It did not: after 0.5 s, 147k parcels had been thrown and the window
stood 4–7 cm lower. Two causes:
- **The bed beside the window.** Under the unactivated ground next to the window,
  the bed used that column's surface normal (up), so grid nodes could slide
  sideways freely and the sand squeezed out under the rigid ground. Now:
  - inside the window, the bed is the lowered sample's floor;
  - under the ground around the window, it is a wall facing the window
    (`mpmBedNormal`).
- **The SVD's trailing 2×2 rotation.** It took √(h² + 4q²) with h = 0 exactly
  whenever singular values repeat (a near-rotation F, which is sand at rest). A
  1e-23 off-diagonal squares to 0 in f32, so t = 2q/0 and the stress was NaN.
  The rotations now take the hypotenuse scaled to the larger term. The captured F
  is a regression case on the CPU and the GPU.

With both fixed, a resting region is the identity: no height change, no parcels,
0.001 m/s residual.

**World.**
- `World::enable_terrain(&TerrainDef)`, then one of:
  - `terrain_impact` (π-scaled region plus Z-model source);
  - `terrain_explode` (buried charge);
  - `terrain_open` (a box, no source, for heavy bodies).
- Every `World::step` encodes the region's coupled substeps ahead of the rigid
  step, in the same submission.
- `Frame::terrain` carries the region's counters; a limit or CFL flag is an error
  of that frame.
- A region bakes on its own once its top speed has stayed under `settle_speed`,
  with nothing airborne, for `settle_frames` frames. Baking freezes whatever creep
  remains, so `settle_speed` is the region's rest tolerance, an explicit setting.
- The bake's heights ride back in a later frame's readback to the shape's host
  geometry. A host edit of that shape in between reads them first (blocking),
  and an edit while a region is open on it is refused.
- Each frame's substeps are bounded by a speed bound, which the GPU checks:
  1.5 × the last reported speed, plus what gravity can add over the readback lag,
  plus `speed_headroom` for bodies entering faster than the sand.

**The crater against π-scaling** (`tests/terrain_crater.rs`, ignored; a 5 mm
aluminium sphere at 2 km/s into dry sand). Prediction: transient radius 0.147 m,
depth 0.037 m, final rim 0.191 m.

| Run | Apparent radius (m) | Deepest (m) | Rim height (m) | Wall time |
|---|---|---|---|---|
| margin 2.0, 16 cells/R | 0.193 (1.31 R) | 0.083 | 0.029 at 0.26 m | 730 s |
| margin 0.6, 16 cells/R | 0.184 | 0.092 | 0.029 (at the box edge) | 127 s |
| margin 0.6, 24 cells/R | 0.190 | 0.092 | 0.025 | 644 s |
| margin 0.6, 8 cells/R | 0.129 | 0.057 | 0.009 | 10 s |

- From margin 0.6 to 2.0 the radius moves 5%. From 16 to 24 cells per radius it
  moves 3%, and the deepest point not at all. At 8 cells the source is
  under-resolved and the crater comes out about 30% smaller.
- The converged crater (16 cells/R, margin 2):
  - apparent radius 1.31× the transient one, 1% from the predicted final rim radius;
  - depth over diameter 0.17, between Holsapple's transient 0.125 and Melosh's
    final 0.2;
  - cavity volume 4.1e-3 m³;
  - a rim 2.9 cm high.
- **Open: volume.** The cavity is 3.6× π-scaling's volume (1.16e-3 m³, Kr = 1.4,
  Kd = 0.35): the radius is 1.31× and the depth 2.25×. The field carries 9.5% of
  the impactor's kinetic energy (269 J, mostly within a few impactor radii),
  against about 3 J to lift the cavity. `terrain_crater` asserts the factor-2
  volume band and fails there; the radius, shape, rim and ledger checks pass.
- **Bounded source (2026-10-25, user-approved).** The field now moves only
  the transient crater (`ZModel::extent` = R about the impact point; zmodel.rs
  "Extent"). It keeps 266 of 268 J. (This note first read the field's 1.26·m·U as a
  momentum-transfer β; that was wrong, see "A momentum error" below. Since
  2026-10-29 the field is bounded where it carries exactly m·U, 0.825 R.)
  At margin 2 and 16 cells/R the cavity is 3.08× (from 3.57×), the radius
  1.25×, the rim 3.2 cm. Still open: what remains follows the field's strength
  at crater scale (α, fitted to Housen & Holsapple's launch speeds) and how long
  the impulsive flow keeps displacing ground, not its far field.
- **Volume diagnosis (2026-10-25): not compaction, not the bed.** Measured with
  the test's env-gated probes (`CRATER_PROBE`, `CRATER_SPLIT`, `CRATER_COLUMN`,
  `CRATER_BED`, `CRATER_PHI`, `CRATER_Z`, `CRATER_CORE`). The baseline is
  margin 0.6 (3.44×) unless stated.
  - **No compression to cap.**
    - The elastic J never falls below 0.978, so p ≤ 0.9 MPa (K = 42 MPa). The
      elastic volumetric energy peaks at 2.1 J.
    - The field's energy is dissipated, not stored: 265 J becomes 156 J after
      the first substep (P2G averages a core under two cells wide) and 72 J by
      0.4 ms.
    - A cap at dry sand's crush pressures (MPa and up) would hardly engage.
      It could act only on the early penetration of the floor, whose
      mass-deficit cavity (a "virtual bake" of the particles' rest volume per
      column) is 2.4e-4 m³ at 2 ms, 6% of the final.
  - **The cavity forms over the formation time.**
    - The virtual-bake cavity grows at a steady 6e-5 m³/ms from 2 ms to T
      (83 ms), then holds: 4.13e-3 m³ at 0.1 s and 4.14e-3 m³ baked (margin 2).
    - Nearly all of it is uplift inside the box; 9e-5 m³ escapes.
    - The particle floor keeps sinking after T (4.5 → 8.3 cm on the axis,
      70–300 ms). This is a column of separated material (log Jp up to 2.9)
      settling, and it leaves the baked volume unchanged.
  - **The bed is not the cause.** A bed at 4 R instead of 1.6 R gives 3.24×.
    The field fills the box with u ∝ r^−Z, so its momentum grows with the box
    (−7.1 and −10.9 kg·m/s, against the impactor's 2.83), but the volume
    barely follows it.
  - **Not the near-field energy, nor friction.**

    | Change | Volume / π-scaling |
    |---|---|
    | Core 3h (field 107 J) | 3.13× |
    | Core 4h (field 58 J) | 2.80× |
    | φ = 35° | 3.18× |
    | Z = 2.7 | 2.92× |

    A 4.7× cut in field energy moves the volume 19%. The excess follows the
    field's strength at the crater's scale: α, fitted to launch speeds from
    the core's edge to R. It also follows how long that flow keeps
    displacing ground before gravity and friction stop it.
  - **That is the source's design, so it is open and reported, not changed.**
    Re-run on the current coupling at margin 2: 3.57× (radius 0.193 m, centre
    depth 0.064 m). No material change is adopted, and the measurements above
    are not adopted either.
- **Volume cause found (2026-10-29): the source counts the excavation push
  twice.** Measured with new env-gated diagnostics in `terrain_crater`
  (`CRATER_LAUNCH`, `CRATER_ALPHA`, `CRATER_EXTENT`, `CRATER_A`; a `summary:`
  line now prints before the asserts). Baseline: margin 0.6, 16 cells/R,
  2.79× (radius 1.19 R, deepest 7.7 cm).
  - **The scaling exponents are right; only the constant is off.** Impactor
    radius 2.5, 5 and 11.6 mm at 2 km/s, two decades of energy (353 J to 35 kJ):

    | a | Transient R | Apparent R | Cavity / π-scaling |
    |---|---|---|---|
    | 2.5 mm | 0.083 m | 1.13 R | 2.68× |
    | 5 mm | 0.147 m | 1.19 R | 2.79× |
    | 11.6 mm | 0.296 m | 1.19 R | 2.76× |

    The log-log slope of volume against energy is 0.836, against π-scaling's
    0.830. For radius it is 0.288, against 0.277. The excess is a constant
    2.7–2.8×, the same at every size.
  - **The emergent launch speeds are too fast at crater scale.** The source
    sets Housen & Holsapple's launch speed v(x) on all ground within R at
    t = 0. `CRATER_LAUNCH=0.5` records each top-cell particle's speed when it
    first stands half a cell above the original surface, binned by its home
    distance x, and compares it with v(x)·(1 − x/1.3R)^0.3 (HH 2011 with the
    rim factor):

    | x / R | 0.26 | 0.36 | 0.48 | 0.65 | 0.87 | 1.18 |
    |---|---|---|---|---|---|---|
    | Field bounded at R (current) | 1.20× | 1.78× | 2.59× | 3.48× | 4.29× | 5.21× |
    | Field bounded at 0.3 R | 0.96× | 1.29× | 1.63× | 1.84× | 1.75× | 2.02× |
    | Field bounded at 0.15 R | 0.30× | 0.37× | 0.43× | 0.38× | 0.24× | — |

    HH's v(x) is the speed with which ground at x *leaves*, after the flow
    from inside x has pushed it. The source gives every x that speed at once,
    and the MPM then also carries the inner ground's momentum outward, which
    pushes the outer ground again. So at crater scale the ground leaves
    2–5× faster than the law says. (Inside 0.2 R the speeds measured are
    below the law: the first substep's P2G averages the core, as noted
    above.)
  - **Volume follows the extent, not the energy.** All the field's energy lies
    inside 0.3 R (244 of 266 J), yet the volume swings 7× with the extent:

    | Field extent | Net down momentum | Cavity / π-scaling | Apparent R |
    |---|---|---|---|
    | 1.0 R (current) | 1.26 m·U | 2.79× | 1.19 R |
    | 0.8 R | 0.98 m·U | 2.61× | 1.13 R |
    | 0.3 R | 0.16 m·U | 1.40× | 0.94 R |
    | 0.15 R | — | 0.41× | 0.56 R |

    The α sweep agrees: α × 0.5 gives radius 0.129 m and depth 5.7 cm;
    α × 0.25 gives 0.083 m and 4.2 cm. That is V ∝ α^1.2, so the 22 J outside
    0.3 R sets the crater.
  - **A momentum error (fixed, see below).** The field bounded at R carries 1.26 m·U
    net downward: 6.79 kg·m/s down and 3.23 up, against the impactor's 2.83.
    The earlier note read this as a momentum-transfer β of 1.26. But β is the
    *remaining* target's momentum over m·U. The whole target, ejecta
    included, must carry exactly m·U, so the field breaks momentum
    conservation by 26%. Bounding at 0.8 R conserves it, and that alone
    takes the volume from 2.79× to 2.61×.
  - **Shape: deeper for its width, at every strength.** Depth over apparent
    radius is 0.43–0.56 in every run above, against Holsapple's
    Kd/Kr = 0.25. The profile has a flat floor out to about 0.3 R and walls
    at about 31°, the sand's repose (φ = 30°). So at a given radius the bowl
    holds about 1.7× Holsapple's volume; the rest is the radius (1.19³ ≈
    1.7). Laboratory sand craters are often quoted near d/D ≈ 0.2 (this
    crater: 0.22), deeper than Kd/Kr implies, so this part may be in
    Holsapple's shape constants, not the MPM. It is not resolved here.
  - **Adopted (2026-10-29, decided): the field conserves momentum.** An
    impact's field is now bounded where its net downward momentum, the
    field's own integral (`ZModel::net_down_momentum`, a midpoint quadrature
    in r and the angle from the vertical), reaches the impactor's m·U, and
    never beyond R (`ZModel::bounded_by_momentum`; zmodel.rs "Momentum"). The
    bound comes from the field's definition and Newton, not from the volume.
    For the 5 mm / 2 km/s sand test it is 0.825 R; a direct Cartesian sum
    over the bounded field checks the momentum (`source.rs`,
    `the_bounded_field_carries_exactly_the_impactors_momentum`). Explosions
    keep their bound at R: a buried charge brings no momentum of its own, so
    the impact rule has nothing to conserve there, and that case is open.

    | Margin 0.6, 16 cells/R | Before | After |
    |---|---|---|
    | Cavity / π-scaling | 2.79× | 2.62× |
    | Apparent radius | 1.19 R | 1.13 R |
    | Deepest | 7.7 cm | 7.6 cm |
    | Rim | 2.85 cm at 0.221 m | 2.84 cm at 0.221 m |
    | Axis ridge, mean | | −0.6 mm (passes) |

  - **Decided against: calibrating the source to the emergent launch
    speeds.** It would fit the extent or α to reach the volume, which is
    tuning to the answer, and it would pull the apparent radius under the
    1.13–1.19 R that passes now.
  - **Literature check on the shape (decided: a quick check only).**
    - Holsapple (2022, §3.4 and Table 2) takes simple craters as a fixed bowl
      with R = Kr·V^(1/3) and depth Kd·V^(1/3); for dry sand Kr = 1.4 and
      Kd = 0.35, d/D = 0.125. The formulas are relative to the excavation
      crater: the rim is placed separately at 1.3 times the excavation
      diameter. So R and d are the excavation crater's, at the pre-impact
      surface, not rim-to-floor. The test measures the same thing: the
      cavity below the pre-impact surface and the depth from it.
    - Laboratory and simple lunar bowls are usually quoted near d/D ≈ 0.2
      (Pike 1974 for bowl craters; impacts into sand near 0.2, falling
      slightly as the impactor gets wider; Daly, Barnouin et al., NASA NTRS
      20200001923). The MPM's 0.22 sits there. Holsapple's Kd/Kr for sand is
      shallower than that.
    - The volume band therefore rests on matching definitions: below the
      pre-impact surface on both sides. It is left as it is. The remaining
      factor splits into the shape (depth over radius 0.46 against
      Kd/Kr = 0.25, about 1.8×) and the radius (1.13³ ≈ 1.4×).
    - Whether Holsapple's sand shape constants or the MPM's bowl is off
      needs measured laboratory profiles, which were not found. Open.

    `terrain_crater` keeps the factor-2 volume assertion, so it still fails
    there (ignored). The radius, shape, rim, axis and ledger checks pass.
- The ejecta blanket thins faster than McGetchin's r⁻³ (about r⁻¹⁰ between 1.4
  and 1.9 rim radii); not yet understood.

**Cost and stiffness (measured, not adopted).** At physical stiffness (E = 50 MPa,
c_p ≈ 200 m/s) a 60 Hz frame needs about 11.5/dx substeps (dx in m), plus the
flow-speed term. The centimetre crater above runs 1,250 substeps over 0.6–2.4 M
particles: about 1,500× slower than real time.

Softer sand, against the margin-0.6, 16-cells/R baseline:

| E | Radius | Depth | Rim height | Wall time |
|---|---|---|---|---|
| 5 MPa | +10% | −13% | −16% | 2.8× faster |
| 0.5 MPa | +5% | −37% | −52% | 6.5× faster |

It is lossy, and it is not adopted. Large events are real-time at physical
stiffness: a crater of radius 450 m at 16 cells/R has 28 m cells, so 1–2
substeps.

## Browser and demo (2026-10-25)

**One plan.** `engine_terrain::plan::TerrainPlan` holds the terrain pool's host
state (limits, `MpmParams`, the material's wave speed). It turns each call into
host writes plus the dispatches of one compute pass:
- activation, the Z-model source, uncoupled and coupled frames of substeps,
  the deposits, and the bake;
- a coupled substep runs `mpm_couple_gather` and `mpm_couple_apply` between
  `mpm_grid` and `mpm_g2p`.

`WgpuSim` replays it, and so does the browser. `RegionClock` is the open
region's frame schedule: speed-bound substeps, and settling with the coupled
bodies and a strike's hold. The refactor is bit-identical. The ignored
fingerprints `engine-webgpu/tests/terrain_fingerprint.rs` and
`engine-world/tests/terrain_fingerprint.rs` match the host before it, through
load, activate, source, uncoupled and coupled frames, and the bake.

**Browser.** `engine-vehicle-wasm`'s `TerrainStep` is the plan plus the region
lifecycle `engine-world` runs natively (π-scaled impact and explosion regions,
coupled frames, the counters' check, the bake once settled). It returns
commands as words. `terrain-host.mjs` allocates bindings 37–51 from the
plan's sizes, builds the kernels, executes the words and copies the counters
back. Nothing else is in JavaScript.
- `mpm_grid` binds 9 storage buffers. `world-host.mjs`'s `buildKernels` fails
  by name when the device allows fewer per stage.
- `tests/terrain_plan.test.mjs` (node) checks the decoded plan and the
  lifecycle through wasm.

**`impact.html`.** A stony 8 m impactor at 20 km/s (308 kt) into dry sand at
E = 50 MPa:
- the transient radius is 151 m, cells are 9.4 m, the margin is 1.0 R, and the
  region holds 0.91 M particles;
- views: contact, ground flow, a cross-section through the impact point
  (particles coloured by starting depth), a tracked particle followed into its
  ejecta parcel, material returning, the baked crater, and a monster truck
  driving from the crater floor up its wall.

Frame cost in Chrome on the Mac it was built on. These are GPU-bound wall times
from `frames()`, rendering included; another session was using the GPU.

| Sim time | Substeps | Mean (ms) | Max (ms) |
|---|---|---|---|
| strike frame (activation, source) | 10 | 94 | |
| 0–0.5 s | 9–10 | 51 | 62 |
| 0.5–1 s | 6–9 | 43 | 51 |
| 1–2 s | 4–6 | 28 | 36 |
| 2–3 s | 3–4 | 21 | 26 |
| 3–6 s | 2–3 | 16.3 | 21 |
| 6–12 s | 2 | 14.7 | 16 |

- Real time is reached about 3 s in. Before that, the material's flow speed
  (850 m/s at the source, from `max_speed`) sets the substeps, not stiffness.
- At margin 0.6 the region holds 0.60 M particles and its 2-substep frames take
  10.6 ms. But the rim then forms at the box edge, as a square rampart (as in
  the crater table above).
- A 1 kt buried charge (R 47 m, 2.9 m cells) needs 6 substeps (36 ms) at
  t = 2.5 s: smaller events are not real-time.

**The crater** (default event, baked at 12.9 s; azimuthal mean of the height
change):
- rim crest at 250 m, 1.28× the predicted final rim of 196 m;
- deepest point 75 m, 2× the predicted transient depth of 38 m;
- the same excess as the crater validation above.

**Open points.**
- **The region does not settle** (with the rigid ground). In a margin-0.6 run,
  20 s after the strike, half the particles still move faster than 1 m/s, deep
  ones at the far side of the box included, and `max_speed` decays only from
  5.7 m/s (15 s) to 2.4 m/s (35 s). Diagnosed and fixed by the absorbing ground
  (below): elastic waves trapped between the rigid walls, most of them set off
  by the activation itself.
  - So the rest tolerance (5 cm/s) is not reached soon. The demo bakes at a set
    time (`World::terrain_close` semantics) once nothing is in flight, and
    says so in its UI.
- **The bake roughened the surface** (the splat bake below, with its home
  fade, smooths it).
  Neighbouring 9.4 m samples of the baked rim differed by up to about 10 m,
  while the particle surface just before the bake was smooth.
  - The bake binned every particle of a column about 300 m deep by its nearest
    sample. A small sideways shift of deep material therefore moved a lot of
    height between samples.
  - Deposits were whole parcels on one sample: 1.2 m each at these cells.
  - The truck could not cross the baked rim, so it starts on the floor.
- **Ejecta escape the field.** About 0.43 M m³ leaves the 6 R field's footprint
  by the bake (the core moves at 1.4 km/s).
- **Not in the browser yet:** the strike look-ahead (`terrain_watch`, binding 52)
  and soil ruts. `engine-world/src/terrain.rs` keeps its own copy of the frame
  schedule; it should switch to `RegionClock`.

**The demo on the absorbing ground (2026-10-25).** `impact.html` bakes once
settled by default. Its rest tolerance is 1% of the crater's gravity speed
√(gR) (0.38 m/s for the default 151 m crater; `?settle=` overrides it).
- The region settles and bakes by itself at t ≈ 26 s.
- Held open to t = 200 s at 5 cm/s, its top speed only fell to 0.077 m/s, and
  baking then gave the same crater: depth 101.6 m against 101.9 m, the same
  apparent radius (198 m) and rim (297 m, 22.5 m high). What still moves is a
  few particles rattling, not flow, so the scaled tolerance loses nothing.
- The inner wall stands at about 35° near the floor and 28° higher up. A
  crater wall is concave in plan, and that supports a steeper face than the
  planar slope test (which relaxes to 29°).
- The demo truck, starting on the floor, stalls at the foot of the 35° wall
  (its tyres hold to about 27°): the crater is a trap.
- Open: along the axis row through the impact (z = 0) the baked inner wall
  sits 4–8 m below its neighbours, a groove the truck drives in.

## Rendering and showcase (2026-10-25)

Drawn as particles, an open region looks like a cloud of sand. A game wants
ground that deforms, and only the material that leaves it flying as debris.

**The live surface** (`slang/kernels/terrain_surface.slang`, planned by
`engine_terrain::surface::SurfacePlan`). This is a render-only pass after each
coupled frame. Its only output is binding 53; nothing in the simulation reads
it. The kernels:
- `terrain_surface_clear` empties an occupancy grid: window samples × the
  region's cells in y.
- `terrain_surface_shift` measures the absorbing walls' give, as
  `mpm_bake_shift` does, into its own buffer.
- `terrain_surface_bin` splats every particle as `mpm_bake_bin` does: the
  tent at home for the share at rest, P2G's B-spline for the share the MPM
  moved, in the unactivated ground's frame. In y it uses a tent over two
  layers, in fixed point.
- `terrain_surface_resolve` gives one height per window sample: the sample's
  own height (the lowered bed plus any parcels deposited there), plus the
  material attached to the bed. A column ends at the first two layers in a
  row that hold less than 30% of a full cell. One sparse layer is a column's
  loose top, not a gap.
- `terrain_surface_debris` lists every particle more than a layer above that
  surface, and every particle in flight. It counts them into indirect draw
  arguments, so a renderer draws only the debris.

*In flight* means faster than a fall of one layer, √(2 g dy): 7 m/s on
2.5 m cells. That sand has been thrown, even before a gap opens under it.
- It stays out of the surface and is drawn as debris. At crater scale the
  ejecta curtain and the excavation flow never rise two layers clear of the
  ground: on 2.5 m cells, a 15 m crater's ejecta at √(gR) ≈ 12 m/s climb
  about 7 m.
- A settled region has nothing that fast, so at the bake the surface and the
  heightfield are the same splat of the same quanta.
- `SurfacePlan::with_ejecta_fall` (`surface_ejecta_fall` in the wasm
  settings) sets the fall height; 0 counts only what a gap separates.
- `surface_cells` (default 2²¹) caps samples × layers, and fails by name when
  a region needs more.

**The switch at the bake.** `terrain-render.mjs`'s `surfaceSwitchError`
compares the last live surface with the baked heightfield. Results:

| Case | rms | worst |
|---|---|---|
| `impact.html`, 2 m meteor, 3 m cells, timed bake at 10 s | 0.15 m (5% of a cell) | 0.57 m |
| showcase, 2.5 m cells, timed bakes at 4.5 and 5.1 s | 0.07 m | 0.57 m |

- The difference is the residual motion and the fixed point.
- The bake also pushes sand off a buried body; this pass does not.
- The first resolve stood on the bed instead of each sample's own height, and
  ended a column at one sparse layer: 0.39 m rms, with parcels deposited on
  window samples and loose top rows missing.

**Hosts.** `terrain-host.mjs` runs the pass with `createTerrain(…,
{ surface: true })`. `terrain-render.mjs` holds the WGSL that draws one
ground: the heightfield's records outside the window, the live surface inside
it, and `debrisParticle(k)` for the debris list.
- `impact.html` draws the surface by default. G or the Surface button turns
  it off. The cross-section and the cut always show the particles (the
  strata).
- Every open area in one surface pass, each with its own view and patch
  ("Several open areas at once").

**`showcase.html`** (hub card "Meteor run"):
- A third-person chase camera behind the dune buggy (`buggy-render.mjs`'s
  mesh), on an autopilot round a stadium track (straights of 140 m, turns of
  58 m), at 45 km/h. Physics and render run at 1/120 s.
- Three meteors strike inside the track: a 0.5 m stone at 20 km/s, giving a
  15.1 m transient crater and a 19.7 m final rim. The second lands on the
  first crater's rim, the third at the centre of the turn the car drives
  round.
- `cells_per_radius` 6 at margin 1.0: 2.52 m cells, a 60 m box of 97,592
  particles, 3 substeps a frame after the first 0.05 s.
- Placement: a strike sits as close beside the track as its region's box
  allows (6 m clear), so the car passes 18 m from the rim while the crater
  still moves. A car must never enter an open window: its wheels' rays would
  find the lowered bed.
- Cosmetic only, from the strike's time and point:
  - the fireball and its smoke trail on approach;
  - the flash, and the light it casts;
  - camera shake;
  - a shockwave ring, a dust curtain, a base surge and a rising column;
  - a dust trail behind each piece of debris, and a hot crater floor.
- Debris is the engine's list, drawn as rocks sized from each particle's
  volume.
- The terrain settings sit in one `TERRAIN` object, where the engine's game
  profile can replace them.
- One region at a time. A strike due while a region is open first bakes it,
  once nothing is in flight (a timed bake, `World::terrain_close`).
  - Here those regions are 4.5 and 5.1 s old: the timed-bake jumps above.
  - Once the engine allows several regions (`config.max_regions > 1`), the
    second strike lands at 3.2 s, while the first crater still moves.
- `record()` steps 1/120 s frames and encodes every other one: WebCodecs H.264
  (Annex B) with explicit timestamps, from the canvas texture's pixels copied
  in the frame's submission. `timings()` gives GPU timestamps for the
  terrain, surface and render passes, and each frame's wall time from
  encode to GPU completion.

Frame cost at 1/120 s, rendering every frame, in Chrome on the Mac it was
built on. Other agents were compiling and running GPU tests at the same time.
Medians, with p95 in brackets:

| | 1280×720 | 1920×1080 |
|---|---|---|
| frame, region open (wall) | 7.9 ms (15.5) | 9.0 ms (11.0) |
| frame, no region (wall) | 4.0 ms (5.3) | 6.8 ms (8.9) |
| terrain pass (GPU) | 1.25 ms (3.7) | 1.25 ms (1.6) |
| surface pass (GPU) | 0.20 ms (0.46) | 0.20 ms (0.26) |
| render pass (GPU, 4× MSAA) | 2.1 ms (2.9) | 2.8 ms (3.7) |

- The terrain is within its 2 ms share of the 8.3 ms budget.
- The strike frame takes 27 substeps (about 12 ms of terrain, once).
- The frame's wall time also holds the rigid step and the vehicle, untimed,
  and a wait for the GPU every frame: at 1080p it sits at or just over 8.3 ms.
- The dust was 9 ms of render before its colour moved to the vertex shader,
  its noise to one octave, and its plume to 700 sprites a strike.

### Second pass: wheels on the sand, bigger craters, two more videos

**Wheels on the live sand.** While a region is open, the window's samples sit
at the bed, under the sand. A wheel's ray found that bed, so a car could not
enter an open region.
- `terrain_surface_publish` (the surface pass, with
  `SurfacePlan::with_patch`) keeps a second static heightfield, the *surface
  patch*, on the live surface each frame:
  - it places the patch over the window, with a ring of the field's own
    samples round it;
  - it gives the patch the surface's heights and height range;
  - at the bake (`SurfacePlan::retire`), it sinks the patch 10 km down.
- Wheel rays and rigid contacts meet the patch, so they meet the sand where it
  is now, still settling. A test checks the plan and the limit ("surface
  patch samples").
- This is not the design asked for, which was a branch inside the vehicle
  step's wheel query. That branch would give `wheel_query` two more bindings
  (37 and 53) on every host. The Metal host binds by its own table, and the
  browser creates the vehicle's bind groups before the terrain's buffers
  exist. The patch leaves the vehicle step, `vehicle_parity` and every host
  as they were. With no region open it is out of reach.
- It also takes rigid contacts (a chassis or debris body on the sand), not
  only rays.
- Wheels push the sand since 2026-10-29: see "Wheels on open sand" below.
  The chassis is coupled as any body is.

**A car in a region.** The dune buggy has 187 collider hulls on its 17 parts.
- Coupling lists colliders per block cell. `MPM_CELL_COLLIDERS` was 8; it is
  now 256.
  - The couple buffer holds 1 + 256 words per block: 8.4 MB at 8,192 blocks.
  - A node only visits the colliders its cell lists. With the buggy in a
    region, the terrain pass took 1.57 ms against 1.25 ms.
- The showcase sets `max_coupled` to 256.
- Native `WgpuSim` now allocates a placeholder at binding 53. It builds every
  exported kernel and binds them by reflection, so the surface kernels had
  broken `terrain_couple` there since they were added.
  - Note: `wgsl_export` only warns when a kernel fails to compile, and leaves
    the earlier WGSL in place.

**Look.** The craters were 15 m and read flat from the chase camera.
- A 1.0 m meteor gives R = 26.9 m and a 35 m final rim. At 6 cells per radius
  that is 4.48 m cells, about 44k particles, and 2 substeps a frame.
- The ground's vertices carry sky occlusion and curvature, measured 1, 2 and
  4 samples out in eight directions. With them the ground shows:
  - a dark, red dug-out bowl;
  - a lit lip;
  - a darker, clumped ejecta blanket, with cosmetic rays of fine dust out to
    4.5 R.
- Rocks are stretched along their velocity (about two frames' blur).
- Dust comes mainly from the debris list: three lagged puffs behind each
  moving piece, and a landing puff where a piece comes down near the ground.
  The cosmetic curtain, base surge and column are thinner and faster, and
  every puff is lit as a ball from the sun.
- The track's straights sit just outside the regions' block lattice, so a
  strike beside them comes as close as its box allows.

**Scenes** (`showcase.html?scene=`):
- Every scene runs the engine's game profile (`TerrainProfile::Game`) since
  2026-10-30; `?profile=physical` keeps the validation profile.
- `drive` (the default since 2026-10-29): a barrage of 11 strikes in 20 s on
  both sides of the road (`side`: inside or outside the loop, the ground
  flat 75 m outside it too).
  - Four put the track just inside the rim, so the car drives up over the
    crest and down (its height peaks at 5.5–6.3 m); three land 0.4 s before
    the car passes and two 1 s after it.
  - Re-placed for the game profile, whose rims stand higher and whose bowls
    are steeper (the car peaked at 6.6 m over them, against 5.0 m):
    - the first behind-the-car strike lands 9 m outside the road, not 4 m
      inside it. Where it was, it landed on the car's line and stopped it
      at 1 km/h;
    - the first two over-the-rim strikes cut 1 m inside the rim, not 2–3 m:
      deeper, the car ran down the bowl's inner wall 17 m off the road.
  - The car then holds the road to 5 m at 41–67 km/h after its run-up.
  - The areas overlap and grow into one; `max_regions` is 8 and the patches
    are 128 samples, both explicit.
  - `trackAt` runs 0.75 s later here, as measured, so near misses land where
    the car is.
- `drive1`: the earlier cut. Strikes land 3.6 s before the car passes, their rims
  3 m off the track. The car rides each rim's flank and blanket while the
  region is open: its height rose 2.6–3.1 m beside each crater, at 60 km/h.
- `pass`. The first cut: strikes kept clear of the track.
- `bts`. Behind the scenes, at 1/300 s steps, in two viewports from one
  camera and one state.
  - Left, the engine's view: the heightfield with its lowered bed, every
    particle coloured by speed, parcels in cyan, the region's box in yellow,
    the window in blue, and a cut facing the camera.
  - Right, the game's view.
  - `record({ pace })` sets the speed: real time, 5× slow motion through each
    strike, 2× through the bake between.

**Videos** (scratchpad, not committed):
- `showcase-v2.mp4`: `drive`, 1920×1080, 60 fps, 20 s.
- `behind-the-scenes.mp4`: 2560×720, 60 fps, 21 s. Labels and the simulated
  time are burned in by ffmpeg from the recording's time sidecar.

**Frame cost** (1280×720, rendering every frame). Other agents shared the
machine; the 1080p recording ran 7× slower than earlier ones.

| Scene and step | Terrain (GPU) | Surface | Render | Frame (wall) |
|---|---|---|---|---|
| `drive`, 1/120 s, region open | 1.57 ms (p95 2.29) | 0.26 ms | 1.25 ms | 7.7 ms (p95 12.4) |
| `drive`, 1/120 s, no region | — | — | — | 3.7 ms |
| `bts`, 1/300 s, region open, both views at 2560×720 | 0.72 ms, 1 substep | 0.33 ms | 2.1 ms | 7.7 ms simulated only, 10.2 ms rendered |

### With overlapping strikes (multi-hit)

The surface pass shows one area: the first open one, set by the plan's area
slot. In both showcase scenes every later strike grows that area, so only
one is ever open:
- drive: windows of 25×25, then 49×25, then 61×33 samples;
- bts: 21×25, then 29×29.

Two separate open areas (far apart, both still moving) showed only the
first; the second's window drew as its lowered bed and its wheels' rays
found that bed. The pass now shows every open area, each with a patch of its
own ("Several open areas at once" below).

**The walls' give on a grown window.** The per-frame shift read each edge
column's activation range. A grown window's older edge samples have none, so
their give read as zero, and material against the walls folded onto the last
row: a straight ridge a sample wide, 1.3 m on the drive scene's grown window.
- `terrain_surface_shift` now adds every particle made at an edge column (by
  its home) into that edge's accumulators, in fixed point.
  `terrain_surface_shift_mean` divides.
- The edge rows now fall off smoothly (0.41 then 0.20 m, where the last row
  had been 1.56 m).

**Open, in the bake (growth).** A grown area keeps a ridge on the ring of
samples just outside the first window: about 2.3 m on the bts scene's first
window ring, at j = 65 and i = 83.
- It is in the baked heightfield too (2.69 m baked against 2.67 m live), and
  the particles' tops there are smooth.
- The cause: the first area's edge columns gave outward by about 0.15 m,
  their particles carrying that as displacement from home. The grown area
  then activated the ring's full columns over the same ground. The column
  just outside holds about 270 quanta (5%) more than its own particles.
- So the volume the walls pushed into the ground is counted twice. It is a
  growth-activation matter, not the surface pass's.

**Fixed (2026-10-29): the ground beyond the old walls carries their give.**
- Before a grown area activates, each absorbing area it takes over gets a
  pass of its own (`Opened::pre`, `mpm_grow_shift`).
  - The pass measures the give of that area's edge columns with the bake's
    own edge lists.
  - It stores each side's give in the old columns' records.
- The grown area then activates its new columns beyond each side that
  stopped being a wall (`mpmGrowGive` in `mpm_activate_emit`).
  - The ground's displacement continues outward from the old face: the
    side's measured give falls linearly to zero at the grown area's far face
    on that side. That is the uniform strain the bake already takes between
    opposite sides (the user's decision).
  - Each particle's home stays where it was laid, so it carries that
    displacement as material the MPM moved, as the old edge columns do.
- Nothing is counted twice: the old material stays where the walls let it
  go, and the new columns stand clear of it. Both sides of the old edge bake
  as moved material under one field.
- What does not change:
  - sides that stay walls keep their give and face displacements
    (`MPM_DOMAIN_KEEP` bits);
  - volume is untouched (the ledger is exact);
  - an ungrown area's activation is as before (activate-then-bake is the
    identity).
- The new particles' F carries no lateral strain from the give (about
  give/D, D the distance to the far face): they start unstressed laterally,
  like every activation.
- The first version (9282acfc) took the give out of the old material
  instead. That left the bake's two kernels meeting at the old edge: moved
  edge columns spilling an eighth across on P2G's B-spline, onto new columns
  at rest on the tent.
- Hosts run the passes before the activation: `WgpuSim` and `engine-metal`
  `terrain_open_area`; the wasm plan, a block each.
- **Test** (`terrain_bake_axis`,
  `growing_an_area_over_its_walls_leaves_no_ridge_on_the_ring`):
  - the demo's event at 8 cells/R, settled (walls give up to 16 m at the
    faces);
  - grown by a sourceless box past its +x side, then baked;
  - the ring (the first samples past the old +x edge, 37 rows) against its
    row neighbours, and against the bake's own noise: the same measure on
    the columns 2 to 4 samples either side.

  | | Ring over the ungrown bake (mean, worst) | Ring over its row neighbours |
  |---|---|---|
  | counted twice | +24.4 m, +31.8 m | +27.0 m |
  | give taken out of the old material (9282acfc) | +4.0 m, +5.9 m | +4.8 m |
  | give carried into the new columns | +2.0 m, +3.3 m | −0.5 m (noise 2.6 m) |

  The grown bake stands higher than the ungrown one beyond the old edge (the
  +2.0 m mean). The ground the walls compressed comes back as a rise across
  the new columns, instead of folding onto the old edge.
- **A side's give applies only across its own span.** Beyond its ends, in a
  corner past two old sides, no old wall stood. At first both sides' gives
  applied there. In bts that pulled the corner ring apart into a 1.5 m trench
  along j = 65 for i > 82, and along i = 83 for j < 66.
- **`showcase.html?scene=bts`, baked at 21.3 s** (Chrome). Each ring sample
  over the mean of its neighbours across the ring:

  | Line | Mean | Worst |
  |---|---|---|
  | ring j = 65, across the old window's span | −0.41 m | 1.29 m |
  | ring j = 65, past it (the corner) | +0.35 m | 1.17 m |
  | ring i = 83, across the old window's span | −0.12 m | 0.77 m |
  | ring i = 83, past it (the corner) | +0.11 m | 0.89 m |
  | rows 60, 62, 68, 70 (the bake's own noise) | up to +0.33 m | up to 3.1 m |
  | columns 78, 80, 86, 88 (the bake's own noise) | up to −0.21 m | up to 1.71 m |

  The 2.3 m rails are gone.

**A strike on a moving rim** takes the live surface's height there, read back
0.25 s before contact: 3.58 m against the heightfield's 0.

Frame cost with overlapping strikes, drive scene, 1/120 s, 1280×720,
rendering every frame:
- whole frame (wall) 6.4 ms median, p95 7.2;
- terrain 1.44 ms, surface 0.39 ms, render 0.98 ms.

The 1080p recording (rendering every other frame): 6.2 ms median, p95 8.3;
107 of 2,580 frames over 8.33 ms.

Behind the scenes, per 1/300 s frame:
- terrain 0.59 ms, 1 substep;
- surface 0.26 ms;
- render of both views at 2560×720, 1.8 ms;
- 3.8 ms per frame simulated only, 7.3 ms rendered.

### Several open areas at once (2026-10-29)

Two strikes far apart open two areas, both moving at once. The surface pass
and the wheels' patch used to show only the first. Now every open area is
drawn and drivable.

**Views.** Each open area is a *view* of the surface buffer, one record per
area slot in a table after the header (`SURFACE_VIEW_WORD`, 16 words × 16
slots). A record holds:
- the area's flags (open, absorbing);
- its layers;
- where its heights, walls' give and occupancy start in their sections;
- its patch collider;
- its window, bed and layer height;
- its first thread in each per-sample dispatch.

The plan lays the open areas back to back in every section.

**One pass for all of them.** The plan dispatches each kernel once, over every
view, instead of once per area.
- The per-sample kernels (clear, the give's mean, resolve, publish) find their
  view by scanning the table for the range that holds their thread. That is at
  most 16 records.
- The per-particle kernels (shift, bin, debris) take the area the particle is
  in (`mpmAreaOf`) and splat into that view only. The bake owns material the
  same way.
- The splat, the give and the resolve are unchanged for each area.
- `tsSurfaceSample`, which the debris test and the patches read, looks across
  every open window. A patch whose ring or extent reaches another open window
  takes that window's live heights, not its lowered bed.

**Patches.** There is one patch per area slot:
- `SurfacePlan::with_patches`;
- `TerrainStep::set_surface_patches` in the wasm host;
- `createTerrain(…, { patches })` in the browser.

Patch k follows slot k: it sits on that area's live sand while the area is
open, and sinks 10 km down while it is not. Every patch is written every
frame, so a slot whose area baked sinks while the others stay open. Because
the patch is tied to its slot, it never jumps to another area when one area
closes, so contacts against it are not handed between areas.

**Limits** (explicit, each failing by name; nothing degrades silently):
- `surface max_cells`: window samples × layers, summed over the open areas.
- `surface edge samples`: 6 (ni + nj) summed, against the shift section.
- `max_columns`: the plan's own pool.
- `surface patches`:
  - more than 16 patches fails;
  - with patches given, an area open in a slot that has no patch fails.
  Give one patch per `max_regions`. `showcase.html` makes `max_regions` of
  them.
- `surface patch samples`: a patch smaller than its window plus a ring fails.

**Hosts.**
- `terrain-render.mjs`'s `groundHeight` looks a sample up in every open view
  (`surfViewAt`). `surfWindow()` and `surfLayer()` give the first open view,
  for single-area tools: the behind-the-scenes box and window lines still draw
  the first area only.
- `surfaceSwitchError` takes the slots that baked (`terrain.bakedSlots`), so
  a partial bake is measured against the baked windows only.
- Native: `WgpuSim::terrain_surface_enable` and `terrain_surface` run the
  pass, for tests.
- The constants come from Rust (`T.SURFACE_VIEW_*`). `SURFACE_WINDOW_WORD`,
  `SurfacePlan::frame_area` and the wasm `surface_area` are gone. `frame` and
  `surface` show every open area.

**Checks.**
- `engine-webgpu` `tests/terrain_surface.rs`: the crater validation's 5 mm
  sphere at 2 km/s, at x = ±0.9 m (transient radius 0.147 m, 0.018 m layers).
  The two impacts open two separate areas, with two patches.

  | Check | Result |
  |---|---|
  | Each window right after activation, against the flat ground | 1.8e-8 m |
  | Each window after 40 frames and landing | craters 0.0505 m deep, equal to 1e-4 |
  | Each patch, against its window's surface plus the field's ring | 0 (bit-equal) |
  | The field between the two windows | the flat ground, unchanged |
  | Area 0 bakes, then the pass runs again | its view closes, patch 0 sinks, area 1's surface and patch unchanged (bit-equal) |
  | Each window's last live surface, against its bake (worst sample) | 1.3e-3 and 1.5e-3 m (7–8% of a layer: the residual motion, as for one area) |
  | Both areas baked | every view closed, both patches sunk |

- `engine-terrain` `surface` tests:
  - the views' sections and dispatch ranges are laid back to back;
  - `max_cells` is summed over the areas;
  - each slot's patch and publish range;
  - "surface patches" fails by name.
- `tests/terrain_plan.test.mjs`: through the wasm, two far impacts give two
  open views with their windows and patches. One patch for two areas fails
  by name.
- `showcase.html?scene=drive&t3=2.2&pass3=19.6`: the third strike lands at
  the far turn while the first crater still moves, so two areas are open at
  t = 3.7–4.5 s (windows 25 × 25 at i0 = 58 and 94). Both craters draw live,
  from their own views. The two windows' surfaces:
  - first: −12.95 to 4.56 m;
  - second: −13.59 to 2.99 m.

  At 4.5 s the second strike lands between them and grows one area over
  both. `impact.html` (one area) is unchanged.

### Wheels on open sand (2026-10-29)

Before this, wheels rode an open area's surface patch, but the sand under them
felt nothing: no ruts, and the car's weight went into the static patch. Now
each wheel standing on the live sand presses its contact force into the MPM,
so the sand ruts and berms under it and carries the car's load.

**How** (`terrain_mpm.slang`, "Wheel loads"; three kernels in the coupled
frame, `TerrainPlan::coupled`):
- **`mpm_wheel_loads`**, once per coupled frame, one thread. It walks the
  vehicle pool (binding 31) in wheel order, so runs agree bit for bit. It
  lists each wheel that, in the rigid step before:
  - was in contact, enabled and attached;
  - stood on a static heightfield other than the area's bed (the area's
    surface patch);
  - had its contact one cell inside an open area's box.

  Each listed wheel carries the force it pressed on the ground there:
  F = −(n·Fn − t·fL − s·fT). That is the reaction of the ground's force on the
  chassis, from the wheel's latched normal and friction forces.
- **`mpm_wheel_mass`**, per substep after P2G, one thread per load. It sums
  the material's mass at the contact: M = Σ w·m over the contact's B-spline
  stencil, counting only nodes that carry velocity.
- **`mpm_grid`** (`mpmWheelLoads`, its own function behind an early return)
  moves each stencil node by F·dt·w/M. That is ahead of the bed and walls, so
  they take what the material passes on.
  - Σ m·Δv = F·dt exactly: the sand takes F·dt per substep, and the frame's
    substeps the impulse the chassis took over the rigid step.
  - The load is shared by mass (Δv the same at every node), not by weight
    alone. A weight-only share would put the load's full share on a nearly
    empty surface node.
- **Timing.** Explicit and staggered, like the rigid coupling. The forces are
  the rigid step's from the frame before, so momentum balances over the pair
  of frames, forward only.
- **The wheel follows the sand down** through the surface patch, which is
  rebuilt from the particles every frame.
- **The contact is one point.** It is spread by P2G's own kernel: a footprint
  about a cell across. No contact-patch model; the tyre's width and length
  are not used.

**Limits.** `TerrainConfig::max_wheel_loads` (default 64; 0 turns it off and
dispatches no wheel kernels). More wheels on open sand than that set
`MPM_OVERFLOW_WHEELS`, and the frame fails naming `max_wheel_loads`. The
records sit in binding 49 after the coupling slots and the face points (three
float4 each). The counter `MPM_C_WHEELS` (word 14) holds the frame's count;
`TerrainStepReport::wheels` reports it. An uncoupled `step` clears it, so
stale loads never press.
- `mpm_grid` still binds 9 storage buffers, `mpm_wheel_loads` 7 and
  `mpm_wheel_mass` 5.
- Metal binds 31 from its own table (`terrain_parity` 7/7, `vehicle_parity`
  4/4 pass). CUDA needs no entry: both kernels are fixed dispatches.

**Measured** (`tests/terrain_wheels.rs`):
- **Set-up.** The sedan (1322 kg, 12 969 N), game profile, physical sand,
  0.15 m cells, over a 4.8 × 6 × 1.8 m area of 55 796 particles, at 1/60 s
  frames and 52 substeps.
- **At rest on the sand:**
  - all four wheels are listed;
  - sinkage 1.9 cm at 1 s and 5.2 cm at 10 s;
  - the rate falls from 21 mm/s in the first second to 0.7 mm/s in the tenth;
  - the wheels carry 13 079 N of the car's 12 969 N;
  - the sand's momentum moved 43 N·s in the last second, against the weight's
    12 969 N·s.

  The sand's stress holds the load. The slow tail is the sand under the
  tyres compacting and flowing out as the bearing area grows. A Bekker
  dry-sand wheel would sink deeper (centimetres to decimetres).
- **Driving through** (throttle 0.3, from rest outside the area):
  - all four wheels pressed at once;
  - after the bake, the ground under the tracks is 4.5 mm lower on average
    (8.5 mm at most), and the window clear of them is +0.7 mm;
  - the berms rise to 7.6 mm;
  - the field's volume is 0 → 0 m³: the rut's sand is all still there, to
    the quanta.
- **Coasting** at 1–2 m/s, the sand's resistance stops the car inside the
  area. At 4 m/s it crosses, with a 1 mm rut.
- **Outside the area.** A car parked on the unactivated ground lists no load,
  and `max_wheel_loads` 0 lists none.
- **Cost** (`wheel_load_cost`, native wgpu on Metal, encoder timestamps):
  - the terrain frame takes 0.274 / 0.281 ms with the four loads, against
    0.268 / 0.268 ms off;
  - so +0.006–0.013 ms a frame of 52 substeps, about 0.2 µs per substep;
  - the single-thread list is a few µs a frame.

**In the browser.** `showcase.html?scene=drive`, stepped by hand:
- the HUD's "wheels on sand" shows 2–4 of the buggy's wheels on the open
  craters from t = 3 s to 12 s, with no frame errors;
- `impact.html` builds the two kernels.
- **The showcase's 4.48 m cells resolve no rut.** The sand's mass at a contact
  is about 10⁵ kg, so a wheel's 3 kN moves it by micrometres. A rut needs
  cells near the tyre's width, as in the test's 0.15 m.

**Open:**
- the tyre's contact patch: width, and length from its sinkage;
- wheels over baked ground: the bake ends the rut, and baked ground is the
  heightfield's, where the Bekker rut tier applies to soil fields;
- sphere-cast wheels are not tested.

### Playable: `showcase.html?scene=play` (2026-10-30)

You drive the buggy; meteors land around you without end. It is the showcase's
renderer, terrain and car, with the schedule filled as you play.

**Controls.**
- W A S D or the arrows drive; Space is the handbrake; R restarts.
- On a phone: on-screen ◀ ▶ ▲ ▼ buttons, shown on a coarse pointer or with
  `?touch=1`.
- Click or tap the ground to drop a meteor there (a ray from the eye marched
  over the heightfield). F (☄ on a phone) drops one ahead of the car.

**The director** places each strike from where the car will be at contact:
- beside the line, the centre RIM + 2–12 m off it, a rim to ride;
- ahead on the line, 30–60 m (steer round the fresh crater);
- behind; or far out, as scenery;
- onto the car itself, with a 2.4 s warning (`lead`), 8% of strikes at first
  and up to 30%.

It strikes every 2.6 s at first, down to every 0.9 s after about two minutes.
Each strike shows a red ring on the ground while it comes in. Runs are seeded
(`?seed=`).

**A run ends** on any of these:
- a contact within 0.5 R of the car (horizontally);
- rolled over for 2.5 s;
- 8 s on the throttle while the car moved under 5 m (sand that holds it at
  about 1 m/s counts as stuck).

A contact within RIM + 12 m that you survive is a near miss. The best time is
kept in the browser.

**Endless.** The engine keeps it going ("Endless areas", below): the page
runs with `settle_rule: "local"`, so an area settles on its own sand while the
car drives over it and the barrage's ejecta fly elsewhere, and the pool takes
back a baked area's slots while others are open. The page closes nothing and
never pauses the director. (Until 2026-10-30 it did: timed closes at
`closeafter`/`maxage`, the director kept out of old areas, and lulls until
every area had baked. `?settle=strict` keeps the strict rule, with none of
those, to compare.)

**Limits checked before a strike is placed**, refused with a message rather
than an engine error:
- the area slots (`max_regions` 12, two kept for yours);
- the pool: its held slots (`live` + `airborne`, below) and its parcels;
- the wheel patch: a strike whose growth would make a window wider than
  `PN` − 8 samples is not placed, since a larger one fails by name ("surface
  patch samples").

A strike that can no longer open by contact burns up.

The ground is a 1.2 km basin (`?half=600`) of low dunes, rimmed by hills.

### Endless areas (2026-10-30)

Two things kept a run of steady strikes from going on by itself. Both are
engine changes now. Neither moves a live particle or changes a result where
it does not apply.

**1. Settling while you drive over it: `TerrainDef::settle_rule`.** An area
bakes once its own sand has stayed under `settle_speed` for `settle_frames`
frames, with no hold on. What else keeps it open is the rule:

| | `Strict` (the default) | `Local` |
|---|---|---|
| Parcels in flight | any, anywhere in the domain | those whose ballistic path can still reach the area's box |
| Bodies | any dynamic body in the box that moves | those whose contacts hold the area's material |

- **Parcels (`MPM_C_AREA_AIR + slot`).** For each parcel in flight after a
  substep (`mpmParcelReach`, in `mpm_parcels` and where G2P throws one), every
  open area whose box, padded by a block, its path crosses before it falls
  below the box's floor counts it. The path is the exact ballistic one, with no
  drag, as the parcel flies. A parcel that lands sooner on the heightfield is
  still counted, which only keeps the area open longer. So under `Local` a
  parcel that cannot reach an area cannot change it, and the area bakes on
  what it holds.
- **Bodies (`MPM_C_AREA_TOUCH + slot`).** `mpm_couple_gather` already sums
  each body's contact nodes. A node with mass now marks its area, and the
  body's surface speed (linear plus spin over its reach, as the strict word)
  goes to each area it touched. The buggy's chassis rides above the sand on
  its wheels: under `Strict` it holds every area it is over; under `Local`
  only the sand under it moving does (the wheel loads move it, and that shows
  in the area's own speed).
- **A body falling onto quiet sand** touches nothing until it lands, so
  `Local` could bake the area just before. That is the strike look-ahead's job
  (`World::terrain_watch_shape`): a predicted strike past its thresholds holds
  the area until it lands, and one under them meets rigid ground, as it would
  anywhere else.
- **Hosts:** `TerrainDef::settle_rule`; in the browser `settle_rule: "local"`;
  the C ABI `RbTerrainDef::settle_local`. The report keeps both views:
  `area_body_speed` and `area_touch_speed`, `airborne` and `area_airborne`
  (`TerrainStepReport::settle_view`).

**2. Slots back while other areas are open (pool reuse).** A baked area's
particles, and a landed parcel's source slot, stay in the pool as dead slots:
P2G sums particles in slot order, so the pool never moves a live one. Before,
those slots came back only when the last area baked. Two GPU passes now take
them back sooner (`slang/kernels/terrain_mpm.slang`, "pool reuse"):

- **Trim, at every bake that leaves areas open** (`TerrainPlan::pool_trim`):
  - `mpm_pool_reserve` marks the slot each parcel in flight will come back to;
  - `mpm_pool_top` finds the last slot still held (live or marked);
  - `mpm_pool_trim` ends the pool there.

  Live particles keep their slots and their order, so the open areas step
  exactly as before. The bake's dithers count slots from the area's first, so
  they are unchanged too.
- **Packing the parcels, at the same bakes:** `mpm_parcels_pack` (one
  workgroup, a scan in slot order) copies the parcels still in flight to a
  staging half of the parcel buffer, and `mpm_parcels_unpack` copies them back
  to the front. Nothing reads a parcel's slot: landing, deposits (seeded by the
  particle it left) and the slot it returns to go by its own record. So packing
  changes no result. `max_parcels` now bounds the parcels at once between such
  bakes, not over the domain's life. The parcel binding doubled for the
  staging copy (96 B per parcel).
- **Free slots, at an activation past the pool's end** (`TerrainPlan::pool_holes`,
  only when the append would pass `max_particles`):
  - `mpm_activate_scan` sees it will not fit and hands its count to the pool
    passes instead of failing;
  - `mpm_pool_holes_count`, `_scan` and `_list` list the free slots in slot
    order, in the sort's scratch (between frames);
  - `mpm_pool_finish` takes them first and appends the rest. It fails by name
    (`max_particles`) only if the free slots plus the room left are too few.

  An activation that fits runs as before. One that reuses slots puts its
  particles among others: its P2G order is its own particles' order (slot
  order of the list, then the tail), but the bake's dithers count from other
  slots. Its crater differs from the same impact alone only in that rounding.
- **`live` (`MPM_C_LIVE`):** the live particles. Activations add their count;
  the bake, G2P's throws and returning parcels keep it. The pool's held slots
  are `live` + `airborne`, so a host can tell whether a strike fits.
  `particles` is now the pool's extent.

**Found on the way:** a parcel coming back over an open area landed only at the
heightfield. An open area lowers that to its bed, so the parcel fell through the
material still there and came back as a particle at the bed, under the sand.
Ejecta falling back into a crater were buried at its floor. Now a parcel comes
back as a particle as soon as it enters an open area's box (one cell in), and
the MPM lands it on the material.

**Validation** (`engine-webgpu` tests, on Metal):

| Test | Result |
|---|---|
| `terrain_multi` `parcels_count_only_for_the_areas_they_can_reach` | Ejecta thrown from one area: every one of them in that area's word, 0 in the far one's, frame by frame, while they fly and land; ledger exact. |
| `a_bake_with_areas_open_trims_the_pools_dead_tail` | The second of two areas bakes: the pool ends at the first's last live slot (60,900 of 121,800 slots back), bit-identical particles; ledger exact. |
| `a_bake_mid_flight_keeps_the_parcels_slots_and_packs_them` | One area bakes with 27,138 parcels aloft: the same parcels (bits) fly on, packed to the front, each with the slot it left inside the trimmed pool; ledger exact. |
| `an_activation_past_the_pools_end_takes_the_free_slots` | A third area (70,644 particles) in a pool of 152,250 holding 121,800 slots, half of them freed by a bake: it fits, the pool grows to 131,544; ledger exact. Its crater against the same impact alone: cavity 6.7215 → 6.7267 × 10⁻⁴ m³ (+0.08%), deepest point 0.3 mm deeper, samples within 1.9 mm (the dithers' rounding). |
| `ejecta_falling_back_into_an_open_area_land_on_its_sand_not_its_bed` | The crater's middle thrown straight up: the 3,572 that flew come back no lower than 0.016 m, on sand topping out at 0.16 m (bed −0.19 m). With the old landing they came back at −0.193 m, the bed. |
| `terrain_couple` `a_falling_box_touches_the_sand_only_when_it_lands` | A box falling 0.4 m through an area's air counts as moving (strict word) but not as touching (local word) until it reaches the sand. |

**Not fixed, found on the way:** in `terrain_multi`'s scene, material thrown
out of an area by hand (most of its sand at 4 m/s) leaves a torn remnant.
Stepped on with that area open, a few of the remnant's particles outrun the
CFL bound about 13 frames later, on main as well. The existing tests close the
area mid-flight, and the new one does too.

## The splat bake (2026-10-25)

The user approved a bilinear splat for the bake and the parcels' deposits.
`mpm_bake_bin` and `mpm_parcels` (`mpmParcelDeposit`) split each particle's or
landed parcel's quanta between the heightfield samples around it. Material
the MPM moved, and every landed parcel, now goes with P2G's quadratic
B-spline over nine samples rather than the tent over four; see "Moved
material: P2G's kernel" below. Each share goes where the whole particle went
before:
- a window sample's column;
- the deposit accumulator, for a sample outside the window;
- escaped, for a sample off the field.

The max-height header is kept by the same raise kernels as before.

**Where a particle is splatted: its position, less a fading share of its rest
offset** (the bake from the home fade until the axis-trench fix below).
Activation lays a sample's particles on a lattice across the
sample's cell. A splat at each particle's own position would spread unmoved
off-centre particles onto the neighbours, so activate-then-bake would no
longer be the identity.
- `mpm_activate_emit` records where it emitted each particle, x₀ in the
  heightfield's frame (`MpmParticle` `c[1][3]`, `c[2][3]`, flag
  `MPM_PARTICLE_HOME`). G2P carries the record.
- The bake splats at u = x/spacing − w·o, where:
  - o is x₀'s offset from its home sample (in spacings);
  - w = max(0, 1 − |x − x₀| / (fade · spacing));
  - fade is `TerrainConfig::bake_home_fade`, params `limits.y`, default one
    sample spacing.
- Unmoved (w = 1), the reference is exactly the home sample, so the identity
  holds.
- From one spacing of displacement on (w = 0), the particle splats where it
  is. Material that has flowed no longer keeps its rest lattice: near a
  crater's centre one column is stretched over several samples. There the
  particle's own position says best where its volume is.
- Particles loaded by the host and parcels landing back in the region carry
  no home, so they splat at their position.
- First version, before the fade: the correction was kept however far a
  particle moved, u = home + (x − x₀)/spacing. A rigid shift was then exact,
  but stretched columns clumped into ridges; see "The home fade" below.
- The alternative was to splat each particle's own footprint (its lattice
  cell) by its overlap with the sample cells. It also keeps the identity, but
  its kernel is a box of the lattice spacing (half a sample by default), not
  the tent of one sample: less smoothing. With one particle per sample axis
  the two are the same.

**Whole quanta.** The split rounds x first, then each half along z, so the
four parts are whole quanta that sum exactly to the particle's. Each split
rounds with a dither from the particle's slot (floor(share + r), r uniform in
[0, 1), from a PCG hash). Its expected value is the exact share, and a zero
weight gets exactly nothing.
- Rounding half up was tried first. It errs the same way for every particle
  moved alike, so a column's errors add up.
- In the rigid-shift test (0.3, 0.55 of a sample, 5 particles per sample axis)
  the worst height error was 1.2 cm, and the centroid moved 0.547 samples
  instead of 0.55. With the dither: 0.65 mm, and the centroid is within 1e-4.
- At the demo's scale (about 240 particles a column, 1.8 cm quanta) rounding
  half up could leave about 2 m steps between neighbours.

**Buried bodies.** `mpmBakeColumn` now pushes covering sand one sample spacing
past the body footprint's half diagonal, so the four samples the splat reaches
stay outside the footprint. Before, it pushed half a spacing, which was the
reach of nearest-sample binning.

**Tests** (`tests/terrain_mpm.rs`):
- activate-then-bake with no motion is still the identity to 1e-5 m;
- `material_moved_rigidly_bakes_as_the_splat_defines`: rigid shifts of
  (0.3, 0.55) samples (inside the fade) and (1.3, −0.55) samples (beyond it),
  made with `WgpuSim::terrain_overwrite_particles`, a diagnostic.
  - The heights equal the splat's definition within one quantum per
    contributing particle. The worst error is 0.77 mm.
  - The volume is exact.
  - The baked centroid is the splat's to 1e-4 samples. It follows the shift
    to 2e-3 samples: a column's `quanta mod n` extra quanta go to its first
    particles, which breaks the offsets' symmetry slightly.
- `an_opened_layer_bakes_flat_without_axis_ridges`: a flat layer is opened by
  the area-keeping radial map r′ = √(r² + r₀²), which leaves it exactly as
  thick around the hole.
  - With the fade, it bakes back with an error of 0.7% rms (2.2% max), and
    the axis lines stand 1.6% of the thickness above their neighbours.
  - With the correction kept, those are 4.7%, 18% and 13.5%.

The volume and ledger tests pass unchanged: flow and bake, ejecta, and the
crater's ledger.

**Smoothness**: the crater validation at `CRATER_CPR=8` (8 samples per
transient radius). Roughness is measured within 3 R; "residual" is the height
change minus its ring mean, which leaves out the crater's own slopes.

| | Nearest sample | Splat |
|---|---|---|
| neighbour step, max | 34.5 mm | 26.0 mm |
| neighbour step, rms | 5.28 mm | 3.91 mm |
| residual, rms | 3.29 mm | 2.26 mm |
| residual, max | 23.3 mm | 17.4 mm |
| apparent radius / transient | 0.875 (fails the 0.9 band) | 1.00 |
| depth | 27.6 mm | 34.2 mm |
| rim | 11.2 mm at 0.184 m | 7.0 mm at 0.165 m |

- The ring-mean profile is smooth now. Before, it was spiky (11.2 mm at ring
  10, 6.5 mm at ring 11, 7.6 mm at ring 13) and cut off at ring 16.
- It now has a tail of a few tenths of a millimetre out to ring 27. That is the
  far field's small outward displacement, which nearest-sample binning
  rounded away.
- The simulation itself is unchanged; only the bake differs.

**Fingerprints.** `terrain_fingerprint` `fingerprint_impact_coupled_and_bake`
changes:
- particles `a0f6c33cccae1a95` → `7bd9e041c0c4bdf5` (the home record);
- heights `73fb3fb0af79c17c` → `79c9061c160a5ae5`;
- bodies `81af402f01161578` unchanged.

The loaded cube's fingerprint (no activation, no bake) is unchanged.

**In the browser demo** (`impact.html?bakeAt=1000`, default event). The
particles were read back at t = 12.9 s, when the top speed was still 9.4 m/s.
The same state was then baked four ways in JavaScript (in floats), and the GPU
splat bake was compared with them:
- The GPU bake matches the JavaScript home-plus-displacement splat within
  0.26 m rms. The difference is the dithered whole-quanta rounding.
- Nearest-sample binning is what the old bake did, so it gives "before" on the
  same state.

Neighbour steps, max / rms (m), by distance from the impact point (R = 151 m):

| | < 0.5 R | 0.5–1 R | 1–1.8 R (rim) | 1.8–2.2 R (window edge) |
|---|---|---|---|---|
| nearest sample (old bake) | 24.8 / 7.66 | 31.8 / 9.00 | 48.3 / 11.99 | 32.6 / 8.44 |
| splat at home + displacement (this bake) | 21.8 / 8.27 | 16.6 / 4.27 | 18.1 / 4.74 | 11.5 / 3.11 |
| splat at the particle's position | 11.3 / 5.50 | 7.6 / 3.56 | 17.9 / 5.75 | 39.1 / 10.71 |

- The rim and the window edge are 2.5–2.7× smoother (rms).
- **The crater centre was not** (fixed by the home fade below). Within 0.5 R, home plus displacement
  leaves ridges along the two axes through the impact point. On the z = 0 row
  they stand 12–15 m above the rows beside it, and nearest-sample binning
  leaves the same ridges.
  - Near the centre, material spread several samples outward. A column's
    particles there end up spread apart, but the bake puts them back at their
    lattice offsets from home, so each column's volume clumps around where its
    centre went.
  - The rows on the axes do not move sideways, so they keep their own volume
    while the rows beside them empty.
  - Splatting at the particle's own position has no ridges, but it breaks the
    identity (it is the roughest at the window edge).
  - The elastic F cannot measure this spread, because plastic flow is not in
    it.
- **The truck** spawned on the z = 0 row at x = −12 m. It sat high-centred on
  that ridge and did not move.
- One alternative was tried in JavaScript first: fade the home correction
  with the displacement. It was adopted (below).

**The home fade (the user's decision, 2026-10-25).** The correction fades
linearly from full at rest to none at `bake_home_fade` sample spacings of
displacement. The default is one spacing, the smallest move that can take a
particle across to another sample.
- Kept as before: activate-then-bake is exactly the identity, volume is exact,
  and the pushed sand of a buried body splats at its new place.
- Given up: a rigid shift is now exact only at zero.

Crater validation at `CRATER_CPR=8` (after the bounded source was merged; same
simulation, only the bake differs). Rough is the neighbour step, max / rms
(mm); "residual" is the height change minus its ring mean:

| | correction kept (fade 10⁹) | fade 1 |
|---|---|---|
| axis ridge within 0.5 R, mean / max | 17.5 / 21.7 mm | −2.7 / 3.6 mm |
| rough < 0.5 R | 23.7 / 14.6 | 18.1 / 9.0 |
| rough 0.5–1 R | 15.3 / 7.2 | 12.3 / 6.1 |
| rough 1–3 R | 6.1 / 0.65 | 4.2 / 0.58 |
| residual rms, within 3 R | 1.84 | 1.08 |
| depth | 28.6 mm | 36.9 mm |

The validation now asserts that the axes stand no higher than the centre's
residual rms.

Browser demo:
- On the earlier demo state (t = 12.9 s, baked in JavaScript), the fade
  changes the rms neighbour step (m) by zone as follows: centre 8.27 → 5.73,
  0.5–1 R 4.27 → 3.66, rim 4.74 → 5.77, window edge 3.11 → 3.03. The rim
  gives back a little; the centre loses its ridges.
- The GPU bake of the current demo (default, baked at t = 13.45 s with the
  bounded source) measures, max / rms (m): centre 15.4 / 6.41,
  0.5–1 R 9.8 / 3.44, rim 15.8 / 4.65, window edge 10.5 / 2.53. The axis rows
  stand 4.2 m below their neighbours on average, not above.
- **The truck** now drives off the centre, from x = −12 m to x = −27 m, at up
  to 6.3 m/s. There it stops at the foot of the crater's inner wall, which
  rises 11 m and then 10 m over the next two 9.4 m samples (about 47°).
  - That wall is steeper than sand's angle of repose because the region is
    baked unsettled (top speed 8.6 m/s at the bake; see "The region does not
    settle"). It is the crater's shape, not the bake's roughness.
  - The truck does not reach the rim.

**Fingerprints after the fade** (both include the bounded source merged
before it):
- `engine-webgpu` `fingerprint_impact_coupled_and_bake`: particles
  `c574ae35b3634fb3`, bodies `368fd4be1705937b`, heights `87fc671ed38d6dde`.
  The loaded cube is unchanged (`9aee277fcc35ea31`).
- `engine-world` `fingerprint_world_impact_to_bake`: reports
  `59df3a636ac25733`, heights `df41faaa6898177b`, box `136fde101445dae3`.

**The axis trench, and moved material on P2G's kernel (2026-10-25).** On the
demo, baked once settled on the absorbing ground, the axis row z = 0 lay
4–8 m below the rows at z = ±9.4 m all along the inner wall. The truck drove
in that groove and stopped.

*Where it came from.* `tests/terrain_bake_axis.rs` runs the demo's event
natively and compares the particles just before the bake with the bake. The
axis lines against the mean of the lines beside them, averaged from 2 samples
to R on all four arms:

| | 16 cells/R (demo) | 8 cells/R |
|---|---|---|
| highest particle over each sample | −4.6 m | −8.8 m |
| rest volume, binned by nearest sample | +12.8 m | +10.1 m |
| bilinear bake (the bake until now) | −4.8 m | −5.6 m |
| P2G's quadratic B-spline at the particles | −0.5 m | −1.5 m |
| lines 2–4 samples off the axes, for scale | −0.2 to −1.1 m | −0.3 to −1.6 m |

The particles explain the sign changes. Across the x axis, in bins of an
eighth of a cell, particles sit in sheets half a cell apart: the activation
lattice, mixed between sheets everywhere except at the mirror plane. No
particle crosses a mirror plane of the symmetric event, so there the two
sheets beside the plane stand 0.62 of a cell apart instead of 0.5, with
nothing between them. That is a sub-cell crack: P2G's B-spline, three cells
wide, does not see it, and nothing in the simulation closes it. Deep
material crowds the sheets beside it, so nearest-sample binning puts more on
the axis, while a one-sample tent splits the outer sheets and leaves the axis
short. With the impact a third of a cell off the grid node, the crack follows
the mirror plane between sample rows and the axis rows lose only 0.6 m.

*The fix.* The bake splits each activated particle in two:
- the share w = max(0, 1 − |d| / (fade · s)) is still the heightfield it came
  from, with d its displacement and s the lattice spacing. It goes with the
  tent to its home sample plus its displacement, so activate-then-bake stays
  exact;
- the rest is material the MPM moved. It goes with P2G's quadratic B-spline,
  in sample spacings, over nine samples, at its own position: the mass the
  grid computed with.

Further details:
- Landed parcels deposit with the B-spline too. Sand pushed off a buried
  body keeps the tent, because its place is the rule's, not the flow's.
- Volume stays in whole quanta. A dithered three-way split of x, then z, sums
  exactly.
- `bake_home_fade` is now in lattice spacings, default 0.5: a particle that
  has left its own lattice cell is moved material. One sample spacing (the
  fade before) left a ridge of +2.2 m at 16 cells/R.

After, the GPU bake's axes against their neighbours:
- 16 cells/R: +0.04 m.
- 8 cells/R: −0.59 m.
- In the browser demo (baked at t = 24.9 s), the axes stand −0.03 m on
  average. By x:

  | x | z = 0 | z = ±9.4 |
  |---|---|---|
  | −28 | −97.7 | −98.7 / −97.4 |
  | −57 | −79.4 | −79.0 / −79.3 |
  | −76 | −63.1 | −61.9 / −62.8 |

- The truck now climbs from the floor (x = −12 m, −100.6 m) to x = −63 m,
  30 m up the wall. It stalls there on a stretch rising 8 m per sample
  (about 40°) and slides back. That wall is steeper than sand's angle of
  repose, so the stall is the crater's shape, not the bake.

*Tried, not adopted.* Measuring the home share against the column's mean
displacement, so that a column moved as a block keeps the tent however far
it went, keeps a rigid shift exact.
- It mixes kernels unevenly between neighbouring columns: a column on the
  tent keeps its own volume while taking the B-spline spill of its
  neighbours.
- At 8 cells/R it raised the crater's centre sample 78 mm, from 40 mm deep to
  38 mm high, and failed `engine-world`'s crater and ball tests.

*Tests.*
- `a_crater_bakes_without_a_trench_along_its_axes`, 8 cells/R, about 7 s. It
  fails on the bilinear bake (−5.5 m against lines off the axes within
  ±1.5 m) and passes now.
- `moved_material…`: the rigid-shift test checks the hybrid definition. At
  (0.03, 0.055) samples, part of each particle goes home; at (1.3, −0.55)
  none does. The worst error is 1.3 mm.
- The opened layer bakes flat to 0.2% rms with a 0.1% axis ridge (4.7% and
  13.5% with the home correction kept).
- At `CRATER_CPR=8`, the axis ridge is 0.5 mm.

*Open: the window's edge.* On the absorbing ground the region's edge columns
move outward as blocks, a few metres, onto the unactivated ground beside the
window. The sample just outside the window then stands 65–80 m high and the
edge sample inside 4 m low (to −10 m with the B-spline). Every bake shows it,
nearest-sample binning included (73 m), so it comes from the particles. The
ground outside the window does not move with the absorbing wall's
displacement. In the browser that sample stands 82.6 m high, at x = −293 m.

**Fingerprints after this change:**
- `engine-webgpu` `fingerprint_impact_coupled_and_bake`: particles
  `079220bca4b1a009`, bodies `a5336ecc75ef2ef8`, heights `8cd48a76cd182790`.
  The cube is unchanged.
- `engine-world` `fingerprint_world_impact_to_bake`: reports
  `6484a2b48140af12`, heights `31f1c80605d3d905`, box `136fde101445dae3`.

## Absorbing ground (2026-10-25)

**Why the region rang.** Measured natively on the demo's event at 8 cells per
radius (`tests/terrain_absorb.rs`):
- With no source at all, the activated region oscillates at up to 10–15 m/s,
  and after 10 s half of it is still faster than 1 m/s.
- Activation makes particles with F = I: unstressed ground, which then sags
  under its own weight. A column H = 288 m deep sags ρgH²/(2(λ + 2μ)) ≈ 10 m
  at its surface, at up to g·H/c_p ≈ 14 m/s. The rigid bed and walls reflect
  that wave, and the strike's waves, back into the box forever.
- Starting geostatic but keeping the rigid ground leaves 3 m/s: the rigid bed
  holds only the node layer below the floor, while the stress puts half the
  floor's load on the layer above it.

**The model** (`RegionBoundary::Absorbing`, `InitialStress::Geostatic`;
`mpmAbsorbNode`, `mpm_geostatic`, `mpm_absorb_init`). The ground around an
activated region is an elastic half-space of the region's own material. Its
faces are the floor (the bed plane) and the window's four sides, up to the
unactivated ground's surface. On each face the ground pushes with
t = p0·n − (k_N·u_n + ρ·c_p·v_n)·n − (k_T·u_t + ρ·c_s·v_t):
- **Dashpots** ρ·c_p and ρ·c_s: Lysmer & Kuhlemeyer (1969), exact for plane
  waves at normal incidence.
- **Springs** k_N = (4/3)·μ/R and k_T = (2/3)·μ/R, R the distance to the
  region's centre: the 3D viscous-spring boundary (Liu et al. 2006, after
  Deeks & Randolph 1994). They are the static stiffness of the ground beyond,
  so a change of load (ejecta leaving, a rim deposited) gives a bounded
  displacement instead of an endless creep at Δσ/(ρ·c_p).
- **Geostatic traction** p0: ρ·g·d on the floor and K0·ρ·g·d on the sides,
  K0 = λ/(λ + 2μ). The material starts in that state: the first step after
  activation (which has gravity) sets each particle's F to the uniaxial strain
  of its depth below its column's surface. It is the K0 initial-stress
  procedure of geotechnical dynamic analysis.
- **The ground's strength.** A face whose normal traction would pull lets the
  node go: sand does not stick to the ground. A face whose shear would pass
  the bed friction, μ·t_n, slides at μ·t_n, as the rigid bed does. The
  dashpots model small-strain waves; without the limit, plastic flow along a
  face meets a drag of ρ·c_s·v_t (176 kPa at 1 m/s), far above the ground's
  strength at shallow depth. While a face lets the material go, the ground
  stays where it was.
  - Found on the dropped ball (engine-world `terrain`): without the limit, the
    sand flowing along its shallow region's bed stopped the ball at 0.089 m.
- **Discretisation.** Node i takes ∫ w_i·t dS over the faces (w_i its
  B-spline), lumped: its share of a face is ∫ w_i dS, from the B-spline's
  closed-form integral, and its dashpot acts on its own velocity. That is the force the particles' geostatic stress transmits
  through the face, so the activated state is in equilibrium.
  - A floor half a cell above a node layer splits its load between two layers;
    walls off the node lattice spread over three.
  - In a 1D check, the dashpot on the boundary node alone sent back 2.6% of a
    pulse's energy and the consistent shares 0.13%.
- **Each node is solved implicitly:** v = (m·v* + dt·(f0 − K·u))/(m + dt·C + dt²·K)
  per axis, stable for any dt.
- **The displacement u belongs to the face.** One record per face point (the
  node lattice's columns through the floor and each side), after the coupling
  slots in binding 49 (`TerrainConfig::max_boundary_nodes`, an error past it).
  `mpm_absorb_faces` moves each point once per substep, after G2P, by the
  material's velocity there (the mass-weighted B-spline mean of the node layers
  across the face), while the material presses on it. Every node the face
  reaches springs against that one displacement, implicit in its own velocity.
  - The first version kept a record per node, each integrating its own
    Eulerian velocity. Under the strike's flow the node layers across a side
    drifted metres apart (−5 m and +8.7 m one cell apart at 8 cells per radius),
    and their springs pushed opposite ways across it.
  - With the face's record, engine-world's dropped ball goes 0.118 m deep
    (0.61× Uehara; it was 0.094 m, 0.49×, below the test's band).
  - `mpm_grid` still binds 9 storage buffers; `mpm_absorb_faces` binds 7.
- **Unchanged:** the region's top and the free surface stay free. Contact with
  rigid bodies, host-loaded regions (no window: the field's surface is a contact)
  and the rigid ground's code path are as before.
- **Validation.** The absorbing ground requires the geostatic start and a
  material with stiffness; either missing is an error.

**The bake in the ground's frame (2026-10-25).** The demo's settled crater
(margin 1.5 R, 12 cells/R) baked a ridge one sample wide at the window's edge:
+42 m at x = −378 m natively (+37 m in the browser), the samples on either
side near 0 and 1 m.
- **Cause.** The absorbing sides give, and the material against them moves out
  with them. The window's edge columns end 1.4–1.5 m outward on average (up to
  3.4 m near the surface) at the demo's scale: 50 MPa sand 384 m deep, springs
  for R ≈ 0.4 km. The bake splats each column at home plus its displacement,
  so that shift crosses the window's edge. Beside the window is unactivated
  ground, a heightfield that cannot move, so the share below its surface folds
  back onto the edge sample (`mpmBakeOnto`). Every column passes ~8% of itself
  outward and the edge keeps it: 0.08 × 384 m.
- **Fix** (`mpm_bake_shift`, `mpmGroundShift`). The bake measures positions in
  the unactivated ground's frame. Material can only move out past a side's
  plane by moving the ground with it (it cannot enter the ground), so the
  ground's give at a side is its edge columns' mean outward displacement
  (inward motion leaves the ground where it was). Each particle has that give
  taken out, interpolated linearly between opposite sides: the uniform strain
  an elastic layer between two displaced walls takes.
  - Volume is kept exactly; only where it is splatted changes. The region's
    lateral give comes back as a uniform rise of its columns (about 2 m here)
    instead of a ridge.
  - The faces' own records are not the measure: they hold the grid's velocity
    at the plane, which the dashpot-held node layers there pull to a fifth to
    a third of the material's a cell in.
  - The splat tests that set displacements by hand
    (`material_moved_rigidly_bakes_as_the_splat_defines`,
    `an_opened_layer_bakes_flat_without_axis_ridges`) run on the rigid ground:
    a displacement set at a wall by hand has no ground that gave.
- **Test** (`terrain_bake_axis`, `the_window_edge_bakes_without_a_ridge`, 8
  cells/R): no edge sample may stand over both its neighbours across the edge
  by more than the same measure 2–4 samples in (the bake's noise) plus 1 m.

  | Cells/R | Before: ridge (edge, inside) | After: worst edge excess | Noise inside |
  |---|---|---|---|
  | 8 | 28.7 m (27.8, 1.4) | 2.3 m (edge 2.6, inside 5.0) | 2.9 m |
  | 12 (the demo's) | 41.8 m (42.0, 1.2) | 2.5 m (edge 2.2, inside 3.2) | 3.7 m |

- **The demo** (Chrome, this tree): settled and baked at 20.9 s. Along −x, the
  edge at −378 m is 1.0 m, the ground beyond −1.6 to −0.5 m (the plain's own
  roll); rim crest 20.9 m at 252 m, floor −96.4 m. The truck climbs from the
  floor to x = −189 m (chassis at −0.8 m, about 95 m up the 117 m wall) and
  stops below the crest, where the wall steepens to about 28°.
- **Root cause, open.** Metres of give at 2.5 R come from sand of constant
  E = 50 MPa down to 384 m. Real sand stiffens with confinement (Hardin's
  G ∝ √p: several times stiffer at those depths), which would give a fraction
  of it. That changes the material and its validated results, so it is not
  done here.

**Tests** (`tests/terrain_absorb.rs`):
- **At rest.** One second under gravity, then the bake. The splat bake moves
  whole quanta by the particles' displacement, so it is checked by its exact
  volume and a bound on what that displacement moves (the elastic layer: one
  quantum, 1.25e-4 m; the sand: 0.9 mm). Activate-then-bake with no step is
  still the identity (`terrain_mpm`).

  | Material | Ground | Top speed | After 1 s | Largest move |
  |---|---|---|---|---|
  | elastic, flat | absorbing | 4.0e-4 m/s | 2.2e-5 m/s | 8.6e-6 m |
  | elastic, flat | rigid, unstressed | 7.0e-2 m/s | 2.7e-3 m/s | 1.4e-4 m |
  | sand, mound | absorbing | 1.6e-2 m/s | 5.4e-4 m/s | 7.1e-4 m |
  | sand, mound | rigid, unstressed | 9.2e-2 m/s | 2.0e-3 m/s | 1.8e-3 m |

  The elastic layer (a rock of sand's stiffness that does not crack here) shows
  the ground holds the geostatic state. What sand keeps is its own slow plastic
  creep near the free surface, where confinement vanishes (log Jp grows there),
  on either ground.
- **A plane P-wave at the bed.** A layer 1.4 m deep and 5 m wide, so the side
  walls' effect (not at normal incidence) has not reached the axis. The energy
  left 8 cells after the pulse's tail reached the bed, over the energy arriving:

  | Pulse | Rigid | Absorbing |
  |---|---|---|
  | one sine period, 8 cells, 2 cm/s, elastic | | 1.2% |
  | compressive cos², 6 cells, 1 m/s, elastic | 80% | 2.3% |
  | the same in sand (330 kPa on 22 kPa) | 86% | 1.3% |

  - The rigid bed's 14–20% is the transfers' dissipation on the way back.
  - The sine wavelet stays under the 6.7 cm/s at which its tension would pull
    on the overburden. Neither ground pulls, so a stronger tension phase comes
    back from both.
  - The compressive pulse displaces the layer by ∫v dt. The springs push it
    back up, which a plane wave in a half-space would not do. In a shallower
    layer, with the springs taken out as a measurement, the sand pulse left
    0.6% against 2.4% with them; the sine wavelet's share did not change.
- **The demo's event settles** (the bounded source). The 8 m stony impactor
  at 20 km/s; the share of particles faster than 1 m/s (and than 5 cm/s):

  | Region | After | Rigid, unstressed | Absorbing |
  |---|---|---|---|
  | 8 cells/R, margin 0.6 (97 k particles) | 10 s | 80% (100%) | 0 (36%), top 0.9 m/s |
  | the demo's: 16 cells/R, margin 1 (0.90 M) | 20 s | 53% (99%) | 0 (0.5%), top 0.25 m/s |

  - The first is a test; the second is ignored (a minute).
  - Before the bounded source, at the demo's resolution, nothing was above
    1 m/s from 8 s on. What still moved at 20 s was the crater floor's loose
    material settling (0.2 m/s) and rim deposits sliding off the box's edge,
    so the 5 cm/s rest tolerance is still some seconds away.
- **The crater** (`terrain_crater`, `CRATER_GROUND=absorbing`, margin 2,
  16 cells/R, the bounded source and the splat bake; nothing tuned to it):
  apparent radius 0.184 m (1.25 R), centre 0.061 m deep, deepest 0.079 m, rim
  2.9 cm high at 0.239 m, cavity 3.10× π-scaling's volume (rigid: 3.08×,
  1.25 R, 3.2 cm). The bed and walls hardly change the crater; the volume
  excess is the source's, as diagnosed above. 782 s wall.

**Open points.**
- **The default since 2026-10-25.** `TerrainConfig::default()` is the absorbing
  ground with the geostatic start (`Rigid` / `Unstressed` remain). The dropped
  ball that blocked it was under-resolved at 6 cells per body. At the default
  8 it penetrates 0.117 m on absorbing ground and 0.123 m on rigid ground (0.60× and
  0.63× Uehara), and the test now runs at the default.
- **The walls' top corner.** The unactivated ground's surface beside the window
  is now a rigid contact for the nodes the absorbing faces reach as well
  (`mpmGroundSurfaceContact`). Before, within 1.5 cells of a side's top edge,
  the surface was the soft wall face rather than a rigid contact. Rim material deposited there sinks into the corner at the spring's
  pace: 0.35 m/s at margin 0.6, where the rim forms at the box's edge. The
  rigid ground does no better there: under unactivated ground its normal faces
  the window, so it holds nothing up.
- **The springs' R** is the distance to the region's centre, the source of an
  impact region. For a load right at a face, a rim deposited on a side's top,
  that is softer than the ground's local stiffness.
- **Oblique waves.** Lysmer–Kuhlemeyer is exact only at normal incidence, and
  reflects a few per cent at grazing incidence: the side walls take a plane
  wave travelling along them.
- **Browser.** The browser runs the same plan and kernels (`TerrainStep`'s
  `config.boundary`, `config.initial_stress`); `impact.html` has not been
  rerun in Chrome with the absorbing ground.

## Performance (2026-10-25)

The browser demo's event (an 8 m stony impactor at 20 km/s into dry sand,
R = 151 m, margin 1.5 R, the absorbing ground), natively on the M3 Max
(wgpu on Metal) and in Chrome. GPU times are the kernels' own, summed
(`RUBBLE_TERRAIN_GPU_MS=1`: every dispatch in a timestamped pass of its own;
a pass holding all of a frame's dispatches reads 0 on Metal). Other agents
shared the GPU throughout: a run's spans move by up to 2-3× between
repeats, so the tables give the quietest of three runs and medians.
- `tests/terrain_perf.rs` (`demo_event_frame_cost_and_crater`, ignored): the
  event from strike to bake, each frame's GPU time, the baked crater.
  `RUNS="GAME=1+CPR=12,BED=1.5"` runs settings in turn.
- `tests/terrain_bench.rs` `terrain_demo_profile` (`RUBBLE_GPU_TIMING=1`):
  each kernel's share; `PROFILE_REST=1` how much of the region is at rest.

**Where a substep went** (before, 60 Hz frames, 0.91 M particles, 2.3-2.8 k
blocks): 4.85 ms per substep, whatever the material did. P2G 1.89 ms (39%),
G2P 0.95, payload (return map) 0.80, touch 0.49, scatter 0.30, count 0.27;
the rest under 0.05 each. Every per-particle kernel is bound by memory
traffic: a particle is 128 B, and P2G reads each 80 B payload once per node
it reaches (27×), from L2.

**Lossless changes** (same work order; fingerprints below):
- `mpm_touch` records each particle's base cell (block key, cell in block)
  and `mpm_count` its cell and rank, in `gMpmSorted` after the sorted ids
  (binding 44 is 12 B per particle). `mpm_count` and `mpm_scatter` no longer
  read the particle or the hash: 0.27 → 0.11 and 0.30 → 0.08 ms. The order
  after `mpm_cell_sort` is the same: bit-identical.
- P2G's payload is 64 B (the momentum and the affine matrix packed): P2G
  1.89 → 1.57 ms. Four lanes per node, each summing a fixed subset of the 27
  cells, added in lane order: → 1.20 ms. Both are deterministic, but the
  sums round differently (Metal's fast-math contracts the repacked terms
  differently), so the fingerprints changed.
- `mpm_clear` writes `mpm_touch`'s workgroup counts (one dispatch less per
  substep); `mpm_touch` loads only the position and flags. Bit-identical.
- Tried, not kept: writing back only F and log Jp in `mpm_payload` (3.6×
  slower on Metal, field stores of the struct), and staging a block's
  neighbourhood in workgroup memory for P2G (0.27 → 0.80 ms at the game
  profile: 14 KiB per 64-thread workgroup leaves two workgroups per core).

**Bed depth** (`TerrainConfig::bed_depth`, in R below the event; it was
the lateral reach, 2.5 R at margin 1.5). With the absorbing ground the bed
need not sit that deep. At 12 cells/R, 120 Hz frames:

| Bed | Particles | Centre depth | Deepest | Apparent radius | Rim | Cavity / π |
|---|---|---|---|---|---|---|
| 2.5 R | 0.91 M | 76.7 m | 100.3 m | 201 m | 23.0 m at 252 m | 2.98 |
| 1.5 R | 0.55 M | 80.3 | 100.8 | 201 | 23.8 at 252 | 2.99 |
| 1.25 R | 0.43 M | 80.8 | 97.5 | 201 | 24.0 at 252 | 2.95 |
| 1.0 R | 0.31 M | 78.1 | 87.9 | 201 | 23.8 at 252 | 2.79 |

- From 2.5 R to 1.5 R the crater moves under 5% (the centre +4.7%, its
  deepest point +0.5%); at 1.0 R the floor reaches the bed (deepest −12%).
- The default is 1.5 R (the game profile's 1.25 R). The region's sides and
  top keep the margin. `CraterPrediction::region` refuses a bed shallower
  than twice the transient depth.
- The crater validation at the new default: below.

**The game profile** (`TerrainProfile::Game`; the user's direction: authentic
to look at, not a structural simulation). Each setting against the
demo's crater at 12 cells/R and bed 1.25 R (centre 80.8 m, deepest
97.5 m, rim 24.0 m at 252 m), 120 Hz frames:

| Setting | Particles | Centre | Deepest | Apparent radius | Rim | Cavity / π | Frame (quiet run) |
|---|---|---|---|---|---|---|---|
| 8 cells/R | 115 k | 76.8 m | 92.6 m | 189 m | 23.4 m at 246 m | 2.88 | impact 4 substeps; then 1-2 |
| + Courant 0.45 | | 77.7 | 93.9 | 189 | 23.5 at 246 | 2.92 | impact 3 |
| + speed growth 1 (the game profile) | | 78.1 | 94.3 | 189 | 23.9 at 246 | 2.96 | impact 2; then 1 |
| 6 cells/R | 44 k | 72.4 | 84.3 | 151 (1.0 R) | 26.1 at 176 m | 2.27 | rejected: rim at 1.17 R |
| margin 1.0 | 71 k | 76.0 | 91.7 | 189 | 25.3 at 227 m | 2.70 | not adopted: the box edge holds the rim in |
| E = 5 MPa | | 80.1 | 93.4 | 208 | 24.3 at 264 m | 3.39 | rejected: not settled at 40 s |
| E = 0.5 MPa | | | | | | | rejected: the crater fills in |

- Rings are one sample: 18.9 m at 8 cells/R, 12.6 m at 12, so radii move in
  steps of a sample.
- **Stiffness is not the bound at the demo's scale**: with cells of 12-19 m
  the elastic CFL step is 19-28 ms, longer than a 120 Hz frame. Softer sand
  saves nothing there and does not settle (5 MPa), or loses the crater
  (0.5 MPa). The game profile keeps physical stiffness. Stiffness bounds
  small events (a 1 kt charge's 2.9 m cells: 4.4 ms at physical E).
- **Speed growth** (`TerrainConfig::speed_growth`, default 1.5): the clock
  bounds a frame's speed by this factor × the last reported speed (the
  source's peak until the first report), plus gravity over the lag and the
  host's headroom (20 m/s). At 1 the impact frame takes 2 substeps instead of
  3. The GPU still checks every substep, so a frame whose material outruns
  the bound fails with the CFL error; it is never silently degraded. The
  crater validation's own clock, which has no headroom, failed that way at
  1.0, so that test keeps 1.5.
- **Scale.** At the crater validation's centimetre scale, 8 cells/R gives a
  crater 30% too small (apparent radius 0.875 R; it fails the band), as
  found before: the source's core (0.015 m) is under a cell there, while at
  the demo it spans 2.5 cells. At 12 cells/R the game profile passes the
  bands there. Small or slow events need `cells_per_radius` 12.

Validation of the game profile:
- crater (`CRATER_PROFILE=game`, cm scale): at 8 cells/R fails the radius
  band (above); at `CRATER_CPR=12`: apparent radius 0.172 m (1.17 R; physical
  1.25 R), depth 0.071 m, rim 3.3 cm at 0.221 m, cavity 2.78× (the known
  volume failure; physical 3.08×), 68 s wall;
- slope (`SLOPE_PROFILE=game SLOPE_HZ=120`): the 40° slope relaxes to
  28.6° (physical 29.7°), the 25° slope moves 1.9 mm (physical 1.9 mm);
- coupling (`COUPLE_PROFILE=game`): all three pass; momentum to 2.5e-6
  (physical 3.6e-6), the heavy box at 0.141 m (0.144 m) after 0.33 s.

**Frame time, natively** (GPU ms per 1/120 s frame; quiet runs, the median of
three for each setting):

| Span | Physical, bed 2.5 R (after the binning and payload changes only) | Physical now (bed 1.5 R) | Game |
|---|---|---|---|
| strike frame | 33.8 (6 substeps) | 20.8 (6) | 2.4-4.6 (2) + activation 0.2 |
| 0-0.1 s | 12.9 (3-4) | 7.5 (3-4) | 1.7 (1) |
| 0.1-1 s | 12.1-12.4 (3) | 7.8-8.4 (3) | 0.72-0.76 (1) |
| 1-3 s | 8.2-9.5 (2-3) | 5.7-6.5 (2-3) | 0.72-0.78 (1) |
| 3 s to the bake | 4.1-4.6 (1-2) | 2.6-3.1 (1-2) | 0.67-0.73 (1) |
| all frames: mean / median / 99% | 5.28 / 4.10 / 12.6 | 3.55 / 2.57 / 8.7 | 0.71 / 0.67 / 2.05 |

The game profile settles at 13 s (physical 15 s). In the slower runs every
span rose by up to 3× (other agents' work), so a single worst frame is not
meaningful here; the strike frame's 2.4 ms is the quietest measurement.

**In Chrome** (`impact.html`, `__impact.timings().log`: the page's terrain
GPU time per 1/120 s frame; launched and stepped with `__impact.frames`):

| Span | Physical (`?profile=physical`, 0.55 M) | Game (default, 115 k) |
|---|---|---|
| strike frame | 20.3 ms (6 substeps) | 4.2 ms (2) |
| 0-0.1 s | 7.1 (3) | 1.0 (1) |
| 0.1-1 s | 7.2-7.5 (3) | 0.8-1.5 (1) |
| 1-3 s | 5.1-6.0 (2-3) | 1.8-2.4 (1) |
| 3 s to the bake | 3.2-3.5 (1-2) | 1.8-2.2 (1) |

Before (the task's measurement, 1/60 s frames, 0.91 M particles): 27 ms per
frame in the first 0.5 s, 22-25 ms to 2 s, 15 ms at 2-3 s, 5-11 ms until the
bake at 22 s. The game crater in Chrome: apparent radius 189 m, deepest
93.9 m, rim 23.7 m at 236 m (physical: 201 m, 99.2 m, 22.0 m at 258 m). The
page's whole frame (rendering included) takes 3-6 ms in the game profile.

**Block sleeping: measured, not built.** `PROFILE_REST=1` reads the particles
back and counts the share in blocks whose fastest particle is below a speed,
alone and with one and two blocks around (a sleeping block needs its
neighbours at rest to keep their stencils' nodes exact):

| t | < 1 cm/s | < 10 cm/s | < 0.38 m/s (the demo's rest tolerance) |
|---|---|---|---|
| 0.02 s | 78 / 44 / 8% | 87 / 72 / 50% | 90 / 77 / 58% |
| 0.25 s | 65 / 30 / 2% | 75 / 54 / 25% | 83 / 65 / 39% |
| 1 s | 10 / 0 / 0% | 30 / 8 / 0% | 35 / 11 / 1% |
| 1.5-6 s | 0% | 0-4% | 0-6 / 0 / 0% |
| 8 s | 0% | 6 / 0 / 0% | 96 / 80 / 63% |
| 10 s | 0% | 44 / 11 / 1% | 98 / 86 / 68% |

- From 1 s to 6 s the whole region moves: the strike's waves (above 1 m/s
  across the region at 2 s) take that long to leave through the absorbing
  ground and the free surface. The bed layer and the surface layer move
  above 1 cm/s from the start (the geostatic start is in equilibrium to a
  fraction of a percent of 6 MPa), and the region is only 7-8 blocks deep,
  so the two-block margin covers most of it.
- So sleeping could not lower the peak, which is the strike frame and the
  first second, and in the tail the game profile already costs 0.7 ms per
  frame. It would only pay at a tolerance near the bake's own (0.38 m/s),
  which freezes motion that the bake does not. It is left out.

**What bounds the peak, and what is left to 2 ms per frame** (Metal; the
user's target is every frame at 120 Hz under about 2 ms, the strike frame
included):
- **The strike frame**: activation (0.2 ms, `mpm_activate_emit` a serial
  loop per column), the geostatic start, and as many substeps as the
  source's peak needs. The Z-model's core edge moves at about 1.7 km/s; the
  first P2G averages it to under 600 m/s. At the game profile that is 2
  substeps, about 1.5 ms at 0.7 ms each when the GPU is quiet.
- **Every later frame**: one substep of 0.67-0.73 ms at 115 k particles.
  About 0.15 ms of it is fixed: a dozen small dispatches (clear, the args,
  the scan, the cell sort, the grid, the absorbing faces, the parcels).
- **To get under 2 ms with margin**, in order of yield:
  1. The strike frame's first substep only at the source's peak, the rest at
     the speed the GPU reports after it (a second params record per frame;
     the plan schedules both). Saves a substep: about 0.7 ms.
  2. P2G is about 40% of a substep and bound by L2 traffic. On Metal and
     CUDA (32-48 KiB of threadgroup memory, 256-1024 threads) a staged
     gather with larger workgroups keeps occupancy; behind a capability
     flag, deterministic (fixed order). A scatter with float atomics would
     be faster still but not reproducible; integer fixed-point atomics are
     reproducible but trade range against resolution (the design rejected
     them for WebGPU).
  3. Fuse the small kernels (the grid update into P2G's write; the scan and
     the cell sort), and emit activation per particle.
  4. SIMD-group reductions for the lane sums and scans (Metal, CUDA):
     deterministic.
- WebGPU in Chrome runs the game profile at 1-2.4 ms per frame and 4.2 ms
  on the strike frame: viewable, not the target.

**Fingerprints** (`terrain_fingerprint`, after these changes: the packed
payload and P2G's lanes round differently, and the default bed changes the
impact case's region to 70,644 particles):
- `engine-webgpu` `fingerprint_impact_coupled_and_bake`: particles
  `a822b412486e9128`, bodies `da87b70009941bcf`, heights `bf7da3456dcaa81e`;
  the loaded cube `5ac333cbab3eb7bb`.
- `engine-world` `fingerprint_world_impact_to_bake`: reports
  `c99f8a2f0a2fd4d4`, heights `7539de010b3a6c4f`, box `136fde101445dae3`.

**The strike frame (2026-10-25, second pass).** The user's target: every
terrain frame under about 2 ms natively at 120 Hz, the strike frame
included (the game profile).
- **Its substeps cannot drop at 120 Hz.** Sizing only the first substep at
  the source's peak (about 1.7 km/s at the core's edge) and the rest at the
  speed after it was the plan's first item. At 8 cells/R and Courant 0.45
  that first substep covers 4.4 ms of the 8.3 ms frame, and a second one
  takes the rest whatever its speed: two substeps either way. It would save
  substeps only on the physical profile (6 on its strike frame), and it is
  not built.
- **Per-particle activation emit** (`mpm_activate_emit`: a thread per
  particle, finding its sample by binary search over the samples' first
  particles, instead of a serial loop per column). It writes exactly what
  the loop wrote: 0.12 → 0.06-0.10 ms.
- **The geostatic start in the first payload.** `mpm_geostatic` had a pass
  over every particle of its own. The first substep's `mpm_payload` now sets
  the geostatic F just before the stress: counter `MPM_C_GEOSTATIC`, set by
  the plan and cleared by `mpm_grid`. It sits in a function of its own behind
  an early return: inline, Metal's fast-math compiled the usual path
  differently and the loaded cube's fingerprint changed. The first substep's
  stress now rounds differently (deterministic; the impact fingerprint
  changes).
- **The grid binding** is 32 B per node, two float4 records. It was sized
  at 16 B, so a region using more than half of `max_blocks` wrote past it
  (WebGPU's robust access clamps such writes). No test region came near that.
- **Tried, not kept:** P2G in cell sums. In three rounds of nine nodes, each
  cell first sums its particles' share, then each node adds its cells'
  sums. Each payload is then read 3 times instead of 27, but it was slower
  on wgpu-on-Metal: 0.70 against 0.60 ms per frame (the best of 20 frames
  each). It costs six dispatches and 144 B more per node. The Metal
  executor's `p2g_cells` fast path does the same in one round with
  threadgroup memory, and pays there ("Native backends").
- **Timing.** A pass's own timestamps around many dispatches read 0 on
  wgpu-on-Metal, and the adapter has no encoder timestamps. The wgpu numbers
  are each kernel's own pass, summed (`RUBBLE_TERRAIN_GPU_MS=1`), and the
  strike frame there measures 4.2-4.3 ms however warm the GPU (`WARM=1`,
  `GAP_MS`). `engine-metal`'s `terrain_bench` times the command buffer
  itself; its numbers are the reference.

The strike frame and every frame after it: activation, source and frame 0
together; all 720 frames of 6 s; three runs each.
`cargo run --release -p engine-metal --example terrain_bench -- --game --secs 6 --runs 3`:

| Native, game profile | Strike frame | Worst frame | First second, mean / max | Steady, mean / max |
|---|---|---|---|---|
| Metal, portable kernels | 2.48 ms | 2.06 ms (frame 0, 2 substeps) | 0.98 / 2.06 | 1.01 / 1.26 |
| Metal, fast paths | 2.03 ms | 1.70 ms (frame 0) | 0.97 / 1.70 | 0.50 / 0.55 |

- Other agents' runs shared the GPU until 3.5 s into this run: the fast
  paths' steady frames read 1.0 ms before that and 0.50 ms after.
- The same run's particle and parcel hashes match across all three runs and
  each backend's own runs. With nothing else on the GPU, the strike frame on
  the fast paths is about 1.6 ms ("Native backends").
- What remains of the strike frame: two substeps (0.5-0.85 ms each on the
  fast paths), activation 0.17 ms (its emit 0.09-0.10), the source 0.04-0.08.
  The source could move into the first payload as the geostatic start did
  (it saves its pass), but a host would then read the source's velocities
  only after the first substep, and `the_source_kernel_applies_the_z_model_field`
  reads them right after it. Left as it is.

In Chrome (`impact.html`, the game profile, 1/120 s frames, all 2,029
frames to the bake): the strike frame 1.7 ms (4.2 before this pass); median
0.72 ms, 90% 2.16, 99% 2.23, worst 3.28 ms; mean 1.25.

**Ejecta, deterministic (2026-10-25).** G2P hands each ejected parcel a slot
by atomic, so the slots' order is the GPU's. The Metal executor's run-to-run
check found that it reached results in two ways:
- the deposit's whole-quanta rounding was seeded by the slot;
- `mpm_parcels`'s last workgroup flew some of the parcels G2P had just
  thrown, chosen by slot, so their landing came a substep early or late. The
  test below found this one.

A parcel landing back in the region also took a particle slot by atomic,
which reorders P2G's sums. Now:
- each parcel records the particle it left (after the parcels in binding
  50, 48 B per parcel), and the deposit is seeded by that particle;
- a parcel landing back takes its particle's own dead slot again;
- `mpm_parcels` flies exactly the parcels thrown before this substep's G2P
  (`MPM_C_FLYING`, set with the workgroup counts; `MPM_COUNTER_WORDS` is
  16), and a parcel counts as airborne from the substep it is thrown in.
- Test: `terrain_mpm` `ejecta_are_the_same_run_to_run`. A layer is thrown
  partly out of the region and partly back into it (14,771 parcels): the
  heights, the particles and the parcels (as a set) are bit-identical over
  two runs. Before the last fix, 7 samples' heights differed.

**Fingerprints now:**
- `engine-webgpu` impact: particles `2d38d5b92f5c7c7f`, bodies
  `ab718fa12584d495`, heights `767eaa69f034c221`; the cube is unchanged
  (`5ac333cbab3eb7bb`).
- `engine-world`: reports `9d870853cf98ba9f`, heights `bd49360d36b558ae`, box
  `136fde101445dae3`.
- These cases throw no parcels, so the ejecta change leaves them as they
  were.

**Validation at the physical defaults** (bed 1.5 R; after the second pass:
the same to the digits shown):
- crater (`terrain_crater`, 16 cells/R, margin 2): apparent radius 0.184 m
  (1.25 R), depth 0.069 m, rim 2.9 cm at 0.248 m, axis ridge 0.5 mm, cavity
  3.08× π-scaling's (the known volume failure; 3.10× before). 347 s wall
  (782 s before);
- slope (`terrain_slope`): the 40° slope relaxes to 29.7°, the 25° slope
  moves 1.9 mm (1.5 mm after the second pass; the game profile 28.6° and
  1.8 mm);
- coupling at the game profile passes after the second pass too;
- every listed suite passes (`terrain_mpm`, `terrain_couple`,
  `terrain_absorb`, `terrain_bake_axis`, `terrain_watch`,
  `terrain_materials`, `mpm_material`, `heightfield`, `vehicle_soil`;
  engine-world `terrain`; engine-terrain; `terrain_plan.test.mjs`).

### The barrage's cost (2026-10-29, measured alone)

The showcase recordings that day gave the drive scene's terrain pass
7.14 ms (v4) and 2.36 ms (v5) median against 1.57 ms that morning. Other GPU
jobs ran during both. Measured again with nothing else on the GPU:

**No commit regressed the cost.** Chrome, 1280×720, rendered, the earlier
three-strike schedule (`scene=drive1`), at 103k particles:

| Build | Terrain median | Strike frames (15 substeps) |
|---|---|---|
| c69092aa (before several areas, wheels, parcel landing) | 1.44 ms | 8.52, 10.42 ms |
| 8077fdda | 1.51 ms | 8.52, 10.88 ms |

The +0.07 ms (5%) is the several-area surface views and the wheel loads.
v4's 7.14 ms was contention.

**The barrage costs more because it holds more sand open.** Chrome,
8077fdda, `scene=drive` (11 strikes):

| | Median | p95 | Worst |
|---|---|---|---|
| Terrain pass (GPU) | 2.43 ms | 4.85 ms | 35.4 ms (a strike frame, 16 substeps) |
| Surface pass | 0.79 ms | 1.38 ms | — |
| Render | 1.64 ms | 2.49 ms | 5.9 ms |
| Frame, an area open (wall) | 8.5 ms | 15.3 ms | 51.5 ms |

- The strikes overlap along the road and grow one area. That area never
  bakes:
  - its sand still moves at 0.5–1.5 m/s between strikes (settled is
    0.16 m/s);
  - the coupled car drives through it at about 43 m/s, and a moving body
    holds its area open (the buried-body rule, `RegionClock::report`).
- So the open sand grows with each strike: 52k particles at 2 s, 335k by
  17 s. The steady frame grows with it, from 1.05 to 3.6 ms, about
  5.4 ns per particle-substep.
- A strike frame takes 15 substeps (the physical profile's Courant 0.3 and
  speed growth 1.5 on the source's peak speed) over all of that sand.
  That makes the worst frames 10–35 ms.
- Non-strike frames spike to 3–4× in pairs about every 0.15 s. Every
  kernel inflates at once, the trivial ones too (`mpm_couple_clear` 0.009 →
  0.60 ms natively). The spikes are there at c69092aa as well: the GPU
  clocking or being shared, not terrain work.

**Native (Metal, `terrain_perf` `barrage_frame_cost`).** The showcase's
meteor, 11 strikes 1.6 s apart in areas of their own, none baked (no car,
so nothing couples):

| Profile | Particles at the end | Non-strike median | p95 | Strike frame | Worst |
|---|---|---|---|---|---|
| physical (the showcase's) | 552k | 5.08 ms (2 substeps) | 6.0 ms | 15 substeps: 5.4 ms (44k) to 38 ms (552k) | 38.1 ms |
| game (`GAME=1`) | 342k | 1.43 ms (1–2 substeps) | 3.4 ms | 7 substeps: 3.2 ms (62k) to 12 ms (342k) | 12.0 ms |

Kernel shares in a steady frame (`RUBBLE_TERRAIN_TIMING=1`, about 300k
particles):

| Kernel | Share |
|---|---|
| `mpm_p2g` | 31% |
| `mpm_g2p` | 29% |
| `mpm_payload` (stress, SVD) | 13% |
| `mpm_touch` | 11% |
| count, scan, scatter, cell sort | 8% |
| `mpm_grid` | 3% |

- Native is about 4.6 ns per particle-substep, close to Chrome's 5.4.
- The pipeline was already lean: `mpm_write_args` is fused into
  `mpm_clear`, the hash insert reads before its CAS, and the payload copies
  each particle once a substep. I found no work to skip whose result could
  not differ.

**What it would take to reach 2 ms at the worst frame.** None is lossless,
so none is adopted:
- **The game profile for the showcase** (`?profile=game`, added for this
  measurement):
  - In Chrome: terrain median 1.77 ms, p95 3.41, worst 17.4 ms;
    208k particles at most.
  - But the car ended the run stuck at 0.8 km/h: the schedule's rims were
    placed for the physical profile's craters. Since re-placed, and the game
    profile is now the showcase's default (see the scene notes): terrain
    median 1.70 ms, p95 2.62, worst 17.6 ms at 1080p, recording.
- **Settling under a moving car:** bake an area's quiet sand while a body
  drives over it, rather than holding the whole area open. A change to the
  buried-body rule.
- **Each area on its own clock:** a quiet area would take its own Courant
  substeps, not the struck area's 15.
  - The barrage gains nothing from it, since its strikes all land in the
    one merged area.
  - Separate areas gain the most: the native bench's last strike frame
    would step 44k particles 15 times and 508k once or twice, not 552k
    15 times.
- **Block sleep** (planned behind a flag since the design): blocks at rest
  skip their substeps until something reaches them.

Both are built since, opt-in and measured: "Area clocks and block sleep" below.

### Area clocks and block sleep (2026-10-30, opt-in, off by default)

Two ways to spend substeps only where the CFL bound needs them, each an
explicit `TerrainConfig` setting, off by default. Both change results (a
substep's dt, or what is stepped), so adopting either is a decision for the
project, not a default (AGENTS.md).

**Area clocks** (`TerrainConfig::area_clocks`). Each open area takes its own
substeps: the host gives each area its CFL count from its own speed bound
(`TerrainPlan::set_area_substeps`, `WgpuSim::terrain_set_area_substeps`;
`RegionClock` per area, `TerrainStepReport::area_speed`). The frame's count
is the finest.
- Areas share no nodes, so each steps exactly as it would alone, at its own
  dt. Only the dt of an area slower than the fastest changes (coarser, still
  inside its own CFL bound, which the GPU checks each substep as before).
- The plan groups the areas by count and emits one substep pass per group
  substep. The finest group's passes fly the parcels (they cover the frame
  once). Each coarser group's passes follow the fine pass that brings its
  clock past them. Each group's first pass sets its fresh particles'
  geostatic F.
- `mpm_pass` (one thread, before each pass) writes the pass's dt and CFL
  bound into the domain record, and which areas sit the pass out
  (`MPM_C_SKIP`). The pass table follows the area records in binding 37, at
  most `max_regions × max_substeps` records (allocated only with area
  clocks); a frame needing more fails by name ("area clock passes"). A last
  record puts the frame's dt back.
- `mpm_touch`, `mpm_g2p` and `mpm_absorb_faces` skip a sitting-out area's
  particles and faces: such a particle is in no block, so the other kernels
  never see it.
- Not wired into `World` or the browser host yet: the bench and the test set
  the counts themselves.

**Block sleep** (`TerrainConfig::sleep_speed`, `sleep_after`). A particle
slower than the rest speed, whose velocity also varies by less than that
across a cell (‖C‖·dx), for `sleep_after` substeps in a row falls asleep.
Each substep:
- Awake particles insert their stencils' blocks as usual (`mpm_touch`). So do
  wheel loads (`mpm_sleep_wheels`). An asleep particle in a block cell that a
  coupled collider's padded bounds reach wakes there.
- Those are the awake blocks. Two rings of blocks round them are added
  (`mpm_sleep_ring`, `mpm_sleep_dilate`); asleep particles whose base block is
  in them join the substep (`mpm_sleep_touch`) with their mass and stress.
  The rest of the asleep sand takes no part.
- **The wake criterion, and why skipped work is ~zero.** A node of an awake
  block gathers from particles whose base block is that block or the one
  below it on each axis, all inside the first ring. A node of the first ring
  gathers from inside the second ring. So every node an awake particle or a
  first-ring particle reads is complete: what it would be with sleep off.
  - Those particles take their full update, asleep or not. An asleep one
    wakes (and draws the rings round it) once its update moves it faster than
    the rest speed, or varies its velocity across a cell by more.
  - Second-ring particles, and those outside the rings, are not updated. Only
    material that has stayed at rest is skipped. Anything that would move it
    reaches the first ring a substep before it reaches the particle (signals
    move under one cell per substep at the CFL bound).
  - Sources wake what they move (`mpm_source`).
- The skipped motion is bounded by the rest speed. It is not zero:
  stresses carried by waves slower than the rest speed stop at the rings.

  The first version froze first-ring particles too, waking them only on
  their evaluated speed. Frozen sand then reflected the elastic waves (a
  0.01 m/s wave in this sand carries several kPa, more than the geostatic
  stress a few centimetres down): the small crater came out 62% too large.

**Results against the shared clock with no sleep** (`terrain_clocks`; two 5 mm
impacts 1.8 m apart, the second 20 frames after the first; same device, and a
repeat of the shared run as the noise floor, here bit-identical):

| | Crater 1 cavity | Crater 1 worst height difference | Crater 2 cavity | Crater 2 worst | Ledger |
|---|---|---|---|---|---|
| Area clocks | −0.08% | 8 quanta (about 0.3 mm) | 0.00% | 0 | exact |
| Block sleep, 0.01 m/s | +1.18% | 79 quanta (about 2.8 mm, on a 47 mm deep crater) | +0.93% | 88 quanta | exact |

Area clocks: deepest point and rim unchanged. Sleep: deepest within 1%, rim
height +1% and −6% (its radius within 3 mm). With sleep, 99% of the first
crater's sand slept 20 frames after the strike, 54% a frame after the
second strike (its own ground woken by its source), and all but 21
particles 40 frames later. At this scale the elastic wave sets the CFL count
(about 700 substeps a frame), so the areas' counts differ little and area
clocks save under 1% of the work here.

**Cost** (native, wgpu on Metal, `terrain_perf` `barrage_frame_cost`, 11
strikes 1.6 s apart in areas of their own, measured alone):

| Profile, option | Median | p95 | Worst (the last strike) |
|---|---|---|---|
| game (two runs) | 1.47, 1.51 ms | 2.89, 3.01 | 11.39, 11.70 ms |
| game, area clocks (two runs) | 1.50, 1.50 ms | 2.54, 2.58 | 7.09, 7.05 ms; the worst frame overall 10.28, 10.09 (an early strike's frame spiking, as the shared clock's do: 10.67 ms at 3.2 s) |
| game, sleep 0.01 m/s | 3.81 ms | 7.78 | 30.24 ms |
| game, both | 3.82 ms | 7.32 | 18.61 ms |
| physical | 3.10 ms | 5.09 | 37.81 ms |
| physical, area clocks | 3.09 ms | 5.05 | 21.75 ms |
| physical, sleep 0.01 m/s | 7.62 ms | 10.21 | 78.89 ms |

- **Area clocks** cut the later strike frames by 35–50% (game, at 160k
  particles: 5.80 → 4.54 ms; at 342k: 11.39 → 7.09 ms), with the median
  unchanged.
- **Block sleep costs more on the barrage, not less.** With 65–80% of the
  sand asleep by 8–12 s (at 0.01 or 0.1 m/s), the frame is still 2.6× the
  cost.
  - The two rings of 18 m blocks round the awake sand, which is scattered
    through every crater, quadruple the blocks of a sand layer 2–3 blocks
    deep: 4,522 blocks at 12 s against 1,062.
  - P2G and the grid work per block. This is what "Block sleeping: measured,
    not built" (above) foresaw: the regions are only a few blocks deep.
  - It pays only where large areas lie fully at rest far from awake sand,
    and a baked area costs nothing at all.
- **Neither reaches 2 ms at a strike frame.** A strike frame's substeps cost
  about 0.5–1 ms each even at 27k particles (game: 7.25 ms for 7 substeps of
  27k particles, against 1.29 ms for 2 substeps the next frame). That is
  about 20 dispatches and their barriers per substep, nearly independent of
  the sand. Seven substeps then cost at least about 3.5 ms. Area clocks add
  passes, so they meet that floor too.
  - The lossless lever left for the worst frame is that fixed cost: fewer
    dispatches per substep (fusing kernels, as the native fast paths began)
    and lighter barriers (Metal directly, `MetalSim`). Measured since: the
    dispatches are not that floor (next section).

### What a substep costs (2026-10-30, measured alone)

Native wgpu on Metal, `terrain_perf barrage_frame_cost`, game profile, alone on
the GPU (only an idle 4-day-old `cargo test` process). `FORCE_K=7` steps every
frame at 7 substeps, so each frame costs what a strike frame's substeps cost at
that particle count.

**A strike frame is 7 substeps of the sand there is, nothing more.**

| Particles | 7-substep frame, forced (median) | Strike frame |
|---|---|---|
| 27k | 4.35 ms | 7.0 ms (the run's first frame: a cold GPU) |
| 182k | 6.50 ms | 6.84 ms |
| 342k | 11.61 ms | 11.42 ms |

That is about 0.5 ms a substep plus about 3.4 ns per particle a substep. The
earlier "0.5–1 ms at 27k" compared the cold first frame with warm ones.

**Dispatches are not the floor.** Twenty extra one-thread dispatches a substep
(140 a frame) added 0.7 ms at 342k and 1.4 ms at 27k: 5–10 µs each. A
substep's 13–16 dispatches therefore cost about 0.1 ms of its 0.6–1.7 ms.
Fusing kernels can save at most that. The frame is already one compute pass
(one serial encoder on `MetalSim`), and every adjacent pair in the substep
depends on the one before through another workgroup's writes, except
`mpm_g2p` / `mpm_absorb_faces` / `mpm_parcels` (one dispatch, about 1%).

**Shares at 342k** (7 substeps, each kernel timed in a pass of its own):
`mpm_p2g` 35%, `mpm_g2p` 28%, `mpm_payload` 12%, `mpm_touch` 11%, the
count/scan/scatter/sort 7%, `mpm_grid` 4%. P2G is bound by its gathers'
loads: evaluating only the weight each node needs (3 quadratics, not 9) gained
nothing, so that change was not kept.

**G2P's speed maxima, once per workgroup** (kept). Every particle did two
atomic maxima on the same two counters (the frame's fastest particle, and its
area's). Each workgroup now reduces its 64 speeds in a tree in shared memory
(the speeds' maximum and the lowest and highest area that moved) and does
one atomic each. A workgroup over several areas falls back to one atomic per
area present.
- **Every lane reaches every barrier.** WGSL requires it (Chrome's compiler
  rejects a barrier after a branch on a storage value; native naga let it
  through). So the block-sleep path no longer returns early from the kernel:
  it is still its own function, and it takes its own maxima.
- **Results identical.** The maximum of the same values is the same in any
  order, and the particles' update is untouched. `STATE_HASH=1` hashes every
  particle's words at the end of a run, together with every frame's substep
  count (which the speeds set). The hash is equal for the old and new kernels:
  game profile with 4 strikes, physical with 3, area clocks, and block sleep
  (0.01 m/s). Repeats on the same device were equal.
- **Cost, native wgpu on Metal:**

  | Game profile, barrage | Median | p95 | Worst (the last strike, 342k) |
  |---|---|---|---|
  | Before (two runs) | 1.44 / 1.45 ms | 2.89 / 2.99 | 11.42 / 11.64 ms |
  | After (two runs) | 1.35 / 1.41 ms | 2.67 / 2.70 | 10.50 / 10.52 ms |

  | 7 substeps forced (median) | Before | After |
  |---|---|---|
  | 27k particles | 4.35 ms | 2.60 ms |
  | 182k | 6.50 ms | 5.96 ms |
  | 342k | 11.61 ms | 10.78 ms |

  The gain is largest where few workgroups contend for one address the
  longest: small, fast sand.
- **Chrome** (`showcase.html?profile=game`, 1280×720, 2,580 frames stepped
  by hand, pane visible, nothing else on the GPU): terrain median 1.376 ms,
  p95 2.556, worst 19.3 → 19.2 ms. Unchanged within noise, with the same
  substep schedule. Dawn's atomics there were not the cost.
- `MetalSim` has no barrage bench, so it was not measured. It runs the same
  kernel; `terrain_parity` passes.

**Where this leaves the 2 ms worst frame.** At 342k particles a substep costs
about 1.5 ms, so seven of them cannot fit 2 ms without stepping less sand or
fewer times. Doing either changes results (area clocks; a cheaper material or
coarser cells, by profile), so neither is adopted here. What a lossless
change could still buy is in the gathers themselves (P2G and G2P loading a
neighbour's particles or nodes once per workgroup, not once per node) or in
not copying the payload each substep. Both are larger rewrites, for perhaps
a few tens of percent, not the 5× the target needs.

### How it scales with several strikes (2026-10-30, measured alone)

Native wgpu on Metal, `terrain_perf barrage_frame_cost`, game profile, the
showcase's meteor (about 27–35k particles per crater), nothing baked. `EVERY=0`
now opens every strike in the same frame. Every run's first frame also pays a
cold GPU: a lone first strike costs 7.0–7.2 ms, a later one at 55k particles
4–5 ms.

**Simultaneous, in areas of their own** (150 m apart):

| Strikes | Particles | Strike frame (7 substeps, cold, with the openings) | Settled frame (1 substep) |
|---|---|---|---|
| 1 | 23k | 6.78 ms | 1.05 ms |
| 2 | 55k | 9.36 ms | 0.5–0.9 ms |
| 4 | 124k | 11.30 ms | 1.08 ms |
| 8 | 249k | 15.37 ms | 1.12 ms |
| 16 (`MPM_MAX_AREAS`) | 513k | 25.13 ms | 2.34 ms |

About 1.2 ms per extra strike in the frame they land, linear: that frame does
the strikes' openings and 7 substeps over all of their sand. The frames after
it cost what the open sand costs, about 4.5 ns per particle per substep.

**Staggered, 8 strikes in areas of their own:**

| Apart | Strike frames (the 1st cold) | Worst | Median, p95 |
|---|---|---|---|
| 0.25 s | 7.25, then 4.6–7.8 ms, rising with the open sand | 7.79 ms | 1.16, 2.26 ms |
| 0.5 s | 7.05, then 4.6–7.8 ms | 7.76 ms | 1.27, 2.22 ms |
| 1.6 s | 7.03, then 5.3–9.2 ms | 9.21 ms | 1.13, 2.05 ms |
| 1.6 s, area clocks | 7.21, then 3.3–7.8 ms (the last 4.49 against 7.79) | 7.76 ms | 1.15, 2.20 ms |
| 0.25 s, area clocks | 7.14, then 4.8–7.0 ms | 7.14 ms | 1.27, 2.86 ms |

Staggering spreads the cost: each strike frame pays its own opening and
7 substeps over the sand open then, so the 8 strikes' worst is about half the
same 8 at once. Area clocks help only once the earlier craters have quieted
(1.6 s apart); 0.25 s apart every area is still fast, and the extra passes
raise the p95.

**Overlapping** (8 strikes 25 m apart, one grown area):

| | Particles | Strike frames | Worst |
|---|---|---|---|
| at once | 67k | 10.61 ms (cold) | 10.61 ms |
| 0.5 s apart | 27k growing to 67k | 7.06 (cold), then 3.3–4.6 ms, one 7.07 | 7.07 ms |

Overlapping strikes share sand, so 8 of them hold 67k particles, not 249k, and
cost the least. The grown area is one clock: it takes the fastest bound and
the longest hold of its strikes (up to 3 substeps between them here).

**What bounds it.**
- Cost follows the sand open, not the number of strikes or the land's size:
  separate craters add their particles, overlapping ones add only the ring the
  area grows by. A baked area costs nothing.
- At most 16 separate areas open at once (`max_regions`, `MPM_MAX_AREAS`);
  a closer strike grows an area instead.
- Slots of baked particles and landed parcels come back while other areas are
  open ("Endless areas", 2026-10-30): a bake trims the pool's dead tail and
  packs the parcels, and an activation past the pool's end takes the free
  slots. `max_particles` bounds the particles live at once (plus one slot per
  parcel in flight), not a barrage's total.

## Native backends (2026-10-25)

The terrain plan on native Metal (`engine-metal`'s `MetalSim`), its Metal-only
fast paths, a benchmark against wgpu-on-Metal, and the plan for CUDA. All on
the M3 Max, sharing the GPU with other agents' runs (see "Timing" below).

**Heightfields in `MetalSim`.** `MetalSim::add_height_field` puts the samples
in the hull arena (binding 18, grown to the next power of two of records)
and the collider on a static body of its own, as `WgpuSim` does. Heightfield
pairs take the rigid plan's convex stages (`has_height_field`), and such
scenes take the full persistence scan (the persist list is narrowphase's;
the convex kernels emit after it). `MetalWorld` and CUDA still reject them.

**The executor** (`engine-metal/src/terrain.rs`). `MetalSim` replays
`TerrainPlan`'s `TerrainOps` unchanged:
- Kernels: `build.rs` compiles `engine_terrain::plan::KERNELS` and the
  look-ahead's entries Slang → MSL in the browser's `ENGINE_WGSL` layout (the
  same source as wgpu and the browser). Slang numbers Metal buffers across
  the whole module, past Metal's 31 per stage for a file that includes
  `world_step.slang`; `msl_layout::compact_buffers` renumbers each kernel's
  signature from 0, and the table maps each index back to its `BIND` slot.
- Bindings 37–52 as the plan sizes them, 53 a placeholder for the render
  surface (as on wgpu), 54 the `p2g_cells` partials.
- Each call is one command buffer: the plan's host writes (and pending flag
  resets) as blits at its head, the queue-ordered `write_buffer` of wgpu,
  then one serial compute encoder. Indirect commands use
  `dispatchThreadgroupsWithIndirectBuffer` on binding 46: the counts the GPU
  wrote, no host readback inside a frame. Counters are read in place
  (shared storage) after the frame.
- `terrain_enable` fills every terrain buffer on the GPU. The GPU's first
  touch of a page is far dearer than a later one: on fresh buffers the
  strike's activation and source took 12 ms, prefaulted 0.5–1 ms (0.9 M
  particles). wgpu does not prefault; its strike frame pays this (below).
- API as `WgpuSim`'s: `terrain_enable`, `_set_material`, `_load`,
  `_activate`, `_apply_source`, `_step` (and `_step_timed`,
  `_step_profiled`), `step_with_terrain` (coupled, the rigid step applying
  the reactions), `_bake`, the readbacks; `terrain_profile_next` times any
  call per kernel.
- After merging the branch's kernel changes (the binning keys, the packed
  payload, P2G's lanes, the fused first `write_args`, binding 53) the
  executor ran the new plan without a change of its own.
- Not yet: the render surface pass (`SurfacePlan`; its kernels are not in
  the table), a host API for the look-ahead (its kernels are compiled), and
  waking sleeping bodies on a coupled frame (wgpu's `wake_pending`).

**Against wgpu** (`engine-metal/tests/terrain_parity.rs`, 7 tests, one
process at a time, about 30 s):

| Stage | Metal against wgpu |
|---|---|
| activation (lowered field, particles), the source, activate-then-bake | bit for bit |
| one substep after the source | 95.5% of particles bit for bit, the rest within 1.8e-12 m/s |
| 4 impact frames (70.6 k particles) | max \|Δx\| 5.8e-6 m, \|Δv\| 3.0e-4 m/s; the same particle, parcel and block counts; the bake within 1.1e-4 m |
| 6 coupled frames, a box falling in | the box within 2.1e-6 m |
| host-loaded sand cube, 6 frames | 1.4e-5 m |
| spheres and boxes on a bumpy heightfield (rigid), 120 steps | 1.6e-5 m |
| Metal run against Metal run | bit for bit |

The integer work agrees exactly. The float work differs in its last bits
from the first substep: the two codegens (Slang → MSL, and Slang → SPIR-V →
WGSL → naga → MSL) contract and order expressions differently, both under
Metal's default fast math (wgpu's Metal backend compiles with it; strict
math on Metal moved the run further from wgpu, to 8.5e-5 m). Bit-identity
would need one codegen for both.

**Fast paths** (`TerrainCaps`, `slang/kernels/terrain_mpm_fast.slang`, off
by default; `ENGINE_TERRAIN_CAPS=p2g_cells` or `all`):
- **`p2g_cells`: P2G in two passes, each particle read once.**
  `mpm_p2g_cells` gives each cell a SIMD group: lane o < 27 owns one of the
  27 nodes the cell's particles reach and sums the cell's particles in their
  sorted order (the loads are uniform across the group) into a partial per
  (cell, node). `mpm_p2g_nodes` then sums each node's 27 partials in the
  portable kernel's cell order.
  - Deterministic: fixed orders, no atomics; checked run to run.
  - Not bit for bit the portable kernel: it adds the same terms grouped by
    cell. Against the portable kernels: 6.4e-6 m and 2.8e-4 m/s after four
    impact frames, within what the two codegens differ by.
  - P2G per substep: 0.76 → 0.51 ms at 0.55 M particles, 0.26 → 0.19 ms at
    115 k (quiet profiles).
  - Binding 54 holds the partials: `max_blocks` × 64 × 27 × 16 bytes
    (453 MB at the benchmark's 16 k blocks; size `max_blocks` to the event).
- Tried and dropped:
  - `g2p_simd`: G2P's top-speed atomic reduced over the SIMD group first.
    Bit for bit, but no measurable gain, and it needed a hook in
    `terrain_mpm.slang`.
  - `sort_keys`: count and scatter reading a per-particle key rather than
    the particle. Bit for bit and 0.5 ms faster per substep, but the
    branch's portable kernels now do the same, so it went.
- Not done: float-atomic P2G scatter. Faster, but its sums depend on
  scheduling; it would have to be explicit and off by default, and the
  deterministic two-pass gather took most of P2G's cost already.

**The benchmark** (`cargo run --release -p engine-metal --example
terrain_bench -- [--game] --secs 6 --runs 3 --profile`). The browser demo's
event: an 8 m stony impactor at 20 km/s into dry sand, margin 1.5 R, the
absorbing ground, 1/120 s frames, the demo's clock (`speed_growth` × the last
speed + 20 m/s + a frame of gravity), the same plan on:
- `metal`, the portable kernels;
- `metal-fast`, every fast path;
- `wgpu`, `WgpuSim` on the same Metal GPU.

Metal frames are timed by the command buffer's GPU start and end. wgpu's
are wall times, submission to counters read: timestamps on empty or tiny
passes around the terrain pass came back out of order (Metal does not order
encoders without a shared resource). Metal's wall time runs 0.1–0.3 ms over
its GPU time, so compare wgpu against Metal's wall column.

**Timing.** The GPU was shared throughout. In busy spells every span rose 3–7×
(a 115 k-particle frame at 4–7 ms, `mpm_clear` at 2.4 ms). The tables are
quiet runs: five of eight repeated samples, giving the range and the median
in brackets.

Physical (the defaults: 12 cells/R, bed 1.5 R; 0.55 M particles, 1.7 k
blocks):

| Span | metal GPU | metal-fast GPU | metal wall | wgpu wall |
|---|---|---|---|---|
| strike frame: activation + source + frame 0 (6 substeps) | 14.8–20.6 (19.7) | 13.5–16.4 (14.6) | 15.6–26.8 (21.2) | 51.6–83.7 (51.8) |
| worst frame | the strike frame | the strike frame | | frame 0 (30.6–54.8) |
| first second, mean (3–6 substeps) | 6.6–8.4 (6.8) | 5.9–8.1 (6.1) | 6.8–8.9 (7.2) | 8.0–10.5 (8.7) |
| steady (from 4 s, 1 substep) | 2.26–3.08 (2.34) | 2.00–2.87 (2.27) | 2.4–5.0 (2.7) | 2.9–4.0 (3.7) |

Game (`TerrainProfile::Game`: 8 cells/R, bed 1.25 R, Courant 0.45, no speed
growth; 115 k particles):

| Span | metal GPU | metal-fast GPU | metal wall | wgpu wall |
|---|---|---|---|---|
| strike frame (2 substeps) | 1.56–5.78 (3.61) | 1.62–3.29 (3.24) | 3.0–10.1 (5.1) | 34–59 (39.7) |
| worst frame | the strike frame | the strike frame | | frame 0 (14–27) |
| first second, mean (1–2 substeps) | 0.57–1.67 (0.61) | 0.61–0.95 (0.77) | 0.74–2.4 (0.78) | 1.1–1.5 (1.4) |
| steady (1 substep) | 0.57–0.58 | 0.50–0.82 (0.50) | 0.73–0.75 | 0.96–1.5 (1.05) |

- **Metal against wgpu**: per frame, native Metal's wall time is 20–30%
  under wgpu's at the same plan and GPU (steady 2.7 against 3.7 ms
  physical, 0.74 against 1.05 game; the first second 7.2 against 8.7 ms
  physical). The strike frame differs most: wgpu's
  pays the first touch of its buffers (Metal prefaults at enable) and
  creates its activation's staging on the spot.
- **The 2 ms target** (every frame at 120 Hz under about 2 ms): met by the
  game profile on Metal in quiet runs. Every frame after the strike costs
  0.5–0.6 ms of GPU. The strike frame, which is activation, the source, the
  geostatic start and two substeps, costs 1.6 ms in the quietest runs,
  3.2–3.6 ms in the median ones. At the physical defaults it is not met:
  the strike frame is 15–20 ms and the first second 6–7 ms a frame.

**Per kernel** (quiet runs, ms per substep, `--profile`: one timestamped
encoder per dispatch; the profiled sums run a few per cent over an unprofiled
frame):

| Kernel | physical, metal | physical, metal-fast | game, metal | game, metal-fast |
|---|---|---|---|---|
| P2G | 0.76 | 0.35 + 0.17 (`_cells` + `_nodes`) | 0.26 | 0.11 + 0.08 |
| G2P | 0.53 | 0.53 | 0.16 | 0.16 |
| payload (return map) | 0.44 | 0.44 | 0.06 | 0.06 |
| touch | 0.32 | 0.32 | 0.09 | 0.09 |
| count, scatter | 0.065, 0.045 | 0.065, 0.045 | 0.04, 0.03 | 0.04, 0.03 |
| grid, absorbing faces, cell sort, scan, clear, args, parcels | 0.11 together | 0.11 | 0.21 together | 0.21 |
| per frame: deposits, couple clear | 0.02 | 0.02 | 0.04 | 0.04 |
| a substep | 2.27 | 2.03 | 0.89 | 0.81 |
| activation, source (once) | 0.45, 0.27 | | 0.15–0.21, 0.05–0.06 | |

**What bounds the peak.**
- **The physical event's strike frame** is six substeps at about 2.3 ms:
  0.55 M particles, and a CFL bound set by the source's speed (1.5 × its
  peak + 20 m/s at first), not by stiffness. Its per-particle kernels are
  bound by memory traffic. A substep moves about 700 bytes per particle:
  - the particle read by touch;
  - read and written by the return map, with a 64-byte payload written;
  - the payload read by P2G;
  - the particle read and written by G2P, with its 27 grid nodes.

  At the M3 Max's ~350 GB/s that is about 1.1 ms per substep at 0.55 M (the
  kernels take 2.0–2.3 ms), so six
  substeps cannot come under 2 ms however the kernels are tuned: the strike
  needs fewer substeps or fewer particles.
- **The game profile's strike frame** is 2 substeps of 0.8 ms plus
  activation and the geostatic start (0.2 ms). About 0.2 ms of each substep
  is a dozen small dispatches (grid, absorbing faces, scan, cell sort,
  clear, args, parcels), each a few tens of microseconds, whatever the
  particle count.

**What remains to reach 2 ms reliably** (the game profile's strike frame in
a median run, 3.2–3.6 ms; then the physical event), in order of yield:
1. The strike frame's substeps (the branch's "Performance" list, item 1):
   the first substep at the source's peak, the rest at the speed the GPU
   measures after it. One substep less is 0.8 ms (game) or 2.3 ms (physical).
2. Fuse the small kernels: grid into the P2G node pass (`mpm_p2g_nodes`
   writes the node once; the grid update is per node), scan and cell sort
   into one per-block kernel, absorbing faces into G2P's tail. About
   0.1–0.15 ms a substep at the game profile.
3. Fewer bytes per particle: G2P writing a compact position array for the
   next touch (saves a 128-byte read), and the return map folded into G2P
   for substeps after a frame's first (the next payload computed where F is
   updated; it then needs the payload in particle order and `dt` folded out
   of it). Up to a third of a substep at the physical scale.
4. CUDA graphs and a quiet GPU. Launch gaps are already small on Metal (one
   serial encoder), but the variance between runs on a shared GPU is larger
   than any of these.

**Determinism findings.**
- Metal runs are bit-identical run to run (particles; the parcels as a set)
  through 1 s of the physical event and through four impact frames.
- **Over 6 s the parcels are not.** Particles stay bit-identical, but the set
  of 2,676 parcels differs run to run, on Metal and on wgpu alike. G2P gives
  a parcel its slot with an atomic counter, so its slot k depends on
  scheduling. `mpm_parcels` then seeds the dither of a landing parcel's
  deposit split with `~k` (`mpmParcelDeposit(xl, quanta, ~k, ...)`). Which
  heightfield samples get the odd quantum, and so the baked field, can
  differ run to run. The particles' own slots are fixed, but a re-entering
  parcel also takes its particle slot from an atomic counter (`mpm_parcels`),
  which would reorder P2G's sums. None re-entered in these runs.
  - **Fix** (not made: a kernel change for the substep's owner): carry the
    source particle's id in the parcel and seed by it. Hand out re-entry
    slots in parcel-id order, a scan rather than an atomic.

**CUDA plan.** CUDA has no indirect dispatch. `CudaSim` (the rigid plan)
reads binding 23 back after any kernel that may have rewritten it. The
terrain plan would need that after every `mpm_write_args`: twice a substep,
12 host round trips in a physical strike frame, each draining the GPU.
Instead:
1. **Grid-stride entries** (built: `slang/kernels/terrain_mpm_cuda.slang`).
   - Each `_gs` entry stands for one indirect dispatch of the plan. The host
     launches it with a fixed grid of `MPM_GS_GROUPS` (2048) workgroups.
   - It walks the range the GPU wrote into binding 46 (blocks, particle
     groups, parcel groups) in a grid-stride loop, calling the plan's kernel
     unchanged with the thread id the indirect launch would have given each
     index.
   - Per-workgroup kernels with barriers (`mpm_scan_cells`, `mpm_p2g`'s
     lanes, `mpm_p2g_cells`) loop at workgroup granularity: every thread of
     a group takes the same trip count, and a barrier ends each trip before
     the next block reuses the group's shared memory.
   - Each index runs the same code once: bit for bit the indirect dispatch.
     An idle launch costs one load and a compare per thread.
   - `engine-cuda`'s build compiles all 32 plan kernels, both fast-path
     kernels and the 16 `_gs` entries Slang → CUDA (`CUDA_TERRAIN_KERNELS`),
     with the rigid kernels' deferred-load hazard check (none found), and
     fatbins where nvcc is installed. macOS has no nvcc: none of it has run
     on a GPU.
2. **A CUDA graph per frame shape.** With no host-dependent launch size, a
   frame's launches are fixed by its shape: the substep count and the passes
   (coupled or not, absorbing faces, the geostatic start). Capture each
   shape once, cache it by that key, and replay it with one `cuGraphLaunch`.
   The host writes (the params record, counter resets) go ahead of the
   launch. A six-substep frame is about 90 launches; replayed as a graph it
   costs one.
3. **Conditional nodes, later.** CUDA 12.4's conditional handles (IF and
   WHILE nodes) let the device decide: skip the parcel passes while nothing
   is airborne, or run the substep loop as a WHILE node until the frame is
   covered, the count chosen on the GPU from the speed it just measured.
   That removes `RegionClock`'s readback lag, and with it the strike frame's
   extra substeps (item 1 above). A measured win to try, not needed for
   parity.
4. **The executor**, still to write:
   - `CudaSim` gains bindings 37–54 (the plan's sizes and the fast paths'
     scratch).
   - `TerrainPlan` is lowered as on Metal: `_gs` for every indirect command,
     the fast paths by flag.
   - The counters are copied once a frame into pinned host memory behind an
     event, with no stream drain.
   - `terrain_parity`'s scenarios run on Modal
     (`engine/tools/modal/run_cuda_tests.sh`) against the wgpu host with the
     tolerances above.

## T5 materials (2026-10-24)

Cohesive soil and damage-softening rock, beside dry sand. Each has its own kernel
function behind an early return in `mpmStress`, so dry sand's path is untouched: a
30-frame sand pile hashes bit-identical before and after, with either yield surface.
`terrain_set_material` takes `impl Into<TerrainMaterial>`: `Some(&sand)` and `None`
as before, `&rock`, or a `TerrainMaterial`.

**Cohesive soil** (`SandMaterial::cohesion` > 0, now in Pa; `mpmSoilProjectMohrCoulomb`):
- Mohr–Coulomb with cohesion in stress space,
  f = (τ₁ − τ₃) + (τ₁ + τ₃)·sin φ − 2c·cos φ, replacing the log-strain cohesion shift.
  The tension cutoff is the apex c·cot φ: past it the material separates and log Jp
  keeps the volume, as at the dry tip. The Drucker–Prager cone takes the same apex.
- φ > 0 is required with cohesion (otherwise there is no apex); it is an error.
- Unit tests: idempotence, returns onto the surface, Koiter normality for ψ = 0, the
  apex, unconfined strengths 2c·cos φ/(1 ± sin φ), and equality with the cohesionless
  map shifted by the apex.
- MPM: a soil tower with c = 10.2 kPa, φ = 30° and γ = 17.7 kN/m³ has
  H_c = (4c/γ)·tan(45° + φ/2) = 4 m (Terzaghi). With dx = H_c/16 and gravity ramped
  on over 1 s, it stands at 0.7·H_c (no particle moved a cell) and collapses at
  1.5·H_c (76% moved; the top fell from 6.0 to 4.3 m). A sweep put the solver's
  critical height between 1.0·H_c and 1.1·H_c.

**Rock** (`RockMaterial`: ρ, E, ν, f_t, G_f; `mpmRockDamage`):
- Hencky elasticity and a damage d ∈ [0, 1] in `MpmParticle.c0.w`, carried by G2P.
- A Rankine driver: d grows when the largest undamaged principal stress passes f_t.
- Exponential softening scales the tensile principal stresses by (1 − d); the
  compressive ones are kept (cracks close).
- The crack band is h = 2·dx (`CRACK_BAND_CELLS`). A crack in quadratic-B-spline
  MLS-MPM strains the particles of two cells, and the bar test measures cracks
  exactly 2.00 cells wide at both cell sizes.
- The band must stay under 2·E·G_f/f_t², so dx < E·G_f/f_t² (0.12 m for the default
  rock). A coarser region fails its step with `TerrainError::CrackBand`; the rock is
  never weakened to fit.
- Ejecta that land back in the region come back broken (d = 1). A bake forgets
  damage: the heightfield has no damage layer.
- CPU: uniaxial tension peaks at f_t (to 2e-4) and dissipates G_f per crack area
  (to 2e-4) at four cell sizes. Unloading follows the secant. Compression to
  10·f_t (uniaxial, confined, hydrostatic) stays elastic and undamaged, and is carried
  in full when cracked.
- MPM: a bar pulled slowly by rigid grips cracks beside each grip. The cracks
  dissipate 46.4 and 47.0 J/m² per crack area at dx = 0.04 and 0.02 m (G_f = 50).
  Dynamic stretching is not used as the test: it breaks into a number of pieces that
  depends on the rate and the cell size, with damage spread before it localises.
- The GPU matches the CPU mirror for both materials: projection 3.8e-6, damage 8e-6.

Not yet:
- **Rock shear failure.** Rock has no shear or compressive failure (crushing under
  confinement), and broken rock has no friction limit: it is not yet a granular
  rubble.
- **Coarse cells.** Regions coarser than l_ch (most terrain scales) need rock
  parameters for their scale (a size effect). That is the user's decision, not a
  silent change.
- **Cohesive craters.** Crater scaling for cohesive (strength-regime) targets:
  `CraterTarget::sand` still rejects cohesion.

## T5 ruts (2026-10-24)

Vehicle wheels on soil sink, leave ruts and pay compaction resistance. The tier is
quasi-static Bekker–Wong pressure–sinkage on the heightfield; no MPM.

**Soil.** A heightfield may carry an `engine_schema::soil::Soil`:
- Bekker's k_c, k_φ and n (SI), and `max_sinkage`.
- Presets `dry_sand` and `sandy_loam` (Wong 2008, Table 2.3).
- API: `HeightField::with_soil`, `World::set_height_field_soil` (which places the
  field again from the host's samples) and C ABI `rbHeightField_SetSoil`.
- In the hull arena, after the heights: two parameter records (k_c, k_φ, n, c_n;
  max sinkage), then a plane of uncompacted heights (`heightfield.rs` table).
- The header's min height sits `max_sinkage` below the samples, so the bounds hold
  every rut without a readback. A wheel that would sink deeper sets
  `OVERFLOW_SOIL_SINKAGE`, and the step fails. So does a wheel of zero width.

**Relations** (`soil.rs`, mirrored in `slang/physics/soil.slang`):
- Pressure p = (k_c/b + k_φ)·zⁿ, with b the tyre width.
- Rigid wheel: Bekker's radial pressure on the front arc with the chord
  x² = D(z₀ − z), integrated exactly: W = b·k·√D·c_n·z₀^(n+½), with
  c_n = B(n+1, ½)/2. Bekker's closed form uses (3 − n)/3 for c_n. That is equal at
  n = 1 and 5 % more load at n = 0.7.
- Against the exact arc integral the load is within 0.3·z/D (1.7 % at z/D = 0.06).
- Compaction resistance R_c = b·k·(z^(n+1) − z₁^(n+1))/(n+1), where z₁ is the
  compaction already ahead of the wheel. For radial pressure it is exactly the
  horizontal reaction (1e-3 against quadrature).

**Each step**, after `vehicle_post`, on the GPU (`vehicle_step.slang`):
1. `wheel_soil`, one thread per wheel standing on a soil field:
   - The sinkage is solved with the soil in series with the suspension: the
     wheel's bottom is a massless node, the soil yields until it carries what the
     suspension pushes, and the push falls by K + D/dt per metre the ground gives
     way. At rest this is the static sinkage of the load. Rolling at a steady
     load, it stays within 0.5 % of it.
   - Setting the static sinkage of the last load directly diverged: the damper's
     reaction to each drop fed the next load with a gain of about 2.
   - The patch to press runs from this step's contact to where the next query
     will find it, 1.5 cells past both ends, so every triangle a contact can
     stand on is level. Beyond that, the wheel's circle. Across, it is the tyre's
     width plus half a cell.
   - z₁ is sampled just past the patch's front.
   - R_c goes to the next step's `vehicle_prepare` as a torque bound on the spin
     servo, the same as rolling resistance.
2. `wheel_soil_press`, 256 threads. Thread L owns the heightfield records
   r ≡ L (mod 256) and lowers their samples to (uncompacted − imprint depth),
   never raising one, for every wheel in order. No two threads write one record,
   so runs agree bit for bit.

The plan schedules both passes only when some heightfield has a soil
(`StepInputs::soil`). Without one the step graph is unchanged. The fingerprint
of a sedan run on a box and on a heightfield (`tests/soil_fingerprint.rs`,
ignored, run by hand) is the same before and after the change:
`ededb32943f4b60c` and `9ac86a20ae54669d`. Metal `vehicle_parity` passes.

**Tests** (`engine-webgpu/tests/vehicle_soil.rs`, sandy loam, sedan, 5 cm cells):
- **Static.** Loaded quasi-statically, the soil carries 12 969 N of 12 969 N.
  Each rut equals the sinkage of its wheel's peak load, which is 0.8 % above
  its load at rest. Front ruts are 30.7 mm and rear ruts 33.0 mm, within 0.1 mm.
  Each wheel's bottom sits on its rut's floor, within 3 mm.
- **Motion resistance.** The car coasts from 6 m/s. On rigid ground it
  decelerates at 0.144 m/s², on soil at 0.972 m/s². The difference is
  0.828 m/s², against ΣR_c/m = 0.833 m/s².
  - The front wheels pay the first-pass R_c of their sinkage.
  - The rear wheels run in the front ruts and pay almost nothing: Wong's
    multipass effect between axles.
- **Rut.** Along the path, the rut is within 1.8 % of each step's sinkage. It
  is full depth inside the tyre and untouched beyond its edge plus 1.5 cells.
- **Second pass.** The same car runs again from the same start at the same
  speed.
  - Rut samples under no deeper sinkage are bit-unchanged.
  - Others deepen only to the second pass's own sinkage: its rear wheels carry
    more, because it decelerates less (up to 2.1 mm deeper).
  - It pays ΣR_c = 144 N, against 1 130 N on the first pass, and decelerates at
    0.251 m/s².
- **Limit.** `max_sinkage` 1 cm fails the step with `OVERFLOW_SOIL_SINKAGE`.
- **CPU relations.** `soil.rs` checks them against the arc integrals.
  `heightfield.rs` checks the packing, and engine-world the API.

**Known limits:**
- **Not modelled:** traction limits (Janosi–Hanamoto shear: tyres keep their
  friction curves), multipass hardening, elastic rebound, bulldozing.
- **Sinkage is vertical**, in the field's frame.
- **Grid effects.** A rut's width snaps to the grid. Where a car stops, the floor
  runs 1.5 cells plus the contact half-length past its wheels.
- **Varying load.** A wheel whose load falls climbs to its new, shallower
  sinkage along its path, at the rate the series solve allows.
- **Sphere-cast wheels** (`WheelQuery::Sphere`) are not tested. A rut narrower
  than the sphere's chord at its depth holds the sphere on its lips.
- **Stale host copy.** The host's copy of a pressed field goes stale (as with
  MPM).
  - `set_height_field_region` uploads whole rows from it, which erases ruts there
    and makes those rows fresh soil.
  - `set_height_field_soil` places the field again, so earlier ruts are lost.
  - `restore_state` does not restore the arena, so ruts outlive a restore.
- **MPM** activation lowers the heights but not the uncompacted plane, so a crater
  floor reads as compacted and wheels do not sink there. Material that bake or
  deposit raises above the plane counts as fresh soil.
- **Bodies on the soil.** A body resting on soil is not woken by a rut pressed
  under it. Reused contacts do not see the rut until the terrain's generation
  changes.
- **Hosts.** No browser path (the wasm plan passes `soil: false`). No Metal or
  CUDA path: they reject heightfields. Metal compiles the two kernels.
- **Cost.** Each `wheel_soil_press` thread visits every soil wheel's patch rows.
  It has not been benchmarked.

## Multiple impacts (2026-10-25)

Several hits may be under way at once, overlapping in time and in space: a
second meteor on the rim of a crater still forming, a charge inside a moving
crater, impacts kilometres apart, a hit on ground baked a minute ago. Wide land
costs nothing where it is at rest: only open areas hold particles and blocks.

**One sparse domain, several areas.** The terrain pool is one sparse MPM
domain. Every open *area* is an activated box over the heightfield, with its
own window of samples, columns, absorbing faces, coupling cells, settling and
bake. All areas share three things:
- one cell size;
- one block lattice: block keys are integer block coordinates from the
  domain's origin (the first area's corner), so the active blocks of every
  area, near or far, are one hash;
- one bed heightfield.

The first area's corner is the domain's origin; it resets once every area has
baked.

Considered and not taken:
- *Growing one box over everything.* Two hits 2 km apart would activate the
  ground between them.
- *A domain per area, stepped in turn.* Each would need its own buffers and
  bind groups (hosts bind one set), and two areas that come to overlap would
  have to move particles between pools.

**Keys and frames** (`terrain_mpm.slang`).
- Keys pack x, y, z in 11, 10 and 11 bits, biased (`MPM_KEY_BIAS`). The areas
  reach ±1024 blocks horizontally (±4096 cells) and ±512 vertically from the
  domain's origin: ±38 km at the demo's 9.4 m cells, ±82 m at 2 cm. Past that,
  opening fails by name.
- Each area's record follows the domain's (`gMpmParams[1 + slot]`, `mpmArea`):
  - its box (origin, cells, bed collider), window and bed;
  - its runs of columns, face records and block cells;
  - its origin's integer node offset from the domain's origin.
- A particle belongs to the area whose box holds it (one cell in, as before).
  It is computed in that area's own frame: base node and weights from the
  area's origin, plus the integer offset (`mpmBaseNodeIn`). Grid nodes also
  take their area's frame (`mpmNodeArea`). So an area steps exactly as it
  would alone.
- A hard requirement holds this together: two areas keep at least two blocks,
  and at least two heightfield samples, of rigid ground between their boxes.
  Then no grid block holds nodes of two areas, and no bake or deposit splat of
  one reaches the other's window. A request that comes closer than that grows
  the areas instead.

**Opening** (`TerrainPlan::open`; `World::terrain_impact`, `terrain_explode`,
`terrain_open`, strikes). A request's box is compared with the open areas:
- *Far from all of them:* a new area, activated as before.
- *Inside one:* covered, nothing is activated (a hit near the centre of a
  forming crater whose margin box already holds the new one's).
- *Within the gap of one or more* (overlapping, on a rim, partly outside):
  they and the request grow into one box, their bounding box on the lattice.
  - The grown area's window holds every sample of theirs.
  - A sample that was active keeps the quanta it gave. It gives only the slab
    between its old bed and a deeper new one (the slab's particles take the
    geostatic stress of their depth below the original surface).
  - A sample that was not active is activated from its current height.
  - The moving material is left exactly as it is.
  - A face point the old area had, on the same plane (a side that did not
    move, the floor at the same bed), keeps its displacement. The rest of the
    grown box's faces start at rest, since the ground beyond them was the
    rigid heightfield.
  - Only the new particles take the geostatic start (`MPM_PARTICLE_FRESH`).
  - The grown area is one area from then on: one clock, one bake.
- A request that bridges several areas grows them all into one.

**Sources superpose.** `mpm_source` adds the Z-model field to the velocity of
the selected area's particles within its extent: the impulses of several
sources on the moving material add linearly.
- The field stays bounded at its transient radius, about its own point.
- On material that has moved since its area opened, the source sits on the
  material's surface under its point, not at the caller's height.
  `mpm_source_surface` takes the highest particle within one lattice spacing
  of the vertical line, plus half a spacing, or the heightfield there if
  higher.
  - In the test below, six frames after a first impact, the material 0.4 R
    from its centre stood 3.5 mm above the original surface: the second
    field's centre sits that much higher.
- The area's speed bound adds the source's peak to its own (the speeds add at
  most). Grown areas take the fastest of their clocks and the longest hold.

**Stepping and settling.**
- One frame steps every area together; its substeps follow the busiest area's
  bound (`RegionClock` per area, the largest count).
- G2P and the coupling keep per-area top speeds (`MPM_C_AREA_SPEED + slot`,
  `MPM_C_AREA_BODY + slot`), so each area settles on its own speeds. The
  frame's still settles only with nothing of the terrain in flight: parcels
  are one pool for the domain.
- A settled area bakes by itself; the others go on.
  - A baked area's particles are marked dead, and their slots are reused once
    the pool empties (the last area's bake).
  - The bake's dithers count slots from the area's first, so an area bakes as
    it would alone.
  - The walls' give (`mpm_bake_shift`) used the activation's slot range of
    each edge column. It now uses a counting sort of the edge columns'
    particles by slot (`mpm_bake_edge_*`), in the same order. That works for
    grown areas, whose edge columns hold particles from several activations.

**Cell size.** Every area shares the domain's cells, so a later event runs at
the open grid's cell, or at the fixed `TerrainConfig::cell_size` when one is
set (`TerrainPlan::event_cell`).
- An event is accepted if that cell is at most its own asked cell divided by
  `TerrainConfig::min_resolution` (default 0.5). An impact that asks for 16
  cells per radius then runs at no fewer than 8.
- A coarser grid fails with `TerrainError::CellTooCoarse`. At 8 cells per
  radius the crater validation's crater comes out about 30% small (T4 status).
- A finer grid is always accepted; it costs particles, which the limits bound.
- Levels (several grids, one per cell size) were not done. They need a domain
  per level, and two hits of very different sizes on the same ground would
  still have to share one.

**Limits** (all `TerrainConfig`, by name):
- `max_regions`: open areas, at most `MPM_MAX_AREAS` = 16.
- `max_columns`, `max_boundary_nodes`: over all areas.
- `max_blocks`: the boxes' blocks over all areas.
- The keys' reach, one heightfield and one material density at a time.

**Cost per frame.** No readbacks and no passes over the whole grid are added.
- The domain record is written each frame (208 bytes, as before). The area
  records (3.3 KB) are written only when an area opens, grows or bakes.
- The per-particle and per-node kernels loop over the slots in use (one with
  one area) to find a particle's or node's area. Per-area atomics run only
  while slots past 0 are in use.
- `terrain_bench` (32,768 particles, 176 blocks, one area): 0.399–0.403 ms per
  substep, against 0.397–0.400 ms before. That is within the run-to-run noise
  of a shared GPU.

**Merged with native Metal and deterministic ejecta (177d8e4e).**
- Activation emits one thread per particle in the selected area's columns,
  the quanta each sample gives now.
- The geostatic start rides the first payload, for fresh particles only
  (`mpmGeostaticFresh`), in their area's frame.
- Re-entering parcels take their source particle's dead slot in whichever
  open area they land in.
- The area counters follow the 16-word frame block: `MPM_C_SURFACE` 13, area
  speeds from 16, bodies from 32, first slots from 48 (64 words).
- `terrain_mpm_fast.slang`'s two-pass P2G takes each block's area frame.
- `engine-metal` gains `terrain_open_area`, `terrain_bake_area` and
  `terrain_apply_source_to`.
- A parcel still in flight when the last area is closed by force kept a
  source slot past the emptied pool, and was lost. Fixed since: such a close
  lands it ("Forced closes" below).

**Merged with the performance and rendering work (80f16fbd).** The surface
pass (`terrain_surface.slang`, render-only) showed one area's window: header
word 7 named its slot (since replaced by one view per open area; "Several
open areas at once"). It binned only that area's particles. For a grown area's older edge columns its walls' give is taken as
zero (it reads the last activation's slot ranges). The host turns the surface
off when the last area bakes. `showcase.html` and `impact.html` treat a kind-3
frame as coupled.

**Validation** (`engine-webgpu` `tests/terrain_multi.rs`: the crater
validation's 5 mm aluminium sphere at 2 km/s, transient radius 0.147 m, at 8
cells per radius; 40 frames, then the bakes).

| Case | Result |
|---|---|
| Two identical impacts 1.8 m apart, against single ones | one area bit-identical, the other within 1.8e-4 to 2.2e-4 m (about 150 of ~1,800 samples, 5–6 quanta) where the shared parcels land; which one depends on the parcel slots' atomic order (the first before the merge with the performance work, the second after); cavity volume within 7e-5 |
| A second source on still material | adds the field's momentum to 4e-7 |
| A third on material 6 frames in motion | adds the field's momentum from the found surface to 1.7e-7 (at the caller's height it would differ by about 8%) |
| Hit inside the forming crater (0.2 R off centre, 10 frames in) | cavity 1.43× the single crater's |
| Hit on its rim (1 R, 10 frames in): the area grows | cavity 1.60× |
| Hit partly outside (2.5 R, 4 frames in): the area grows | cavity 1.88× |
| Two far hits in one frame, then a hit on the baked crater | two areas; the baked ground reactivates; cavity 7.07e-4 → 9.31e-4 m³ |

- The volume ledger is exact in every case: the field's change, the live
  particles and the escaped quanta sum to zero quanta.
- The same runs single are bit-identical to themselves.
- The World API (`engine-world` `tests/terrain.rs`) runs two far meteors and a
  third on the first one's rim after 8 frames. The rim hit grows the first
  area; the far one bakes while the grown one still moves. The cavities are
  1.09e-3 and 6.71e-4 m³, and the ledger is exact to the quantum.
- The C ABI test opens two areas with `rbTerrain_Impact`; the wasm plan test
  runs kind 3 frames (one area bakes, another steps on).

**Single-area runs.**
- An area's arithmetic is the old region's. But the kernels are new code, and
  Metal compiles them differently, so the fingerprints changed at the ulp
  level. The particles of both fingerprint scenes lie on half-cell lattices,
  exactly where a base node's floor can flip.
- After one substep of the loaded cube, velocities differ in the last one or
  two bits.
- Every validated suite passes unchanged.
- The crater validation (margin 2, 16 cells/R, absorbing ground) gives what
  it gave before: apparent radius 0.1838 m (1.25 R), centre 0.069 m deep, rim
  2.9 cm high at 0.239 m, ledger closed. The cavity is 3.09× π-scaling's, so
  the volume band still fails there, the known open point (3.10× before).
- New fingerprints: `engine-webgpu` impact particles `c71c7e64cf7aef94`,
  bodies `63ce7712cd6c0b07`, heights `39a743e448abe527`; cube
  `a0c6daecfc7ce3b6`. `engine-world`: reports `6ee2a0a7740af3dd`, heights
  `cfa3765bdac07cb1`, box `136fde101445dae3` (unchanged).

**Measured on the way: the source frame's time step.** A harness bug once
doubled the first frame's speed bound for both areas of the far-apart test,
and so about halved that frame's substep. Both craters came out with 0.765 of
the cavity volume (5.41e-4 against 7.07e-4 m³). The crater depends on the time
step of the frame the field starts in. That bears on the open volume excess
(3.1× π-scaling). It is reported, not investigated or adopted.

**Open points.**
- **Slot reuse.** Done without moving a live particle ("Endless areas",
  2026-10-30): trimming at bakes, parcels packed, free slots for an
  activation past the pool's end.
- **Parcels are one pool.** An area's landing ejecta take slots, deposit
  dithers and re-entry order shared with the others, in the atomics' order.
  So an area matches a single run only to a few quanta where its ejecta
  land.
- **Bounding boxes.** Growth is to a bounding box: an L of two hits activates
  the corner between them. The union of rectangles would need faces on any
  column boundary.
- **Between areas.** The ground between two areas is rigid heightfield, and
  each area's absorbing faces treat the ground beyond as an infinite
  half-space. Waves one area sends into the ground do not reach the other.
- **Settling.** One area's ejecta in flight delay another's bake; quiet areas
  pay the busiest one's substeps.
- **Pre-existing.** A parcel over an open window lands on the window's
  lowered heightfield, the bed, and becomes a particle there, under the moving
  material. It was not changed here.
- **Browser.** `TerrainStep.frame()` returns kind 3. `terrain-host.mjs`
  executes it; a page that reads `kind` must treat 3 as coupled.
  `terrain.areas` lists the open areas.

**The calls** (the showcase).
- World:

  ```rust
  w.enable_terrain(&TerrainDef { config: TerrainConfig { max_regions: 4, max_particles: 4 << 20, ..Default::default() }, ..Default::default() })?;
  w.terrain_impact(ground, &meteor_a, [x1, y1, z1])?;
  w.terrain_impact(ground, &meteor_b, [x2, y2, z2])?; // same frame, or any later one
  w.terrain_areas();          // slot, shape, box of each open area
  w.terrain_close_area(slot); // or terrain_close() for all; each bakes by itself once settled
  ```
- JavaScript (`terrain-host.mjs` over the wasm `TerrainStep`):

  ```js
  const terrain = await createTerrain(world, { config: { max_regions: 4, max_particles: 4 << 20 } });
  terrain.impact(collider, field, [x1, 0, z1], radius, density, speed);
  terrain.impact(collider, field, [x2, 0, z2], radius, density, speed);
  terrain.encodeFrame(enc); // each frame: kinds 0–3
  terrain.areas;            // JSON per area: slot, origin, end, dx, window, prediction
  terrain.closeArea(slot);
  ```

  With `TerrainStep` itself:
  - `impact(collider, fieldWords, at, radius, density, speed, gravity)` and
    `explode(…)` return the command words;
  - `frame(dt, gravity, colliders)` returns kind 0–3;
  - `areas()`, `close_area(slot)` and `close()`.
- C: `rbTerrain_Impact` as before, any number of times; `rbTerrain_AreaCount`;
  `RbTerrainDef.max_regions`, `cell_size`, `min_resolution`.

### Forced closes (2026-10-29)

The World bakes an area by itself only once it has settled with nothing in
flight. A host can still close areas early: `World::terrain_close` and
`terrain_close_area`, `TerrainStep.close()` and `close_area(slot)` in the
browser, the C close, or `WgpuSim`/`MetalSim::terrain_bake`. The ledger stays
exact through every such close.

| Close | What happens to parcels in flight |
|---|---|
| One area of several (`bake_area` while others stay open) | They keep flying with the open areas' substeps. Each lands on the heightfield, or back in an open area as a particle in the dead slot of the particle it left. That slot stays held while it flies ("Endless areas"), so no one else takes it. Unchanged. |
| The last area (`bake`, or `bake_area` on the only one) | Before: no frame runs once no area is open, so they hung in the air. The next fresh domain reset the parcel counters and dropped their quanta. Their source slots also pointed into an emptied pool that the next activation refills. Now: in the bake's pass, after the window is raised, `mpm_parcels_land` flies each one on to where it lands, all in that pass. |

**How `mpm_parcels_land` flies them:**
- Each parcel is flown exactly as `mpm_parcels` would: symplectic Euler substeps at the last frame's dt and gravity.
- It lands on the baked ground, with `mpmParcelDeposit` and the same seed.
- It escapes when it crosses the field's edge.
- No area is left to take it back, so it never re-enters as a particle.
- A parcel rising with no gravity to bring it back escapes, since it never lands.
- The flight runs at most 2²² substeps, far past any flight under the CFL bound. A parcel still up after that stays counted airborne, though none can be under gravity.
- The open checked the field against `max_field_samples`, so every share has its deposit slot.

**A forced close ends the flight now.** The quanta land where the parcels would have come down with no area open, and in the same frame as the close. Their deposits are on the field before the pool empties, so a domain opened afresh starts with nothing of the old one in flight.

**Other paths checked:**
- Every other way material leaves the pool or the ledger is already exact, or fails by name:
  - a particle leaving its box with no parcel slot raises `max_parcels`;
  - with `max_parcels` 0, a particle outside every area raises `MPM_OUTSIDE`, and the last bake still takes it;
  - a deposit with no slot raises `max_field_samples` and stays in flight.
- `TerrainPlan::load` replaces the domain on purpose: it is for host-loaded test material and drops what was there. It is not a close.
- Parcel slots come back at every bake that leaves areas open ("Endless areas": the parcels still in flight are packed to the front). Between such bakes, a domain that throws more than `max_parcels` fails by name even if most of them have landed.

**Validation** (`engine-webgpu` `tests/terrain_multi.rs`, the scene above). The area's material is thrown up at 4 m/s and out at 1 m/s, as a blast would; this impact's own ejecta stay inside its box.

| Case | Result |
|---|---|
| The only area closed 3 frames (0.05 s) into the flight | 27,181 parcels (1,739,584 quanta) in flight land in the close's pass. Nothing airborne after it; 82,806 quanta escaped off the field. Ledger (field + live + escaped) exactly 0 quanta. Without the landing pass, all 27,181 stay in flight and the test fails. |
| A second impact in the fresh domain afterwards, settled and baked | Ledger over both domains exactly 0. |
| One of two areas closed mid-flight (27,138 parcels in flight); the other steps on, lands them, settles and bakes | Ledger exactly 0. |

## Validation targets

Tests come from physics and published laws, never from another program's outputs.

- **Conservation**, every frame: initial volume = heightfield + particles + airborne +
  escaped. Activate-then-bake with no forcing is the identity.
- **Heightfield collider**: a box at rest on a flat heightfield matches the plane; a
  sphere rolling down an inclined heightfield matches the analytic acceleration
  (5/7 g sin θ); no tunnelling under CCD.
- **Granular**: column collapse runout and final height follow Lube et al. (2005) scaling
  with aspect ratio; settled slopes stay at or below the friction angle.
- **Coupling**: penetration depth of a sphere dropped into sand follows the Katsuragi &
  Durian (2007) trend (d ∝ H^1/3).
- **Cratering**: final diameter in dry sand follows gravity-regime π-scaling (log-log slope
  across two decades of energy, absolute agreement within a stated band); transient
  depth/diameter ≈ 1/3 and final ≈ 0.2 for simple craters (Melosh 1989); ejecta blanket
  thickness ∝ r^−3 beyond the rim (McGetchin et al. 1973); a vertical impact is
  axisymmetric; azimuth rotates the result covariantly; a deeper bed does not change it.
- **Limits**: exceeding any limit returns its documented error.
- **Performance**: 10k rigid bodies on a heightfield against the plane baseline; MPM cost
  per substep against particle count.

## References

- Klár, Gast, Pradhana, Fu, Schroeder, Jiang, Teran (2016). Drucker-Prager elastoplasticity
  for sand animation. ACM TOG 35(4).
- Hu, Fang, Ge, Qu, Zhu, Pradhana, Jiang (2018). A moving least squares material point
  method with displacement discontinuity and two-way rigid body coupling. ACM TOG 37(4).
- Wolper, Fang, Li, Lu, Gao, Jiang (2019). CD-MPM: continuum damage material point methods
  for dynamic fracture animation. ACM TOG 38(4).
- Gao, Wang, Wu, Qiu, Sifakis, Jiang (2018). GPU optimization of material point methods.
  ACM TOG 37(6).
- Maxwell (1977). Simple Z model of cratering, ejection, and the overturned flap.
- Holsapple (1993). The scaling of impact processes in planetary sciences. Annu. Rev.
  Earth Planet. Sci. 21.
- Melosh (1989). Impact Cratering: A Geologic Process. Oxford University Press.
- McGetchin, Settle, Head (1973). Radial thickness variation in impact crater ejecta.
  EPSL 20.
- Lube, Huppert, Sparks, Freundt (2005). Collapses of two-dimensional granular columns.
  Phys. Rev. E 72.
- Katsuragi, Durian (2007). Unified force law for granular impact cratering. Nature
  Physics 3.
- Terzaghi (1943). Theoretical Soil Mechanics. Wiley.
- Lysmer, Kuhlemeyer (1969). Finite dynamic model for infinite media. J. Eng.
  Mech. Div. ASCE 95(EM4).
- Deeks, Randolph (1994). Axisymmetric time-domain transmitting boundaries.
  J. Eng. Mech. 120(1).
- Liu, Du, Du, Wang, Wu (2006). 3D viscous-spring artificial boundary in time
  domain. Earthquake Engineering and Engineering Vibration 5(1).
- Bažant, Oh (1983). Crack band theory for fracture of concrete. Matériaux et
  Constructions 16.
