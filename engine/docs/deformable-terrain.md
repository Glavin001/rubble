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
fail with a clear error when exceeded: active regions, particles, grid blocks, parcels,
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
  and not on the Metal or CUDA backends (they reject heightfield colliders: their
  step graphs have no convex stages).
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
  "Extent"). It keeps 266 of 268 J and carries a momentum-transfer β of 1.26.
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

## The splat bake (2026-10-25)

The user approved a bilinear splat for the bake and the parcels' deposits.
`mpm_bake_bin` and `mpm_parcels` (`mpmParcelDeposit`) split each particle's or
landed parcel's quanta between the four heightfield samples around it, with
tent weights. Each share goes where the whole particle went before:
- a window sample's column;
- the deposit accumulator, for a sample outside the window;
- escaped, for a sample off the field.

The max-height header is kept by the same raise kernels as before.

**Where a particle is splatted: its position, less a fading share of its rest
offset.** Activation lays a sample's particles on a lattice across the
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
  strength at shallow depth. Along an axis no face holds, the ground stays
  where it was.
  - Found on the dropped ball (engine-world `terrain`): without the limit, the
    sand flowing along its shallow region's bed stopped the ball at 0.089 m.
- **Discretisation.** Node i takes ∫ w_i·t dS over the faces (w_i its
  B-spline), lumped: its share of a face is ∫ w_i dS, from the B-spline's
  closed-form integral, and its dashpot and spring act on its own velocity and
  displacement. That is the force the particles' geostatic stress transmits
  through the face, so the activated state is in equilibrium.
  - A floor half a cell above a node layer splits its load between two layers;
    walls off the node lattice spread over three.
  - In a 1D check, the dashpot on the boundary node alone sent back 2.6% of a
    pulse's energy and the consistent shares 0.13%.
- **Each node is solved implicitly:** v = (m·v* + dt·(f0 − K·u))/(m + dt·C + dt²·K)
  per axis, stable for any dt. It keeps its displacement in a record of its own:
  the floor's three node layers and each side's three, after the coupling slots
  in binding 49 (`TerrainConfig::max_boundary_nodes`, an error past it).
  `mpm_grid` still binds 9 storage buffers.
- **Unchanged:** the region's top and the free surface stay free. Contact with
  rigid bodies, host-loaded regions (no window: the field's surface is a contact)
  and the rigid ground's code path are as before.
- **Validation.** The absorbing ground requires the geostatic start and a
  material with stiffness; either missing is an error.

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
- **Not the default.** `TerrainConfig::default()` keeps the rigid ground and
  the unstressed start, so existing results and fingerprints are unchanged.
  With the absorbing ground on, every suite passes except engine-world's
  dropped ball: it penetrates 0.094 m, 0.486× Uehara's 0.194 m, just under
  the test's 0.5–2× band (rigid ground: 0.115 m, 0.59×). Its region is shallow,
  so the ball's flow reaches the bed and walls; most likely the rigid box
  sends back energy the absorbing ground lets go (not verified). At
  6 cells per body the penetration is under-resolved either way. Making it
  the default needs that settled (`TERRAIN_GROUND=absorbing` runs it).
- **The walls' top corner.** Within 1.5 cells of a side's top edge, the
  unactivated ground's surface is the soft wall face rather than a rigid
  contact. Rim material deposited there sinks into the corner at the spring's
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
