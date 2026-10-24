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
  particle's quanta are added to the heightfield column it stands over, and the
  particles are freed.

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

Not yet:
- hit events as the activation trigger;
- coupling for bodies that rotate fast within a frame (poses are the frame's start);
- the browser path (`mpm_grid` binds 9 storage buffers, past WebGPU's default 8);
- the host's copy of an edited heightfield (it goes stale until a readback);
- performance (about 1,900 dispatches per 60 Hz frame at physical stiffness).

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
- Bažant, Oh (1983). Crack band theory for fracture of concrete. Matériaux et
  Constructions 16.
