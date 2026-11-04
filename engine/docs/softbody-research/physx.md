# PhysX GPU deformables

Read-only review (2026-11-02) of the PhysX 5 source under `physx/source/`.

## 1. Solvers

- **Deformable volumes:**
  - **Meshes:** a tet mesh for collision, plus a separate simulation mesh made
    of voxel hexes split into five or six tets.
  - **Solver:** XPBD projection with one multiplier per constraint. Compliance
    scales with dt² and inversely with stiffness × element volume.
  - **Materials:** co-rotational (ARAP-style deviatoric term plus a nonlinear
    volume term), which needs good tets at high stiffness; or Neo-Hookean, for
    soft materials.
  - **Colouring:**
    - greedy, at cooking time, capped at 8 partitions;
    - overflow elements go to a ninth Jacobi partition on duplicated vertices,
      whose deltas are averaged.
  - **Iterations:** follow the rigid solver's position iteration count (PGS or
    TGS). Under TGS the multipliers reset every iteration, so iterations act as
    substeps.
- **Cloth:**
  - XPBD fixed-corotated membrane (2×2 deformation gradient) plus dihedral
    bending;
  - small partitions merge into one cluster launch.
- **PBD particles:** Jacobi with atomic deltas, relaxation 0.125.

## 2. Against jello

- **Stiffness measures:**
  - co-rotational FEM with physical moduli;
  - compliance scaled by element volume;
  - TGS substeps;
  - Gauss–Seidel within 8 colours.
- **Damping:** a (1 + damping/dt) factor in the XPBD update, plus global linear
  damping.
- **Plasticity:** a per-axis yield kernel exists, with a deform threshold and
  low and high ratios, but it is hard-disabled with a note that it doesn't
  really run.
- **Tearing and fracture:** none.

## 3. GPU structure

- **Batching:** one launch per element type covers every active body. The block
  index selects the body, and each block copies its body's descriptor to shared
  memory. The grid is sized for the largest body's partition.
- **Launches per iteration:** more than 10 serial launches: 8 partitions, the
  Jacobi extra, deltas, averaging and contacts.
- **Contact pairs:** a BVH midphase against rigids, heightfields, particles and
  other deformables.
- **Self-collision:**
  - volumes: vertex against tet, filtered by rest distance and a stress
    tolerance;
  - cloth: vertex against triangle, plus edge against edge.
- **Contact resolution:** Jacobi with reference counts.
- **Sleep:** per body, with counters read back each step.

## 4. Coupling with rigid bodies

- Two-way contacts are solved inside the rigid iteration loop.
- Contacts are counted per rigid body with atomics, and averaged deltas are
  written into the rigid velocity pool.
- All of it runs on the GPU.

## 5. Benchmarks

None found in the source.

## 6. Lessons for Rubble

- **Batching:**
  - batch all vehicles per dispatch through an indirection table;
  - pack colours into one flat work list rather than a grid sized for the
    worst body;
  - avoid launch counts that grow as colours × iterations × element types.
- **Colouring:** cap the colour count and send overflow to a Jacobi partition.
- **Stiffness:** multipliers plus per-constraint damping, with substeps rather
  than iterations.
- **Plasticity and breaking:** ours to build, forward-only, applied after the
  solve, which is where PhysX meant to put it.
- **Two meshes:** the beam graph simulates and a detailed mesh is skinned to it.
- **Self-collision:**
  - filtering by rest distance is lossless, so adopt it;
  - skipping pairs based on stress drops contacts, so it can only be an
    explicit option.
- **Sleep:** per vehicle, reading back only the changes.
