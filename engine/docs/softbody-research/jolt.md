# Jolt soft bodies

Read-only review (2026-11-02) of JoltPhysics (`Jolt/Physics/SoftBody/`).

## 1. Solver

**"Small steps" XPBD.** The iteration count (5 by default) is really the
substep count. Each substep:
1. Integrate.
2. One Gauss–Seidel pass over all constraints.
3. Collisions.
4. Velocity from (x − x_prev)/dt.

There is no accumulated λ.

**Constraints:**
- edge distance, including shear and bend edges;
- dihedral bending with a rest angle;
- tet volume;
- skinned constraints (maximum distance plus a backstop);
- long-range attachment tethers: a one-sided maximum distance to the nearest
  kinematic vertex, Euclidean or geodesic;
- pressure.

There are no FEM elements.

**Compliance:** α/dt² per substep. The default is 0, which is hard.

## 2. Stiffness

- **Step size:** scaling compliance by 1/dt² removes the dependence on dt, but
  convergence still depends on the substep count and mesh size.
- **Rigid look comes from:**
  - tethers, which bound stretch in one projection;
  - constraints sorted from kinematic anchors outward;
  - volume and pressure;
  - skinning to a driven pose.
- **Shape matching:** none.

## 3. Plasticity and fracture

None. Rest data is immutable and shared between instances. Adding plasticity
would need per-instance rest overlays and masks of live elements.

## 4. Collision

- **Pairs:** soft against rigid only. There is no soft–soft or self-collision;
  it is still a TODO in the source.
- **Contact plane:** one per vertex per frame, reused across substeps.
- **Friction:** position-based Coulomb.
- **Rigid coupling:** a two-way impulse to dynamic rigid bodies, one frame late,
  because soft bodies step after rigid ones.

## 5. Performance

- **Jobs:** one job graph per body.
- **Clusters:** an optimise step builds spatially compact clusters of 256
  vertices that run in parallel, plus one serial group for constraints that
  cross clusters.
- **Memory and frames:** vertices are an array of structs. Each body's local
  frame is recentred every step.
- **Sleep:** per body, on the maximum vertex velocity.
- **No GPU path,** and no benchmark numbers.
- **Vehicles:** rigid bodies with raycast wheels. There is no soft-body vehicle.

## 6. Lessons for Rubble

- **GPU shape:** XPBD's one-dispatch-per-substep structure suits the GPU, but
  its stiffness error rules it out for sheet metal (see
  [jgs2-and-benchmarks.md](jgs2-and-benchmarks.md)).
- **Anti-jello:**
  - tethers from chassis hard points to panel nodes;
  - volume and pressure terms for tyres and cabins.
- **Ours to build:** plasticity (per-instance rest data), breaking, and
  self-collision and vehicle–vehicle contact.
- **Lossy, so explicit option only:** one contact plane per frame.
- **Adopt:** recentre each vehicle's local frame, and sleep per vehicle.
