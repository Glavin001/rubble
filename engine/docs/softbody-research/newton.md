# NVIDIA Newton: VBD, XPBD, Style3D and MPM

Read-only review (2026-11-02) of the solvers under `newton/_src/solvers/`. VBD is
Chen et al., *Vertex Block Descent*, 2024.

## 1. VBD

**Step:**
1. Predict each node: position plus (velocity + gravity × dt) × dt.
2. Run iterations (10 by default). Each is a pass per colour of 3×3 Newton
   steps per vertex: inertia m/dt² plus the terms of adjacent elements, found
   through a compressed adjacency list. A tiled variant packs vertices per GPU
   warp.
3. Take velocity from displacement.

There is no Chebyshev acceleration.

**Elements:**
- **Springs:** exact Hessian, not clamped to positive definite under
  compression.
- **Triangles:** stable Neo-Hookean membrane, projected to positive definite.
- **Bending:** dihedral, Gauss–Newton.
- **Tets:** stable Neo-Hookean.

**Colouring:**
- Done on the CPU at build time, greedy or MCS, with balanced colours.
- **Springs are not in the colouring graph,** so a spring's two ends can share
  a colour and update partly as Jacobi. For beam-only vehicles that is a hole.

## 2. Stiffness and damping

- **Penalties, not multipliers:** particles use plain penalty energies, so stiff
  meshes with high mass ratios go soft when under-iterated.
- **Damping:** implicit, along stiffness directions, measured from the start of
  the step.
  - Springs damp axially.
  - Membranes damp by the rate of the stretch tensor, so rotation is undamped.
  - Contact damps the gap rate only.
- **Augmented Lagrangian (AVBD, Giles 2025)** is used only for rigid bodies:
  - each row keeps a multiplier, decayed by γ = 0.999;
  - effective stiffness is k·ρ/(k + ρ);
  - a hard mode stabilises against the stored constraint offset.

## 3. Plasticity and fracture

There is none in VBD, XPBD or Style3D. Only implicit MPM has yield, for granular
material and snow (Cam-Clay). It is grid-based and does not transfer to beams.

## 4. Self-contact

- **Pairs:** vertex–triangle and edge–edge from a BVH, re-detected every N
  iterations.
- **Contact energy:**
  - quadratic penalty out to the contact radius;
  - a log barrier inside half the radius;
  - a C² extension below the minimum distance.
- **Friction:** IPC-style smoothed friction with a fixed normal.
- **No pass-through (DAT truncation):** each vertex's displacement is capped per
  iteration to about half the query radius, with relaxation 0.85. The same
  scheme handles rigid–soft contact.

## 5. GPU structure

- **Launch count:** about three to five launches per colour (attachments, body
  contact, springs, self-contact, elasticity, truncation). That is colours × 5
  kernels per iteration, and it can be captured as a CUDA graph.
- **Batching:** many bodies merge into one particle array with a world index.
- **Colouring:** CPU only.
- **Contact forces:** scattered with atomics.

## 6. Lessons for Rubble

- **Solver skeleton:** VBD with displacement-based state.
- **Colouring:**
  - colour the beam graph ourselves; dense BeamNG nodes need about 25–40
    colours;
  - gather (each vertex loops over its beams) rather than scatter per colour
    over beams.
- **Stiffness:**
  - clamp the spring Hessian to positive definite;
  - per-beam λ with warm start, for stiffness independent of the iteration
    count;
  - axial implicit damping.
- **Plasticity:** update plastic rest lengths after convergence, and break
  beams by masking, which keeps the colouring valid.
- **Self-collision:** barrier contact plus IPC friction plus displacement
  truncation.
- **Avoid:** Style3D's preconditioned CG, which assumes fixed topology.
- **Caveat:** Newton has no spring-network or vehicle example, so beam trusses
  at 10 iterations are untested there.
