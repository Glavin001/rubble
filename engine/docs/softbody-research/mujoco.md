# MuJoCo and MuJoCo Warp: flex bodies

Read-only review (2026-11-02) of MuJoCo's flex implementation and its GPU port,
MuJoCo Warp.

## 1. Model

- **Bodies and coordinates:** a flex is ordinary bodies (its vertices, each with
  slide degrees of freedom) joined by massless capsules, triangles or tets. It
  is solved in generalised coordinates with sparse Jacobians per edge.
- **Elasticity, two choices:**
  - edge equality constraints, lumped and suited to large steps;
  - or continuum St. Venant–Kirchhoff, with a per-element metric precomputed on
    changes in squared edge length.

  Tets can also use stable Neo-Hookean, with Rayleigh damping.
- **2D bending:** Wardetzky's quadratic bending, with a constant 4×4 stencil
  per hinge plus a term for curved rest shapes. The force is linear and the
  Hessian constant.
- **Integrator:** flex elasticity requires the `discrete` integrator.
  - The constraint solve uses M + hD + h²K, so stiffness is implicit in
    position.
  - Flex forces a monolithic CG or Newton solve, with no islands and no PGS.

## 2. Stiffness

- **Example models:**
  - soft, with Young's modulus 1e3–3e7 Pa;
  - timesteps 0.5–2 ms;
  - solved with CG.
- **Rigid look:** comes from edge equality constraints or the discrete metric.
- **Damping:** implicit rows damp more as the step grows, which suits a dead,
  crumpling look.
- **IPC contact:** an experimental flex mode (augmented Lagrangian plus
  continuous collision detection). It is frictionless and works with CG only.

## 3. Plasticity and fracture

None. Rest lengths, metrics and stencils are fixed in the model.

## 4. Collision

- **Per element:**
  - GJK or MPR;
  - forces spread over up to 8 bodies;
  - elements that share a body are skipped.
- **Self-collision:** an SAP or BVH midphase, using only surface elements
  (active layers).
- **MuJoCo Warp:**
  - flex–geometry, self and flex–flex contact all share one segmented
    sort-and-sweep;
  - a separating-axis prefilter on triangles;
  - farthest-point reduction down to a contact budget.

## 5. Batching in MuJoCo Warp

- **Layout:**
  - one model and nworld worlds, in 2D [world, item] arrays;
  - kernels launch over (nworld, nitems);
  - per-world parameter overrides by modulo.
- **Buffers:** fixed-size contact and constraint buffers with per-world overflow
  bits.
- **Solver:** one per world, with done flags inside a captured while-loop, so
  the host never syncs.
- **Stated limits:** it is built for throughput, not latency. Single mechanisms
  beyond about 60 degrees of freedom scale poorly. A 1,000-node car is about
  3,000.

## 6. Lessons for Rubble

- **Layout:**
  - batch by (vehicle, item);
  - structure-of-arrays buffers sized at load;
  - overflow bits that fail loudly;
  - one recorded stream with convergence flags kept on the GPU.
- **Solver:**
  - avoid a global generalised-coordinate solve per vehicle;
  - make stiffness implicit in position for any implicit tier.
- **Sheet metal:**
  - Wardetzky constant-stencil bending;
  - when a panel yields, update its curved rest term.
- **Plasticity and breaking:** mutable per-edge rest length and curvature, and
  fixed-size masks of live elements.
- **Collision:**
  - one segmented sort-and-sweep shared by self and inter-vehicle contact;
  - surface-only collision;
  - skip pairs that share a node.

  Contact budget reduction is lossy, so it can only be an explicit option.
- **Not a model for crashes:** MuJoCo's frictionless penalty and IPC contact.
