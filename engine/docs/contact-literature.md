# Intersection-free contact: what the literature offers a 2 kHz explicit node-beam solver (2026-11-26)

Status: **research notes for `continuous-contact-design.md`.** No code was changed. Each method
below is described from its primary source; where a detail is my reading or my own derivation
rather than the paper's, it says so. The papers were read from copies in the worktree's gitignored
`out/lit/` folder.

Our case, as the design states it: soft-body cars as node-beam structures with an explicit
integrator at 2,000 Hz (about 34 steps per 60 Hz frame), on the GPU with a CPU reference. Each car
has a triangle contact surface at its paint. Motion is at most about 3 cm per step. The goal is
that surfaces never intersect, car against car and a car against itself. It must create no
energy and cost about what today's contact costs.

## The short answer

- The guarantee does not need CCD root finding, iterated impulse rounds or impact zones.
  **Divide and Truncate (DAT, Chen, Hsu, Ayman and Macklin, SIGGRAPH 2026)** gives each nearby
  point-triangle and edge-edge pair a separating plane at the step start. It then scales each
  vertex's step so that the vertex stays on its own side of every plane.
- Each plane is fixed for the step and each side is convex, so this is one parallel pass with an
  atomic minimum per vertex. It cannot create secondary collisions and it does not iterate. For
  straight-line motion it holds over the whole step, not only at its end (derivation in §2).
- It is the generalisation of OGC's per-vertex bounds (§1). It is shipped in NVIDIA's open-source
  Newton engine (`SolverVBD`).
- It fits an explicit step better than any other method here, because the step's displacement
  `x1 - x0` is known before the plane is placed.
- What the guarantee needs:
  - a start state with no intersections;
  - every pair inside a query radius found;
  - a global cap on per-step motion of half the query radius times a relaxation factor (about
    3.5 cm at r_q = 8 cm). With motion at most 3 cm a step, this cap never binds.
- What it costs: a swept-box broadphase and one light pass over the pairs. The pair list is
  needed anyway.
- What it does not do: transfer momentum. Truncation alone stops the colliding nodes but does not
  push the struck side. So it is the failsafe, and today's dissipative proximity contact (with
  OGC's feature-region normals) stays the response. This is the same split as Bridson 2002:
  repulsion does nearly all the work and the failsafe is rare.

The recommendation (§10) replaces the draft's steps 3 to 5 (CCD rounds and impact zones) with
this failsafe. It keeps CCD only as a CPU test oracle.

---

## 1. Offset Geometric Contact (OGC), Chen et al., SIGGRAPH 2025

Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel. *Offset Geometric
Contact.* ACM TOG 44(4), 2025. Utah and NVIDIA.
PDF: <https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf>,
project page: <https://ankachan.github.io/Projects/OGC>.

**Contact model: feature blocks, the "Voronoi regions".** The mesh is offset by a radius r. It is
not a plain Minkowski offset, which merges layers that are close together. It is a union of
*blocks*, one per face of every dimension, and the blocks never overlap their own sub-faces
("local exclusiveness"):

- **Face (triangle) block U_t:** the prism offset along ±n_t by r (both sides, eq. 6). A point is
  inside if dis(x, t) ≤ r and its projection falls strictly inside the triangle (eq. 10).
- **Edge block U_e:** a cylinder of radius r about the edge, cut by two planes square to the edge
  at its ends and by one plane per adjacent face square to that face (eq. 9). This is the wedge
  between the two face normals.
- **Vertex block U_v:** a ball of radius r cut by planes through v square to each neighbouring
  edge, (x − x_v)·(x_v − x_v') ≥ 0 (eq. 8). A saddle vertex has an empty block. That is allowed,
  because an adjacent face or edge then covers the point.

**Vertex-face contact set.** F_OGC(v) = {a : x_v ∈ U_a, v ⊄ a} (eq. 12). For each triangle within
r, find the closest sub-face a (vertex, edge or interior). Add it unless it is already listed,
after checking the block test for a vertex or edge (Algorithm 1).

- Gap d = dis(x_v, a), penetration of the offset d_p = r − d (eq. 13).
- Normal = (x_v − c(x_v, a)) / d. By construction this is square to the feature, so the force is
  "always orthogonal", and the same point never gets two fighting forces from one triangle.

**Edge-edge.** The same construction on the edge-only manifold (eqs. 14 to 17). A pair {e1, e2}
with no shared vertex is active only if each edge's closest point to the other lies in a block of
that edge (its interior cylinder, or an end vertex's cut ball). The force acts at the two closest
points c(e1, e2) and c(e2, e1), along the line between them. Near-parallel pairs are filtered
smoothly, as in IPC.

- This is exactly the rule our `EDGE_CONE=1` measures: a pair only where its normal is inside
  both edges' cones.
- In discrete mode it lost crossed pairs (contact-design.md, "The exact rule alone"). In an
  intersection-free mode crossed pairs cannot exist, so OGC's rule becomes the correct one there.

**Energy.** A two-stage activation g(d, r) (eq. 18):

- quadratic (k_c/2)(r − d)² for τ ≤ d ≤ r;
- −k_c' log d + b below τ;
- τ = r/2, with k_c' and b chosen for C² continuity at τ.

Friction is IPC's lagged friction. The force is discontinuous where a point enters a face's block
through its open boundary on the concave side (limitations, Fig. 22a).

**The penetration-free guarantee: per-vertex conservative bounds** (§3.7, from Wu et al. 2020).

- Bound: b_v = γ_p · min(d_min,v, d^E_min,v, d^T_min,v) with 0 < γ_p < 0.5 (eq. 21). The three
  distances are:
  - d_min,v: v to the nearest triangle not containing v;
  - d^E_min,v: the smallest edge-edge distance over v's edges;
  - d^T_min,v: the smallest triangle-to-other-vertex distance over v's triangles.
- Claim: if every ‖x_v − x_v^prev‖ ≤ b_v, a penetration-free X^prev stays penetration-free
  (eq. 27).
- Reason: every point of a primitive is a convex combination of its vertices, so it moves no more
  than its fastest vertex. Two primitives at distance D, each moving less than D/2, cannot meet.
  This also holds along the straight path, not only at the end.
- The distances come out of the same BVH queries as contact detection: an atomic min per
  triangle, initialised to the query radius r_q ≥ r (Algorithms 1 and 2). In practice
  r_q = r + inertial displacement.
- Algorithm 3: detect at X^prev, solve, truncate each vertex's displacement back onto its ball,
  and re-detect only when more than γ_e·K vertices hit their bound (γ_e = 0.01; γ_p = 0.45 "to
  remain conservative with floating point rounding").

**Cost and GPU fit.** Implemented in NVIDIA Warp on an RTX 4090. Examples:

- 50 layers of cloth, 1M vertices, 1/1200 s step, 40 VBD iterations: 6.3 ms a step on average,
  11.5 ms at most;
- a twisted 10K-vertex cloth: 0.9 ms a step.

Against GIPC on a volumetric mat: 1,893 ms against 5.51 ms a step. Collision detection is about
10 % of the VBD run time.

**Fit to an explicit 2 kHz solver, and the problem.** The bound is **isotropic and absolute.**
It caps a vertex's motion in every direction, in the world frame, at under half the distance to
its nearest neighbour primitive.

- An explicit step is one "iteration", so a bound that binds is applied for the whole step.
- The authors' own limitation (Fig. 22b): two points falling side by side at speed are held to
  under d/2 a step even though they never approach. "The issue worsens as velocity increases."
- For a car: two crushed layers 5 mm apart moving together at 14 m/s (7 mm a step) would be
  clamped to 2.25 mm a step, about 4.5 m/s.
- This is the frame dependence our s29 trial ran into.

**Energy.** A truncated vertex keeps a fraction t of its step, so its kinetic energy scales by
t² ≤ 1. Kinetic energy is never added, but momentum is not conserved when a bound binds.

**Our earlier trial** (`DECISIONS.md` s29, AVBD soft solver) was not the OGC guarantee:

- the bounds covered cached contacts only;
- they had no edge-edge distances and no global r_q cap;
- the start state already had crossings (nodes "already behind at the step start").

It therefore says nothing about the guarantee itself, only that bounds on an intersecting start
do not help.

## 2. NVIDIA's follow-up: Divide and Truncate (DAT), 2026, and Newton

Anka He Chen, Jerry Hsu, Youssef Ayman, Miles Macklin. *Divide and Truncate: A Penetration and
Inversion Free Framework for Coupled Multi-physics Systems.* SIGGRAPH 2026 (arXiv:2604.15513,
April 2026). <https://arxiv.org/abs/2604.15513>. Implemented in Newton (Warp),
`newton/_src/solvers/vbd/` (`apply_planar_truncation_parallel_by_collision`,
`place_dat_division_plane`, `planar_truncation_t`): <https://github.com/newton-physics/newton>.

**Idea.** Give every primitive an exclusive region of space. A primitive whose vertices all stay
in its convex region cannot meet another (Theorem 3.1: a triangle is the convex hull of its
vertices). OGC's balls are "Isotropic-DAT". **Planar-DAT** uses one half-space per nearby pair.

**Per pair.**

1. **Plane.** For a separated vertex-triangle pair, take the closest point c on t, the normal
   n = normalize(x_v − c) and the gap g. For edge-edge, use the two closest points
   c(e,e') and c(e',e) and the normal along the line between them.
   - For two disjoint convex sets, the plane square to any closest-point segment, through any
     point strictly between its ends, separates them. So parallel or degenerate edge pairs need no
     special case.
2. **Placement from the known displacements ΔX** (eqs. 9 to 12). Approach amounts:
   - δ_v = max(−Δx_v·n, 0);
   - δ_t = max over the triangle's vertices of Δx_i·n, and 0;
   - the same with δ_e and δ_e' for edge-edge.

   The plane is placed so that each side's room is proportional to its own approach.
   - In Newton the plane sits at negative_support + λ·g·n with λ = δ_neg / (δ_pos + δ_neg),
     clamped to [0.05, 0.95]. It falls back to 0.5 when nothing approaches or g < 2ε.
   - The arXiv v1 text writes eq. 11 from the vertex side and appears to give the room to the
     other side. Newton's code is the evidently intended form. **Unverified: test it.**
3. **Truncation per vertex and plane:** with signed distances s0 at the start and s1 at the end,
   - if s1 ≥ 0 keep the step;
   - otherwise t = γ_r·s0/(s0 − s1).

   Each vertex takes the atomic **min** of t over all its pairs and scales its whole Δx by it.
   Newton's default γ_r is 0.85.
4. **Pairs beyond r_q** are not listed. Every vertex's |Δx| is capped at 0.5·γ_r·r_q (Algorithm 2,
   line 10), so unlisted pairs cannot meet.

**Guarantee.** No intersection at the end of the step, from a start with none.

- My derivation for straight-line motion over the step: each vertex's signed distance to its
  fixed plane is linear in time and ≥ 0 at both ends, so it is ≥ 0 throughout. Each primitive is
  the convex hull of its vertices at every instant, so the two stay on their sides at every
  instant.
- So this is continuous-time safe with no root finding. Newton adds a float32 separation epsilon
  scaled by coordinate magnitude (`dat_separation_epsilon`) and certifies the plane against all
  primitive vertices.
- Rigid bodies (curved paths) need sampling, K = 8 at 1/600 s, plus bisection or interval
  arithmetic (§3.6).
- Newton's docs: DAT "requires initially separated geometry". Where no separating plane can be
  certified, the primitives freeze. A contact-buffer overflow makes the coverage incomplete, and
  Newton asks you to stop the run.

**Why it suits us.**

- There are no secondary collisions and no iteration. The planes are fixed from the start state,
  so one vertex's truncation cannot invalidate another's.
- min is exact and independent of order, so a bit-exact mode is unaffected.
- The explicit step gives ΔX = x1 − x0 directly. DAT is documented as a post-process after any
  solver.
- Sliding and separating motion is untouched, unlike OGC: δ is 0 for a side moving away or
  sideways.
- The paper reports Planar-DAT about 10 % dearer per iteration than isotropic, and CCD-based
  truncation "∼16 ms per iteration — over 10× slower than either DAT scheme". In its tests
  Planar-DAT preserves 1.4 to 178× more displacement than isotropic and 265 to 512× more than
  global CCD (Table 2).

**Its stated limits.**

- Pairs beyond r_q fall back to the isotropic cap.
- Truncation "does not preserve momentum or energy".
- Rigid rotation needs sampling.
- It inherits OGC's force discontinuity at concave regions.

**One gap for crashes, and a fix (my derivation, not in the paper).** The planes are fixed in the
world frame. A layer that is followed at speed by another along n is still capped at γ·g a step,
even when the leading layer moves away just as fast.

- The fix: let each pair's plane translate by a common normal motion u·n over the step, for
  example the mean of the two sides' normal displacements. Each signed distance (x + τΔx − p −
  τu n)·n stays linear in τ, so the guarantee is unchanged. Only relative approach then counts.
- This must be tested before it is relied on.
- The global cap for unlisted pairs stays absolute. It is harmless when r_q ≥ 2·(max |Δx|)/γ_r,
  about 7 cm for 3 cm a step.

## 3. Speculative contacts and speculative CCD (PhysX, Jolt, Box2D v3)

**Algorithm (rigid).** Contacts are generated before touching, within a margin that grows with
velocity. The contact row is a velocity constraint that allows approach up to the current gap:
v_n ≥ −s/h for separation s > 0, and the usual push-out for s ≤ 0.

- **Box2D v3** `contact_solver.c`: "Speculative bias, zero when overlapped",
  `velocityBias = s > 0 ? s * inv_h : 0`. `B2_SPECULATIVE_DISTANCE = 4 * B2_LINEAR_SLOP`, which is
  2 cm. The v3.0 release combines speculative contacts with time-of-impact for fast bodies
  against static ones. Bullet against dynamic bodies is "not as robust".
  <https://box2d.org/posts/2024/08/releasing-box2d-3.0>,
  <https://github.com/erincatto/box2d/blob/main/include/box2d/constants.h>.
- **Jolt** `PhysicsSettings::mSpeculativeContactDistance = 0.02 m`. Its comment warns that a
  large value gives ghost collisions, "as speculative contacts are based on the closest points
  during the collision detection step which may not be the actual closest points by the time the
  two objects hit". `LinearCast` adds a shape cast and "time stealing".
  <https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/PhysicsSettings.h>.
- **PhysX** `eENABLE_SPECULATIVE_CCD` inflates contact offsets by motion and relies on the solver.
  The docs say it "can fail to ensure objects do not pass through each-other", for example a body
  the solver accelerates through others within one step.
  <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>.
- Catto, GDC 2013: <https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf>.

**Guarantee.** None. The contact feature and normal are fixed at detection, so features that
change during the step (sliding off an edge, rotation) are missed. Stale normals give ghost
contacts.

**For us.**

- The node-beam form is a velocity filter on listed pairs. A mass-weighted Jacobi impulse enforces
  (v_rel·n) ≥ −(g − d)/h. The pair list and broadphase are the same as for DAT.
- It is valuable as the momentum-conserving part: it transfers a hit's impulse to the struck
  panel, which truncation cannot do.
- It is dissipative: restitution 0, and the impulse only removes approach beyond the gap. With
  Jacobi it needs mass splitting or relaxation to avoid overshoot, as in our existing sweeps.
- Cost: k Jacobi sweeps over the pair list.

## 4. Continuous collision detection for point-triangle and edge-edge

**Root finding (Provot 1997; Bridson et al. 2002).**

- With straight-line motion, the four points are coplanar when
  (x21 + t v21) × (x31 + t v31) · (x41 + t v41) = 0, a cubic in t.
- Each root in [0, Δt] is then checked with a proximity test, using a 10⁻⁶ m tolerance in
  Bridson.
- In plain floating point it has false negatives. Wang et al. 2021 found 5,184 vertex-face and
  2,317 edge-edge false negatives on their simulation set, at 7.5 and 0.23 µs a query.
- Brochu, Edwards and Bridson 2012 (exact root parity) and Tang et al. 2014 (Bernstein sign
  classification) still show false negatives on handcrafted degenerate cases (RP 5/8, BSC 13/47).
  TightCCD has none, but many false positives (95,638 on the simulation vertex-face set).

**Tight Inclusion (Wang, Ferguson, Schneider, Jiang, Attene, Panozzo, TOG 2021)**
<https://arxiv.org/abs/2009.13349>, code <https://continuous-collision-detection.github.io/tight_inclusion>.

- Parameterisation:
  - F_vf(t,u,v) = p(t) − ((1−u−v)v1(t) + u v2(t) + v v3(t));
  - F_ee(t,u,v) = ((1−u)p1 + u p2) − ((1−v)p3 + v p4);
  - both over [0,1]³.
- Inclusion: the axis box of F at the 8 corners of a parameter box, provably tightest.
- It bisects the widest dimension breadth-first, with a priority queue by t.
- Each box is accepted when narrower than δ (10⁻⁶) or inside the floating-point ε-box, with
  ε_vf = 6.66×10⁻¹⁵·γ³ and γ = max(|x|max, 1).
- **Minimum separation:** the box is inflated by d in the L∞ norm.
- **Guarantee:** no false negatives (0/0 on all sets), false positives possible. It is 0.74 to
  0.78 µs a query on the simulation set.
- At the iteration cap (10⁶) it returns a conservative t, but accuracy is no longer guaranteed.
- GPU: *Scalable and Conservative CCD* (Belgrod et al., arXiv:2112.06300) ports it with a
  queue-based bisection and an axis-sweep broadphase, and reports an RTX 3080 Ti comparable to a
  64-core CPU.

**Additive CCD (ACCD; Li, Kaufman and Jiang, C-IPC, SIGGRAPH 2021)**
<https://arxiv.org/abs/2012.04457>.

- Conservative advancement. Subtract the mean displacement of the stencil (valid in any frame),
  then l_p = max ‖p_i‖ on one primitive + max ‖p_j‖ on the other.
- The distance cannot fall faster than l_p, so a step of (d − ξ)/l_p is safe:
  - first t_l = (1−s)(d² − ξ²)/((d + ξ) l_p);
  - then 0.9·(d² − ξ²)/((d + ξ) l_p);
  - stop when (d² − ξ²)/(d + ξ) < s·(initial value);
  - s = 0.1, ξ = thickness.
- It uses only distance evaluations, with no polynomial roots. It returns a lower bound on the
  time of impact, so it is never late.
- Iteration count: the paper gives no typical number. By its construction it is small for head-on
  approach and grows when l_p overstates the true approach rate (sliding, spinning). It is
  unbounded in theory for "diverging displacement fields", which the paper says do not occur in
  elastodynamics.
- GPU: branch-light, a few dozen flops an iteration, but the iteration count varies between lanes.

**For us.** CCD gives a time of impact, not a response. Every response built on it (impulses at
t*, line search) must re-run CCD on the corrected motion until nothing is found. That loop is the
cost and the unbounded part.

- DAT makes the per-step guarantee without it.
- We keep ACCD (simple, conservative) in the **CPU test oracle**: every step of every acceptance
  scene, every point-triangle and edge-edge pair, assert no time of impact at zero distance.
- Tight Inclusion is the stricter oracle if ACCD's iteration count is a problem offline.

## 5. Bridson, Fedkiw and Anderson 2002; Harmon, Vouga, Tamstorf and Grinspun 2008

**Bridson et al. 2002**, *Robust Treatment of Collisions, Contact and Friction for Cloth
Animation*, SIGGRAPH (<https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf>). Per step:

1. Internal dynamics give x̄ⁿ⁺¹. Average velocity v̄ⁿ⁺½ = (x̄ⁿ⁺¹ − xⁿ)/Δt.
2. **Proximity at xⁿ** (thickness h ≈ 1 mm, AABB tree, boxes enlarged by h), point-triangle and
   edge-edge:
   - an inelastic impulse I_c = m v_N/2 when approaching;
   - plus a spring repulsion on the overlap d = h − (x4 − Σ w_i x_i)·n̂, impulse
     I_r = −min(Δt k d, m(0.1 d/Δt − v_N)), applied only if v_N < 0.1 d/Δt;
   - the spring k is matched to the stretch springs;
   - Coulomb friction from the repulsion impulse.
   - Impulses are spread by barycentric weights: Ĩ = 2I/(1 + w1² + w2² + w3²) for point-triangle,
     and the analogue with a and b for edge-edge.
   - Jacobi application uses relaxation 0.25 or division by the contact count.
   - The edge-edge "normal" is the direction between the closest points (§6).
3. **Geometric collisions:** cubic-root CCD along the straight paths, tolerance 10⁻⁶ m (1,000×
   smaller than h). Inelastic impulses are applied and the corrected paths re-checked, with the
   swept-box tree refitted each round.
4. **After "a few iterations", rigid impact zones** (corrected from Provot 1997):
   - nodes of each collision are merged into zones, union-find style;
   - each zone moves rigidly, keeping linear and angular momentum (ω = I⁻¹L about the zone's
     centre);
   - positions follow an exact rotation, so lengths are kept, which Provot's formula lacked;
   - zones grow until the step is collision-free.

Guarantee: "the cloth is separated by at least the rounding error tolerance at every time step
and never self-intersects during time steps" *if* collisions are resolved. The zones are what make
that terminate.

**Harmon et al. 2008**, *Robust Treatment of Simultaneous Collisions*, SIGGRAPH
(<https://www.cs.columbia.edu/cg/pdfs/RTSC.pdf>).

- It replaces rigid impact zones with **inelastic projection**. Collect an impact zone's
  constraints C (point-triangle: N·[x4 − Σα_i x_i]; edge-edge: N = edge cross product, parallel
  pairs skipped) with gradients ∇C at the time of impact.
- Solve ∇C M⁻¹ ∇Cᵀ λ = ∇C q̇ and set p' = p + ∇Cᵀλ (eqs. 3 to 5). This is the M-orthogonal
  projection that zeroes every constraint's normal velocity and keeps tangential motion.
  Friction is per zone.
- Zones grow as new collisions appear (Algorithm 1). The system can be ill-conditioned: "SVD or
  GMRES is strongly recommended". It runs at most 15 % slower than rigid zones.
- Rigid zones fail if a zone holds scripted (infinite-mass) vertices that disagree, which forces a
  halved step. Rigid zones also visibly lock sliding cloth.

**Energy.**

- Both responses are M-orthogonal projections of velocity (onto rigid motions, or onto the
  constraints' null space), so kinetic energy never rises.
- Both remove relative motion: rigid zones remove all of it, which is "unphysically dissipating"
  and makes free-flowing fabric lock.
- Positions then follow the new velocities, so beam potential energy is not bounded by the
  argument. It must be measured.

**For us.**

- It is principled, but on a GPU it means an unbounded loop: CCD, impulses, CCD again, then
  connected components and per-zone reductions or a small dense solve.
- The cost varies with the crash, and it needs fixed round caps and fallbacks. That conflicts with
  bounded kernels and single-graph frames.
- In a hard crash the zone is the crush zone itself. Rigidifying it for a step removes exactly the
  relative motion that crushes metal: a physics change, not only an artifact.
- Planar-DAT has the same guarantee in one bounded pass, and it removes only the approaching part
  of the bad vertices' steps.

## 6. IPC and GPU IPC (GIPC 2024): what it would cost us

IPC (Li et al., SIGGRAPH 2020) is an implicit Euler step solved by projected Newton.

- Contact is a log barrier b(d) = −(d − d̂)² ln(d/d̂) on every point-triangle and edge-edge pair
  within d̂.
- Every line search is filtered by a CCD step bound (ACCD in C-IPC), and every energy evaluation
  needs distance computation over all pairs.
- GIPC (Huang, Chitalu, Lin and Komura, TOG 43(2) 2024, <https://arxiv.org/abs/2308.09400>)
  gives analytic PSD barrier eigensystems and a fully GPU Gauss-Newton with PCG.

Measured against OGC on a 15K-vertex volumetric mat: GIPC 1,893 ms a step on average and over
20 s at most, against 5.51 ms for VBD + OGC (OGC §5.4).

For us:

- it replaces the explicit node-beam integrator (and BeamNG parity) with an implicit optimisation;
- a barrier's stiffness is unbounded as d → 0, which an explicit integrator cannot step stably;
- even at a hypothetical 1 ms a step, 34 steps is 34 ms a frame for contact alone, against today's
  1.0 to 3.7 ms a frame for the whole edge pass in two-car impacts (contact-design.md, CUDA T4).

**Not applicable.**

## 7. Vertex Block Descent (VBD), Chen, Liu, Yang and Yuksel, SIGGRAPH 2024

<https://arxiv.org/abs/2403.06321>.

- Collision handling: DCD at the step start for vertices already penetrating, and CCD (only in
  the first iteration in the experiments, n_col = n_max) for vertex-triangle and edge-edge.
- Detected pairs stay as force elements until the next detection.
- Energy E_c = ½ k_c d² on the penetration d = max(0, (x_b − x_a)·n̂), with a frozen normal and IPC
  friction.
- Graph colouring covers material forces only. Contacts between same-colour vertices are updated
  Jacobi-style, and colliding vertices skip Chebyshev acceleration.
- The paper states it "cannot guarantee penetration-free results". That is why OGC, then DAT, were
  built on top of it.

For us VBD is an integrator (our AVBD soft solver is its relative). Its own collision handling is
weaker than what we have.

## 8. GPU broadphase for deformables

- **BVH.**
  - LBVH: Morton codes, radix sort, a parallel hierarchy build (Karras, *Maximizing Parallelism in
    the Construction of BVHs, Octrees, and k-d Trees*, HPG 2012).
  - Refit updates the boxes bottom-up in one pass but degrades as the mesh deforms. A crash is
    exactly that, so rebuild per step or on a quality test.
  - OGC and DAT use a GPU BVH: Warp's `wp.Bvh`, BVH queries in Newton. Bridson 2002 builds a
    topological box tree once and refits it.
- **Grid or spatial hash** (Teschner et al., *Optimized Spatial Hashing for Collision Detection of
  Deformable Objects*, VMV 2003; Le Grand, *Broad-Phase Collision Detection with CUDA*, GPU Gems
  3 ch. 32: sort by cell key).
  - Best when primitives are about one size and the cell is about that size plus the query
    radius. A generated car's triangles are roughly uniform, so a sorted uniform grid is
    competitive and simpler than a BVH.
- **Sort and sweep.** Belgrod et al. (above) found that sweeping along one axis beats tree
  approaches on modern GPUs for CCD candidate sets.
- **Self-collision culling:** normal-cone tests on patches (Volino and Magnenat-Thalmann 1994;
  Provot 1997) skip low-curvature regions. They are exact only for static proximity. Bridson notes
  they do not cover swept tests.
- **Frame-coherent pair lists** (molecular dynamics Verlet lists).
  - Build the candidate list with radius R = r_q + skin. It stays exact while every primitive has
    moved less than skin/2 since the build.
  - We already do this for edge candidates (12 cm skin). contact-design.md measured that pile
    rebuilds dominate (58 ms of 87 ms a frame for a pile of 8) and named per-owner rebuilds as the
    lossless lever.
  - At 2 kHz and 3 cm a step, a 12 cm skin lasts at least 2 steps under worst motion and many
    steps in a pile.
- **What DAT needs from it:** every point-triangle and edge-edge pair whose gap at x0 is below r_q,
  with r_q ≥ 2·max |Δx|/γ_r. Equivalently and tighter: every pair whose swept boxes, inflated by
  the margin, overlap.
  - A pair whose gap exceeds the sum of both sides' maximum approach divided by γ_r gets t = 1, so
    it can be dropped exactly. This is a lossless cull.

## 9. Side by side

| Method | Guarantee | Per step, per pair | Iterates? | GPU fit | Energy |
|---|---|---|---|---|---|
| OGC bounds | Yes, from a clean start, pairs within r_q, cap for the rest | Distances (already in contact detection), atomic min | No (re-detect when bounds bind) | Good | KE ≤; momentum lost when binding; isotropic damping, frame-dependent |
| **Planar-DAT** | **Yes, same conditions; continuous for straight paths** | Plane from closest points (≈ the proximity pass), 4 dot products, atomic min | **No** | **Very good**, order-independent | KE ≤; momentum lost when binding; only approach is cut |
| Speculative impulses | No | One velocity row, k Jacobi sweeps | Sweeps | Good | Dissipative, momentum kept |
| CCD (ACCD / TI) | Detection only | 10s of flops × iterations / bisection | Yes, plus response rounds | Varies by lane | n/a |
| Bridson + rigid zones | Yes (terminates via zones) | CCD rounds + union-find + rigid fit | Unbounded | Poor | KE ≤, dissipates relative motion |
| Harmon projection | Yes, if the zone solve succeeds | As Bridson + per-zone linear solve | Unbounded | Poor | KE ≤, keeps sliding |
| IPC / GIPC | Yes | Barrier + Newton + CCD line search | Heavy | Possible but 10³× too slow for us | Implicit-Euler damping |
| VBD (own) | No | Penalty from DCD/CCD | Solver iterations | Good | Penalty |

## 10. Recommendation for our case

Use **Planar-DAT as the intersection-free failsafe**, inside today's explicit step, with today's
dissipative proximity contact as the response. Drop the CCD rounds and impact zones from the
draft.

**Each step, in continuous mode:**

1. **Start state is intersection-free.** This is a hard precondition, checked at spawn and in
   tests: the lint's `skin_self_crossing` must be 0 and spawned cars must not overlap. It includes
   the contact surface's own offset corners, where offsetting inward corners to the paint can make
   the surface cross itself.
2. **Pairs.** All point-triangle and edge-edge pairs (sharing no node) within r_q at x0, from the
   frame-coherent list (§8). Set r_q ≥ 2·v_max·h/γ_r + d, about 7 to 8 cm for 3 cm a step and
   γ_r = 0.85.
   - v_max is an explicit engine-wide setting. A step whose max |Δx| exceeds 0.5·γ_r·r_q is
     reported as an error and counted, never silently clamped (AGENTS.md).
3. **Proximity contact (unchanged law, §37)** on the pairs within the band.
   - Use OGC's feature-region rule to choose each pair's feature and normal: a face's prism, an
     edge's wedge, a vertex's cut ball.
   - This is the `EDGE_CONE=1` exact rule, which is correct once nothing is crossed. It removes the
     washboard normals behind the jitter.
   - In the same pass, store each listed pair's plane data at x0: n, the supports and the gap, as
     4 to 8 floats a pair.
4. **Integrate** as today, giving Δx = x1 − x0.
5. **Optional, decided by measurement: speculative inelastic impulses.** Do this only if step 7's
   count shows truncation doing real work in hard hits. Jacobi with mass splitting on pairs
   predicted to close past d, transferring momentum to the struck side.
6. **Planar-DAT.** One pass over the pairs: compute δ and λ, with frame-relative planes (§2, to be
   tested), and atomic-min t per vertex. One pass over the nodes: x1 = x0 + t·Δx, v = t·Δx/h (or
   the integrator's equivalent).
7. **Count** truncated nodes, the kinetic energy and momentum removed, and cap hits. These are the
   mode's health measures, reported like `JITTER`.
8. **Rigid bodies:** DAT's sampled-trajectory form against their shapes. The ground and terrain
   also need planes against the heightfield triangles if "never intersect" includes them.

**What we add:**

- the plane store in the proximity pass;
- one pair pass and one node pass;
- swept or skin-based broadphase radius r_q;
- the health counters;
- a CPU ACCD oracle in the tests (assert no time of impact over every step);
- the start-state gate.

**What we drop in this mode:**

- the draft's CCD narrowphase at run time, its "few rounds", and rigid impact zones;
- anti-clip side memory, `EDGE_DEPTH`, the cone clamp (`EDGE_CONE=2`) and the band as the last
  line of defence (as the draft already says).

**Where the cost goes.**

- Mostly the broadphase: r_q grows from the 25 mm band to about 7 to 8 cm, so candidate pairs
  grow by up to about (r_q/r)² where triangles are small against r_q. Mitigations:
  - the exact cull in §8 keeps only pairs that can actually approach;
  - per-owner skin rebuilds keep the rebuild rate down.
- The DAT pass is a few dozen flops and four atomics a pair, cheaper than the proximity pass it
  follows.
- Expect contact cost close to today's at rest (few truncations, the same pairs) and dominated by
  the broadphase in crashes. That is not the draft's 2 to 3 times, which assumed CCD rounds.
  **This is an estimate:** measure pair counts and pass times on the CPU first (draft stage 2),
  then on CUDA.

**Energy.**

- Contact forces are today's dissipative law.
- Impulses, if added, are inelastic.
- Truncation only scales steps by t ≤ 1, so it never adds kinetic energy. It does remove momentum
  when it binds, which is why the counters exist.
- Total energy is not provably monotone because beams see the truncated positions. The energy
  ledger tests judge it.

**Open questions to settle by test, not by assumption:**

- λ's form (Newton against the arXiv text);
- frame-relative planes;
- how often truncation binds in the eight acceptance scenes;
- whether speculative impulses are needed;
- the pair-count growth with r_q on the generated sedan;
- f32 precision far from the origin: the separation epsilon must scale with coordinate magnitude,
  as Newton's does, or positions must be made pair-local.

**Unverified details flagged above:**

- Wu et al. 2020's exact bound (abstract only read);
- GIPC numbers (taken from the OGC paper, not GIPC);
- ACCD's typical iteration count (not reported);
- DAT's eq. 11 direction.

## Sources

- Chen et al. 2025, Offset Geometric Contact, TOG 44(4): <https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf>
- Chen, Hsu, Ayman, Macklin 2026, Divide and Truncate, SIGGRAPH 2026: <https://arxiv.org/abs/2604.15513>
- Newton (NVIDIA, Warp), `SolverVBD`, Planar-DAT kernels: <https://github.com/newton-physics/newton>
- Wu, Wu, Yang, Wang 2020, A Safe and Fast Repulsion Method for GPU-based Cloth Self Collisions, TOG 40(1): <https://wanghmin.github.io/publication/wu-2020-sfr/>
- Kikuchi, Kanai 2026, Convex Collision-Free Regions (per-vertex convex feasible regions; not read beyond the abstract): <https://arxiv.org/abs/2607.26901>
- Box2D v3: <https://box2d.org/posts/2024/08/releasing-box2d-3.0>, <https://github.com/erincatto/box2d>; Catto GDC 2013: <https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf>
- Jolt `PhysicsSettings.h`: <https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Physics/PhysicsSettings.h>
- PhysX 5.4 Advanced Collision Detection: <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>
- Li, Kaufman, Jiang 2021, Codimensional IPC (ACCD): <https://arxiv.org/abs/2012.04457>
- Wang et al. 2021, Large-Scale Benchmark and Inclusion-Based CCD: <https://arxiv.org/abs/2009.13349>
- Belgrod et al., Scalable and Conservative CCD (GPU): <https://arxiv.org/abs/2112.06300>
- Bridson, Fedkiw, Anderson 2002: <https://www.cs.ubc.ca/~rbridson/docs/cloth2002.pdf>
- Harmon, Vouga, Tamstorf, Grinspun 2008: <https://www.cs.columbia.edu/cg/pdfs/RTSC.pdf>
- Huang, Chitalu, Lin, Komura 2024, GIPC: <https://arxiv.org/abs/2308.09400>
- Chen, Liu, Yang, Yuksel 2024, Vertex Block Descent: <https://arxiv.org/abs/2403.06321>
- Karras 2012, HPG (LBVH); Teschner et al. 2003, VMV (spatial hashing); Le Grand 2007, GPU Gems 3 ch. 32 (cited from memory, not re-read).

## 11. Simultaneous impact: Generalized Reflections (2012) and its termination proof (2017)

Read 2026-11-27 (night), from copies in `out/lit/` (gitignored), after measuring that the
head-on's rounds fail to settle mostly at the impact itself, at crowded nodes held by rows with
nearly opposite normals (`continuous-contact-design.md`, "Three diagnostic measurements").

**Smith, Kaufman, Vouga, Tamstorf, Grinspun 2012**, *Reflections on Simultaneous Impact*, TOG
31(4) (<https://www.cs.columbia.edu/cg/rosi/rosi.pdf>).
- Five desiderata for a multi-impact operator: break-away (BRK), symmetry (SYM), kinetic energy
  bounded (KIN), momentum (MOM), one-sided impulses (ONE). The inelastic LCP (our projection at
  e = 0) satisfies all only for c_r = 0; with c_r > 0 it sticks (no break-away). Gauss-Seidel
  pairwise breaks symmetry; Jacobi breaks energy. Their table: no prior model has all five.
- **Generalized Reflections (GR):** repeat { V = constraints whose normal velocity is
  approaching; solve the LCP (a QP) on V alone with the elastic factor 1 + c_r; apply } until V is
  empty. Each pass is the unique reflection of the velocity in the span of the violated normals.
- **Restitution without inelastic collapse:** iterated dissipative impact (pairwise or GR with
  c_r < 1) can need infinitely many iterations, the range of c_r where this happens grows to 1
  with the number of bodies. Their model solves the elastic problem (GR, c_r = 1) and the purely
  inelastic one (one LCP, well posed) separately and takes v⁺ = (1 − c_r) v₀⁺ + c_r v₁⁺. All five
  desiderata hold.
- Time stepping (their Alg. 4): free flight to find the active set, the impulses computed from
  the step-start state (q_t, p_t), then the step. Rigid bodies and balls; CPU interior-point QPs,
  up to 2M constraints; 11-15 GR iterations a step in their table.
- Limits they state: sustained contact (drift-free resting contact) not addressed; GR's
  termination was open.

**Vouga, Smith, Kaufman, Tamstorf, Grinspun 2017**, *All's Well That Ends Well: Guaranteed
Resolution of Simultaneous Rigid Body Impact*, TOG 36(4)
(<https://www.cs.utexas.edu/~evouga/uploads/4/5/6/8/45689883/term-revised.pdf>).
- Abstract, in short: iterative impact resolution needs no failsafe. For frictionless,
  energy-conserving operators with a minimal set of properties (pairwise Gauss-Seidel and GR
  included), finite termination is fully characterised.
- **The only obstruction is implicit equality constraints:** constraint gradients that together
  force the velocity into a subspace (for example two opposite normals on one DoF, n·v ≥ 0 and
  −n·v ≥ 0). They give an algorithm to detect and remove them, and prove termination in
  floating point.
- Each property is necessary. With energy removed at each application (inelastic iterations),
  inelastic collapse makes termination fail; dissipation is added as in 2012, by interpolating
  two terminating solves.
- Their Discussion names the practical workaround they make unnecessary: abort after a fixed
  count and either leave collisions or apply Provot's rigid impact zones, "artificial dissipation
  and sticking".

**For us (my reading, to be checked by measurement):**
- Our velocity rows are an inelastic projection iterated in rounds: the class the 2017 paper shows
  can fail to terminate (inelastic collapse). Our rounds' failure ends in rigid zones: their
  "failsafe".
- Our stuck clusters, rows on shared nodes with nearly opposite normals, are near-implicit
  equality constraints: the one obstruction the proof names, with a detection and removal step.
- Our nodes are point masses (a diagonal M): the simplest case of their setting.
- Not covered by the papers: our outer CCD rounds re-linearise the constraints as the closest
  points slide within a step; theirs are fixed at an instant. Sustained, pressed contact is out
  of their scope. Their solves are CPU QPs, not real time; GPU cost is unknown.

**What transfers and what does not (checked against our solver, `ccd_vel.rs`):**
- The papers' constraints are homogeneous: J v ≥ 0 at contact (g = 0). The energy proof (each
  pass is a reflection on the ellipsoid of constant kinetic energy) needs that. Our rows are
  affine, J v ≥ −b with b = max(s₀ − ξ, 0)/h: a pair may still close by its gap this step. A
  reflection about an affine bound does not keep kinetic energy, so the proof does not cover our
  rows as they are.
- Positions follow velocities (x₁ = x₀ + h·v). An elastic reflection applied at the step's start
  ends the pair at s₀ + h·|u| instead of the true 2ξ − s₀ + h·|u| (it bounces from up to 2.8 cm
  early at 55 m/s). That is their Alg. 4 too, but there it costs nothing (rigid balls, no beams).
  In ours the beams see the difference, the position-step energy error measured on 11-26.
- This is consistent with the earlier measurement that Newton's law in the rows (J v⁺ ≥ −e·u⁻ on
  coupled rows, u⁻ from before the kick) did +591 kJ of work at head 100.
- What would make it apply: impacts resolved at their time of impact within the step (a
  contact's own instant, as asynchronous methods do), or rows that are homogeneous (b = 0) with
  the gap's closing done by free flight up to the contact. Both are open designs here, with costs
  not measured.
