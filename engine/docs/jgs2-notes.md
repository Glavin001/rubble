# JGS2 (and JGS2-GQ) for Rubble's node–beam solver: notes

Phase 7 of `softbody-beamng-parity.md`. Sources (local only, `inspiration/` is
gitignored, nothing is copied from it):

* Lan et al., *JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU
  Elastodynamics*, ACM TOG 2025, arXiv 2506.06494 (`inspiration/papers/`). Read in full.
* Guo et al., *JGS2-GQ: Training-free 2nd Jacobi with Gaussian Quadrature*, ACM TOG
  45(4) art. 123, SIGGRAPH 2026, doi 10.1145/3811274. **Not read yet** (ACM blocks
  scripted download); only its abstract is known.
* `CallumFerguson/JGS2-implementation` (unofficial WebGPU/TypeScript, no licence:
  read only). No official code exists for either paper.

## The method (JGS2)

The step is `argmin E(x)`, E = inertia + elasticity, solved by per-vertex block
updates (as VBD). **Overshoot**: a vertex minimising only its own energy E_i raises
the energy of the rest of the body, which is why VBD-like sweeps converge slowly on
stiff structures.

JGS2 gives each vertex i a model of the rest of the body, E_Ci = E − E_i, through a
**perturbation basis** U_i: when x_i moves by δx_i, the complementary DOFs move by
U_i δx_i, the static response of the body with x_i prescribed:

* U_i = −H_{Ci,Ci}⁻¹ H_{i,Ci}ᵀ (Eq. 8). With the exact current Hessian, the local
  solve (H_ii + U_iᵀ∇²E_Ci U_i) δx_i = −(g_i + U_iᵀ∇E_Ci) equals the vertex's share of
  the global Newton step exactly (Eqs. 10–13: a Schur complement). That is the
  "second-order" claim.
* Practical version: U_i from the **rest** Hessian H̄ (inertia M/h² + rest stiffness),
  co-rotated per vertex, φ̃ = R Ū_i R_iᵀ δx_i (Eq. 14), with R from a polar
  decomposition of each vertex's local deformation gradient, every iteration.
* Precomputation (Eqs. 19–23): factor H̄ once; Ū_i = H̄⁻¹S_iᵀ (S_i H̄⁻¹ S_iᵀ)⁻¹, three
  solves per vertex (the source block is exactly I).
* The complementary terms H̃_ii = UᵀHU and g̃_i = Uᵀ∇E_Ci would cost O(N) per vertex,
  so **Cubature** (Eqs. 17–18) approximates them with 4–6 weighted sample elements
  per vertex, weights fitted by NNLS against training poses (low-frequency
  eigenvectors of H̄). Only a good "damping Hessian" is needed, so sparse samples do.
* Contacts (penalty or IPC) enter as unconstrained energies; with IPC a per-vertex
  line search is needed (Newton's quadratic model fails near the barrier).
* Results: 30–60 iterations per step converge million-element scenes (‖Δx‖ < 1e-3);
  15–40× fewer than VBD or 2nd stencil descent, and VBD fails outright on stiff
  material (Fig. 2: 3 iterations to 1e-3 against > 500 for VBD/PD/XPBD on a beam).

## What the reference implementation does (read, not run)

Faithful but small-scale: dense hand-written Cholesky of H̄, Ū per vertex as above
(`precompute.ts:141-206`), training poses by inverse iteration plus Rayleigh–Ritz,
cubature candidates = every element, fitted to gradients only, greedy selection
with exhaustive small NNLS, residual ≤ 1% else precompute fails. GPU per
iteration: per-element and per-vertex polar rotations, then the local solve with the
exact local block plus the cubature terms (its own block removed from samples that
touch it), a 3×3 eigenvalue shift and per-vertex Armijo line search (its
"globalization", for indefinite Neo-Hookean in f32), and a global inversion check
that reverts the sweep. Contacts are local only (never in H̃/g̃). Pinned vertices are
removed from H̄. Nothing is recomputed at runtime: **no plasticity, fracture or rest
change**. Its tests check gradients, the exact-basis step against full Newton (1e-8),
and GPU against CPU; there is **no convergence-rate test** of JGS2 against plain
Jacobi, so its central claim is not verified there.

## Mapping onto Rubble

| JGS2 | Rubble |
|---|---|
| vertex sub-problem (3 DOF) | particle block solve in `softAccumulate` (already) |
| element (tet, 12×12) | beam (6×6), element (LBEAM 9×9, torsion 12×12) |
| H̄ = K + M/h² | Σ beams k d̂d̂ᵀ (rest) + M/h²; sparse, 3N (3,042 for the high-res car) |
| polar rotation per vertex | shape-matching fit of each node's neighbours (as `calibrate::rigid_fit`) |
| Ū_i storage | per node: K samples × (beam id, weight, two 3×3 blocks) ≈ 100–150 floats |

Cost: H̄ for the high-res car is 3,042² dense or far less sparse; factoring and 3N
solves take seconds on the CPU, once per vehicle *type* (cacheable in the desc).

Rubble-specific questions the papers do not answer:

1. **Plasticity.** A beam's rest Hessian k d̂d̂ᵀ does not depend on its rest length, so
   yielding does not change H̄. But a beam flowing at its threshold has near-zero
   tangent stiffness while Ū assumes k: Ū then overstates coupling, H̃ over-damps and
   the step is too short (slower, not unstable). To measure.
2. **Breaking.** A broken beam still carries perturbation in Ū; across a tear the
   basis says a detached piece moves with the node. Options: rebuild a vehicle's Ū
   on the CPU when its breaks pass a threshold (seconds, async), drop samples whose
   beam broke, or JGS2-GQ if its training-free sampling extends to this.
3. **Rotations of other nodes.** The cubature terms use the rotations of the sample
   beams' endpoints, not only the source node's, so per-node rotations must exist
   before the sweep: one more pass per iteration (dispatch latency), or once per step
   (an approximation the paper does not make).
4. **Rigid core joints and contacts**: not in H̄; treated locally, as the reference
   does with contacts.
5. **Mixed stiffness** (engine block 10× the shell, cell 5× the crumple zones) is
   where JGS2 claims its biggest gains over VBD.

## Proposed order (each a gate; negative results go to DECISIONS)

1. CPU, exact: in the CPU reference (`cpu.rs`), the local solve with the exact
   complementary terms (dense U_i from the current or rest Hessian, no cubature) on
   the high-res car's static sag and a crash. Measure iterations to reach the
   oracle's sag (11 mm) against plain VBD. If this does not converge far faster, stop.
2. CPU, co-rotated rest basis: the same with Ū from H̄ and per-node rotations.
3. CPU, cubature (or GQ, once read): sample count against accuracy; then plasticity
   and breaking (questions 1–2).
4. GPU, behind a setting, against the oracle with `soft_calibrate`.

## JGS2-GQ

From its abstract: replaces Cubature (sample elements and weights trained on poses)
with classic Gaussian quadrature, training-free, as efficient as JGS2, more stable
under large and novel deformation, and over 50% better than JGS2 when deformation
is novel. That is our crash case, so it is likely the variant to build. How the
quadrature is defined over a vertex's complementary domain, and whether Ū still
comes from the rest Hessian, are unknown until the paper is read.

## Gate 1 results (`examples/jgs2_probe.rs`, 2026-11-01)

One backward-Euler step at 240 Hz, elastic beam energy only (no damping, yield or
contacts; the lowest nodes fixed as ground), solved exactly by projected Newton with
dense f64 solves. Error ‖x − x*‖ / ‖x₀ − x*‖ from the inertial guess. `jgs2-rest` is
Jacobi with the co-rotated rest-shape basis and the complementary terms summed
exactly (no cubature); `-ls` adds a global Armijo line search; `jgs2-exact` uses the
current Hessian's basis (a check: one iteration is one Newton step).

| case | method | it 1 | it 3 | it 10 | it 40 |
|---|---|---|---|---|---|
| sedan (367 nodes), gravity | VBD | 7.3e-1 | 3.7e-1 | 8.3e-2 | — (5.5e-3 at 30) |
| | JGS2 rest | 2.2e-4 | converged | | |
| sedan, impact (rear 2/3 at 15 m/s, nose stopped) | VBD | 6.6e-1 | 4.2e-1 | 1.4e-1 | 4.1e-3 |
| | JGS2 rest | 1.3e-1 | 7.1e-3 | 4.1e-7 | converged |
| sedan impact, 10% of front beams broken after the basis | JGS2 rest | 1.3e-1 | 9.0e-3 | 5.4e-5 | 1.4e-9 |
| sedan impact, 30% broken after the basis | JGS2 rest | 2.1e-1 | 5.3e-2 | 6.1e-3 | 9.6e-5 |
| high-res (1,014 nodes), gravity | VBD | 1.2 | 9.2e-1 | 5.5e-1 | 1.9e-1 |
| | JGS2 rest | 1.9e-2 | 7.7e-7 | converged | |
| high-res, impact | VBD | 9.6e-1 | 8.5e-1 | 6.5e-1 | 3.5e-1 |
| | JGS2 rest | 1.27 | 1.92 | 1.38 | 6.0e-1 |
| | JGS2 rest + line search | 7.9e-1 | 8.0e-1 | 4.6e-1 | 2.8e-1 |

Reading:

* Where the step is near the rest Hessian's regime (gravity, driving, a moderate
  impact on the sedan), JGS2 turns VBD's slow linear convergence into near-Newton
  convergence. On the high-res car's gravity step VBD is at 0.19 after 40
  iterations; JGS2 is at 7.7e-7 after 3. That is the high-res car's sag problem.
* The basis degrades gracefully when beams break after it was built: 10% of the
  front's beams cost a little; 30% loses most of the advantage but stays ahead of VBD.
  A basis rebuild after heavy tearing would recover it.
* The high-res impact step (front beams 12 cm long compressed about 50% in one step)
  is violently nonlinear: Newton itself needed 49 line-searched iterations. Plain
  Jacobi JGS2 diverges at first there; with a global line search it is slightly
  ahead of VBD but stalls. The paper says the same (line search needed when Newton's
  quadratic model fails). Real beams cap their force at yield, which this elastic
  probe does not, so the real crash step is less stiff than this one.

Not covered yet: cubature or quadrature (the exact complementary sums are O(N) per
node, too slow for the GPU), yield inside the energy, contacts, the per-iteration
cost on the GPU.
