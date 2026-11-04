# JGS2, an elasticity-solver benchmark, webphysics and beat-box

Read-only review (2026-11-02) of the JGS2 paper and its WebGPU reproduction, the
mesh-elasticity solver benchmark, webphysics and beat-box.

## JGS2

Lan et al., *JGS2*, ACM TOG 2025, arXiv:2506.06494.

**The idea:**
- Per-vertex local solves "overshoot". JGS2 fixes this with a precomputed local
  subspace of perturbations, taken from the rest-shape Hessian and co-rotated
  each step by polar decomposition.
- Cubature estimates the coupling terms from 4–6 elements per vertex.

**The authors' claims (unverified):**
- 40–173x faster than VBD, projective dynamics and GPU-IPC.
- 3 iterations, against 500+ for XPBD, PD or VBD, on a stiff ARAP beam.
- Jacobi is about as good as coloured Gauss–Seidel.

**Costs:**
- Precompute takes minutes to an hour per mesh.
- Plastic yield and breaking make the precomputed subspace stale.

**The reproduction:**
- one thread per vertex;
- pre-recorded iterations with a GPU stop flag;
- colouring and cubature on the CPU.

## Mesh-elasticity solver benchmark

A NumPy harness comparing iteration counts at equal energy and stopping test.

**Mass-spring implicit step, to a 1e-3 tolerance:**

| Solver | Iterations |
|---|---|
| Projected Newton | 2 |
| Projective dynamics | 8 |
| Nonlinear Gauss–Seidel (VBD-style) | 14 |
| XPBD | stalls at 0.14 |
| PBD | diverges |

**XPBD error against the true implicit step, after 20 sweeps:**

| Stiffness | dt | Error |
|---|---|---|
| 1e3 | 1/60 | 0.2% |
| 1e5 | 1/60 | 9.1% |
| 1e5 | 1/8 | 65% |

**Stiff Neo-Hookean step:**

| Solver | Iterations |
|---|---|
| Newton | 3 |
| L-BFGS (Laplacian start) | 6 |
| Anderson | 6 |
| Chebyshev | 7 |
| Projective dynamics | 10 |

**Other findings:**
- **Ordering:** Gauss–Seidel beats relaxed Jacobi for VBD: residual 0.52x
  against 0.93x after 24 sweeps.
- **Inertia keeps the Hessian positive definite:** unfiltered Newton took 3.0
  iterations per step, against 4.8 with clamping.

## webphysics and beat-box

- **webphysics:** soft bodies are lattices of rigid bodies with fixed joints
  under AVBD, and cloth is springs between bodies. There is no plasticity or
  breaking.
- **beat-box:** rigid bodies only (AVBD, PGS and TGS in Slang, with an LBVH).

## Lessons for Rubble

- **No XPBD or PBD for sheet metal:** its stiff error is 9–65%.
- **Converge to the true implicit step.** Fix overshoot either:
  - with cheap global awareness (JGS2-style coupling);
  - or with Anderson or Chebyshev acceleration, which need no precompute and
    survive yield and breaking.
- **Small steps:** with a small dt, inertia keeps unfiltered local Newton steps
  safe.
- **Stop on the GPU:** decide early stopping there.
- **Jacobi:** acceptable only with coupling.
- **Expect collision to become the main cost.**
- **Validate:** plot residual against iterations on a real vehicle truss, using
  a Newton-solve reference.
