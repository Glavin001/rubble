# AVBD codebases: augmented vertex block descent

Read-only review (2026-11-02) of the reference C++ demos (`avbd-demo3d` and
`avbd-demo2d`), three-avbd (WebGPU) and avbd-sandbox. The paper is Giles, Diaz and
Yuksel, *Augmented Vertex Block Descent*, ACM TOG 44(4), 2025.

None of these codebases simulates node–beam point masses. Their "soft bodies",
cloth and ropes are rigid six-degree-of-freedom bodies joined by joints or
springs.

## 1. Solver

**Each step:**
1. Predict the inertial target: position plus velocity × dt plus gravity × dt².
2. Warm-start adaptively.
3. Run iterations. Each solves a 6×6 block per body (LDLᵀ, sequential
   Gauss–Seidel), followed by a dual (multiplier) pass.
4. Take velocity from the change in position (BDF1).

**Defaults:**

| | Reference C++ | three-avbd |
|---|---|---|
| dt | 1/60 | |
| Iterations | 10 | |
| β (linear) | 1e4 | 1e5 |
| β (angular) | 100 | |
| α | 0.99 | 0.95 |
| γ | 0.999 | 0.99 |
| Penalty range | [1, 1e10] | |
| Post-stabilisation | | off |

- **2D demo:** β is 1e5, γ is 0.99, and post-stabilisation is on.
- **β units:** three-avbd notes that the paper's β = 10 is wrong for these units.
- **three-avbd fixes:** bodies start at rest, and the penalty has a floor
  scaled by mass.
- **Multipliers:** λ accumulates only for hard rows, those of infinite
  stiffness. For finite rows the penalty ramps by β·|C| up to the material
  stiffness.
- **avbd-sandbox** runs a 64-bit CPU path plus WebGPU. It adds geometric
  stiffness for springs and caches contacts as Taylor expansions.
  Post-stabilisation is off, because it collapses above about 33k bodies.

## 2. Stiffness: where jello comes from

- **Finite springs behave as plain VBD.** Their effective stiffness depends on
  the iteration count, with implicit damping on top.
- **Ramping a material stiffness causes jello.** The sandbox measured a 1e6 N/m
  spring behaving like 7.5e4 N/m with the ramp on. It is off by default.
- **True stiffness comes only from hard rows with λ.**
- **Stiffness spreads slowly.** It travels about one link per colour sweep, so
  long chains sag at low iteration counts: a 20-link cantilever sagged by 0.10.
  Two iterations collapse, four hold, ten settle.
- **Colouring:** hard rows need zero colour clashes. A Jacobi fallback
  exploded.

## 3. Fracture and plasticity

- **Breaking:**
  - The demo breaks a joint when |λ| exceeds a threshold, for hard rows only.
  - The sandbox tears springs by strain inside the solve, which is
    forward-only.
- **Plasticity (three-avbd findings):**
  - Angular yield in the dual update moves the rest rotation, zeroes the
    stored constraint offset, clamps λ, and stops the penalty ramp while the
    joint is yielding.
  - A joint breaks in bending past a plastic limit.
  - There was no measurable cost.
  - It maps directly onto a per-beam plastic rest length.

## 4. Contact

- Three rows per contact (normal and two tangents), linearised by Taylor
  expansion.
- Friction is clamped to the Coulomb cone.
- Sliding friction is applied at about half its true value. An opt-in
  rescaling fixes it.
- The margin is 1 cm.

## 5. GPU structure (three-avbd)

- **Submission:** one command buffer with indirect dispatches.
- **Colouring:** incremental Jones–Plassmann on the GPU.
- **Dispatch cost:** an empty dispatch costs about 12 µs, so below about 100k
  bodies the dispatch count dominates.
- **Lanes:** one to eight lanes per body.
- **Fused tail colours:** running the small tail colours in one workgroup gave
  about 3.4x. Running everything in a single workgroup was 2.5–3x slower.
- **Throughput:**
  - 34k bodies and 71k joints at 10 iterations: 6.4 ms;
  - a 102k-body 2D lattice: about 5 ms.

## 6. Lessons for a Rubble fast tier

- Make stiff beams hard rows with λ. Never ramp a finite material stiffness.
- Put yield and breaking in the dual update, using λ as the beam force.
- Use α 0.95, β about 1e5, γ 0.99, start bodies at rest, floor the penalty by
  mass, and solve each node block with LDLᵀ.
- GPU structure:
  - one command buffer with indirect dispatches;
  - one dual pass per iteration;
  - fused tail colours;
  - lanes for high-valence nodes;
  - 3×3 node blocks.
- Colour the static beam graph once. Masking broken beams keeps the colouring
  valid.
- **Avoid:**
  - a Jacobi fallback with hard rows;
  - post-stabilisation at scale;
  - the paper's β = 10;
  - expecting few iterations to carry stiffness along long chains (use
    substeps or ordering).
