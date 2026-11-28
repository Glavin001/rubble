# Continuous contact's response, redesigned: rows, one phase an iteration, energy-neutral impacts

Design only (2026-11-26). Nothing here is implemented or measured: every time and count below is
an estimate from the figures in `continuous-contact-design.md` ("GPU cost, measured"), and says
so. It replaces the *response* (`ccd.rs resolve`, `dat.rs speculative`,
`explicit_beams_split_cc_respond.slang`); the listing, the grid and the split seam are not
changed by it.

**Recommendation: design C below.** It is expected to take the response from ~38 ms to about
3-5 ms a frame and to stop contact deleting crash energy. It does not by itself make the two-car
crash real time: the split path (~29 ms) and the listing (~20 ms) remain ("What else real time
needs").

## The problem in numbers

- 2 kHz, 34 steps a 60 Hz frame, budget 16.7 ms. Two sedans head on, M3 Max: ~92 ms a frame;
  the response is ~38 ms (~1.1 ms a step), one workgroup a group.
- The response is ~130 dependent barrier phases a step: 8 speculative rounds and 8 CCD rounds at
  about 5 phases each (flag, count, solve, apply, clear), then up to 16 zone rounds, each with a
  label-propagation loop. An empty phase costs ~4 µs, a full ACCD pass ~60 µs. It is latency,
  not throughput.
- It damps the crash: at head 100 it removes 156-173 kJ of about 1 MJ and the beams' permanent
  set falls 14-18 %. Three mechanisms, all in the present code:
  1. every impact is perfectly inelastic, so each struck node's closing kinetic energy
     (½·m_eff·c²) is deleted, between cars and in every fold of a car against itself;
  2. rigid zones (1-2 a step in hard crashes) delete all relative motion inside a zone;
  3. a node pressed into contact loses the speed the beams gave it this step. This third part
     is not a loss: the node did not move, so the beams' energy is unchanged. The present
     counter (kinetic energy before and after the response) counts it, so the 156-173 kJ
     overstates the true loss by an amount not yet known. The permanent-set figures are the
     evidence that the loss is real.

## Candidates

| | A: same maths, re-plumbed | B: conservative penalty in the node update, CCD as guard | C: rows, accumulated impulses, energy-neutral impacts |
|---|---|---|---|
| phases a step (hard crash) | ~45 | 3-5, plus C's whole guard when it binds | 13-21 typical, ~70 at the caps |
| work a phase | pairs × 4 nodes | pairs | rows × 4 nodes, 24 integer adds a row |
| no crossing | as today | only through the guard | as today (conservative test, zones last) |
| momentum | exact | exact | exact at every iteration |
| energy created | none | none while stable | none at every iteration (proof below) |
| energy removed by contact | as today (the bug) | none in the spring; the guard's as C | impacts none (e = 1); zones only, counted |
| bit for bit | yes | yes | yes |

**A.** Keep the algorithm; drop the apply and clear phases (below) and skip ACCD for certified
pairs. It fixes neither the damping nor the zones. Rejected.

**B.** BeamNG's practice: a spring between a node and a triangle once the paint overlaps,
integrated with the beams. A spring is conservative, so what a node loses comes back to it and
ends in its beams. But an explicit spring's stiffness is bounded by stability (k < 4·m/h², about
3·10⁷ N/m for a 2 kg node), so a 200 km/h closing needs about 1 cm of paint overlap, and a
harder hit goes through: the guarantee would still be C's machinery, engaged in every hard hit.
An IPC barrier (Li 2020) has unbounded stiffness, which an explicit step cannot integrate, and
its activation distance outside the paint is a cushion (forbidden by AGENTS.md). B adds a
tunable stiffness and removes nothing from C. Not recommended now; it can be added in front of
C later if measurements ask for it.

**Graph-coloured Gauss-Seidel** is ruled out as the present doc says: a pair touches four nodes
of valence 20-40, so 20-40 colours, each a phase.

## Design C

Each step, after the node update (`x0`, `x1`, `v_f` the velocities after it, `v_s` the
velocities at the step's start, a new scratch slot):

1. **Detect (one phase, a lane a listed pair).**
   - *Speculative rows:* a near pair whose gap, linearised with its start normal and weights,
     would end under its thickness becomes a row with bias `b = max(g − d, 0)/h ≥ 0`
     (`dat.rs speculative`'s constraint, unchanged).
   - *Certificate:* for the others, the closest-point normal `n` at the start gives a slab:
     every vertex of one primitive is at least `g` from every vertex of the other along `n`
     (the closest-point property of convex sets). Each vertex pair's distance along `n` is
     linear in time, so if all (3 for point-triangle, 4 for edge-edge) are at least `ξ + ε` at
     the step's end, the primitives are at least `ξ` apart all step. This is Planar-DAT's plane
     used as a proof, not a truncation. Certified pairs skip ACCD.
   - *Impact rows:* the rest run ACCD as today; a hit that still closes becomes a row at its
     impact (normal, weights, positions: `ccd.rs Row`).
   - Each node's row count `n_i` is an atomic integer sum. Rows live in their pair's record;
     no ordered list is needed because every sum below is order free.
2. **Solve (one phase an iteration, K at most).** Each row keeps its accumulated impulses
   `λx` (positions) and `λv` (velocities), both ≥ 0. A lane reads its four nodes' current state
   as base + accumulator slot `r mod 3`, updates `λ ← max(0, λ − c/D)` with
   `D = Σ w_j²·im_j·n_j` (Tonge 2012's mass splitting), and adds the row's *whole* contribution
   in fixed point to slot `(r+1) mod 3`; the same lanes zero slot `(r+2) mod 3`. No count,
   apply or clear phase. Rows that change set the group's flag; an iteration in which none
   changes ends the solve (bit-identical fixed point, so the early exit is lossless).
   - Positions: no row closer along its normal than at its start or impact (Harmon 2008).
   - Velocities: `J·v ≥ −b`.
3. **Detect again** over pairs with a moved node (the existing stamp); new hits add rows, a row
   hit again is re-linearised. Steps 2-3 repeat R times at most. Rows are kept, so a resolved
   pair stays resolved.
4. **Rigid zones** (Bridson 2002, `rigid_zone` unchanged) for pairs still colliding after R
   rounds; then the `unresolved` counter, which fails the frame. K, R and the zone rounds are
   explicit engine-wide settings; nothing is dropped or truncated.
5. **Restitution (two phases).** Each row returns `e·σ·λv` along itself, where `σ` is the
   share of its closing that was there at the step's start
   (`σ = clamp(closing(v_s)/closing(v_f), 0, 1)`), with `e = 1` by default. Velocities only;
   positions are already safe for this step. An exact guard scales the whole return by the
   largest `θ ≤ 1` that keeps the group's kinetic energy at or under `v_f`'s (three sums, a
   quadratic). Then the write-back.

### How it meets each requirement

- **No crossing.** Unchanged in kind: every listed pair is either certified (a proof) or
  ACCD-tested (conservative) on the final paths, zones take the rest, and the listing's bound
  and `unresolved` are errors. It does not depend on the solve converging.
- **Momentum.** Every row's impulses are equal and opposite along its normal at its closest
  points (signed weights sum to zero), for any `λ`, so linear and angular momentum hold at every
  iteration, to fixed-point rounding as today (≤ 0.0003 N·s a step).
- **No energy created.** The solve is a majorise-minimise step on the dual problem
  `f(λ) = KE(v_f + M⁻¹Jᵀλ) + bᵀλ`. By Cauchy-Schwarz, `λᵀ(J M⁻¹ Jᵀ)λ ≤ Σ_q D_q·λ_q²` when
  `n_i` is at least the rows at node `i`, so each iteration lowers `f`. With `b, λ ≥ 0`:
  `KE(v_k) ≤ f(λ_k) ≤ f(0) = KE(v_f)` at **every** iteration, converged or not. Restitution is
  bounded by the guard. Zones project onto rigid motion.
- **Deterministic.** Cross-lane sums are integer atomics. Each iteration re-adds a row's total,
  so rounding does not accumulate over iterations. The guard's three sums are float sums striped
  by lane in a fixed order, which the CPU reference mirrors. ACCD and the closest-point helpers
  stay the strict twins they are. Early exits are uniform (`splitUniform`).
- **Nothing lossy.** Iterations buy how seldom the zones act, not the guarantee. The
  certificate changes which pairs run ACCD, never an outcome a correct test would give.

## The energy argument

Between frictionless surfaces contact does no net work: the reaction is normal and the surfaces
do not move apart or together while they touch. All of a crash's dissipation is in the material.
Lumping mass at nodes adds an artefact: an inelastic nodal impact deletes ½·m_eff·c², which
vanishes only as the mesh is refined (the reason for energy-consistent contact schemes: Laursen
and Chawla 1997; mass redistribution, Khenous, Laborde and Renard 2008). With 548 nodes a car
and the whole front third in impacts and folds, it is the measured ~15 %.

The discrete law that removes nothing is Moreau's impact law with `e = 1`:
`v⁺ = v_p + e·(v_p − v⁻)`, with `v_p` the inelastic solution. Since `v_p` is the mass-orthogonal
projection onto a cone, `KE(v⁺) = KE(v_p) + e²·KE(v⁻ − v_p)`, equal to `KE(v⁻)` at `e = 1`. It
is also the stiff limit of BeamNG's node-triangle spring. The struck node keeps its energy and
is thrown back into its own structure (in a symmetric head-on at 100 km/h, 56 m/s against a car
still arriving), which strains its beams past yield: the energy goes into bending the panel
round the impact, where the owner's rule puts it.

`σ` keeps this to true impacts. A node pressed into contact has no closing at the step's start,
so `σ = 0`: its reaction is a plain constraint force, no bounce and no jitter at rest.

Two things remain and are reported apart in every crash test: what the zones remove, and what
the guard withholds. The ledger changes to total energy (kinetic + beam elastic + plastic work
+ damping work), split into reaction, impact, zone and guard parts.

## GPU mapping

- **Workgroups:** one a contact group, as today; groups run side by side in one dispatch.
- **Dispatches:** unchanged (the response stays dispatch C of the continuous module). The new
  kernel is its own file behind its own switch, not more code in the existing module (the
  contact modules are at the browser's compile limit).
- **Memory, a node:** three rotating accumulator slots of 6 i32 (18 words, against 9 now), the
  row count, `v_s`. **A row:** normal, 4 weights, bias, `λx`, `λv`, `σ`, and for impact rows the
  4 impact positions: about 24 words beside the pair's 16, in a row table of explicit capacity
  (overflow is an error). About 0.2 MB a two-car group.
- The buffers' layout changes, so the fixtures are saved again (`fast-check.sh --make`,
  `scenario-matrix/run.sh --make`, `matrix.py --make-portable`).

## Expected cost (estimated, not measured)

From ~4 µs an empty phase and ~60 µs a full ACCD pass. Assumed: a solve phase with its loads
~6 µs; the certificate and the moved-node rule leave a first detection ~25 µs and later ones
~10 µs.

| step | phases | µs a step | ms a frame |
|---|---|---|---|
| no pair near | 1 | ~10 | ~0.3 |
| resting or pressed contact (1 round, 2-4 iterations) | ~8 | ~60 | ~2 |
| hard crash (2 rounds of 8) | ~20 | ~135 | ~4.6 |
| at the caps (4 rounds of 16, zones) | ~75+ | ~500 | ~17 |

So ~38 ms becomes about 3-5 ms in the head-on, if K = 8 and R = 2 are what crashes need. That
is the largest unknown and is measured on the CPU before any Slang (plan step 2).

**What else real time needs.** With this response the head-on is still about 29 (split path)
+ 1.5 (prep) + 20 (grid and listing) + 3 (seam) + 4 ≈ 58 ms. The Verlet list already proposed
takes ~15 more. The rest is the split path itself (the matrix's head-on row is 15.4 ms without
it): since the response is one workgroup a group with barriers, it could run inside the group
kernel's step loop once the listing does not need a dispatch each step. That is a separate
design. **Scaling:** many groups cost the same wall time each. One large group (a pile) has more
rows than lanes and its phases lengthen in proportion; splitting a group into row islands (rows
sharing no node are independent, exactly) is the follow-up.

## Risks

- **Iterations.** Mass-split Jacobi converges slower than the CPU's Gauss-Seidel sweeps by up
  to the node valence. If K = 8 leaves zones as frequent as today, the damping from zones stays.
  Mitigations, in order: warm-start `λ` from the last step (needs an energy check on the start
  point), monotone accelerated projection (Mazhar 2015).
- **Chatter at e = 1.** Nodes will bounce against their beams; the jitter measure may rise.
  `e` is an explicit engine-wide setting; the default is decided on the measured table, by the
  owner.
- **The guard is one scalar a group.** When it binds it couples distant impacts. Count how
  often; make it per island if it is not rare.
- **Restitution on speculative rows** acts at the paint thickness; with thick paint shells
  (3-6 cm at the sedan's front) its effect on the depth bar is unknown.
- **Position shifts and beam energy.** Positions are only moved back along this step's own
  displacement, which should create no strain; the total-energy ledger checks it.
- **The certificate's rate** is unmeasured; if most close pairs fail it, detection stays ~60 µs.
- **Phase cost** may be 8-10 µs with loads, not 6: then ~6-7 ms a frame.
- **No ground truth** for the right amount of crumpling: the run without continuous contact has
  52 crossings. The 1-D tests below have exact answers; the crash is judged by the ledger.

## Plan (CPU reference first, then Slang, rung by rung)

1. **Evidence first (CPU, no physics change).** The total-energy ledger in `vehicle_film`'s
   SUMMARY. Tests with exact answers: two elastic node chains head on (all energy returned after
   2L/c), a node pressed on a triangle by a spring (no drift, no jitter), a node between two
   closing surfaces (the guard). Record today's response on them.
2. **CPU reference** in `ccd.rs` as a second response beside `resolve` (`DAT_RESPONSE=rows`;
   the old path stays): certificate, rows, the three-slot fixed-point iteration exactly as the
   kernel will run it, restitution and guard, zones unchanged. Then `tests/continuous_contact.rs`
   (8 runs each, the adopted depth bar, 0 crossings, 0 unresolved, momentum) and a table for
   head 60/100/130, offset 100, side 60, ball 100 at e = 0 and 1: rows a step, iterations to a
   fixed point, the certified share, zones, the ledger's four parts, permanent set, jitter.
   **Gate:** K and R chosen from this table; e's default is the owner's decision. No Slang
   before it.
3. **Slang**, a new kernel file behind an `--ab-switch` value, written from the CPU reference
   operation for operation; every loop budgeted (`EB_END`), a CPU run then a tiny-budget GPU
   run first. Read `fast-kernels-and-builds` and `gpu-safe-runs` before.
   - Rung 1: `fast-check.sh SWITCH` on new continuous fixtures (first hit, pile, rest),
     GPU == CPU every state word, bit-exact on.
   - Rung 1.5: `scenario-matrix/run.sh SWITCH`, then `--cuda`.
   - Rung 2: `FRAMES=20`, `--ab-profile`, `explicit_loop_cap`, Chrome's module check, and
     `compile_cost.py` for the new module.
   - Rung 3: `perf-verify.sh`, `engine-metal` parity, `engine-cuda tests/continuous.rs`.
   - Rung 4: the 40 s runs; `continuous_cost` for the cost table, each block by running it
     twice in a step as before.
4. **Report** in `continuous-contact-design.md`: the measured phases and ms against this
   estimate, what contact removed in every crash test, and the zones' count.
