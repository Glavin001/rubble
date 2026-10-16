# AVBD as the stress solver: fidelity and cost against the explicit truth

Written 2026-10-16. Question (from the stress-fracture plan): can the struck building's
internal stress be solved by AVBD itself (implicit, augmented Lagrangian, block
Gauss–Seidel) faster than by the oracle's explicit substeps, at the same physical
fidelity? If so, the in-engine "impact windows" (~1,000–2,000 explicit substeps per
1/120 s step) could become a few AVBD iterations inside rubble's step.

## Verdict

**No: at the explicit truth's fidelity AVBD is not cheaper than explicit substeps for
impact windows.** Measured on the gate cases (CPU f64, M3 Max):

- **AVBD's own integrator (backward Euler) is not faithful.** Its numerical damping flips
  the b5 failure regime at the push-over/punch-through boundary (15–30 m/s turn into clean
  punch-throughs) from a 25 µs step, only ~3× the explicit true step (8.5 µs), and cuts b4's
  dynamic overshoot by 40 % at 18 steps per period (1.56 vs 1.94; within 1 % needs ~900).
- **The same block descent on Newmark's step is faithful up to ~25 µs** (all b5 speeds,
  the free wall and the house inside the truth's own spread), *provided* contact stays the
  oracle's compliant penalty law inside the solve (AVBD's hard contact is rigid contact:
  the ram stops at 13.8 m/s instead of 35.9, or the solve diverges) and the damage law is
  evaluated from the step-start history on every visit (frozen damage turns the 40 m/s
  punch-through into a push-over). At 100 µs it diverges on three b5 cases.
- **It needs about one Gauss–Seidel sweep or more per explicit substep it replaces**:
  a sweep moves load one bond, an explicit substep moves a wave one bond. At 25 µs Newmark
  needs 9–24 sweeps per step on b5 (10 fixed sweeps also pass), i.e. 3,000–8,000 sweeps per
  1/120 s against 975 explicit substeps, and a sweep costs 2.8–3.9 explicit substeps of
  work (two law evaluations, the tangent and a 6×6 solve per bond end) and one
  synchronised pass per graph colour (2–5 on the walls, 6 on the house) on the GPU.
- **Net cost against explicit at its true step:** b5 and the free wall **8–32× more work**
  (single-thread wall time on the 40 m/s impact: 5.1×); house-1story **~1.5× more work, 0.8×
  the wall time** (its explicit step, 1.3 µs, is set by the contact law's stability bound,
  which an implicit contact escapes) and ~3× on a GPU latency model. Nowhere a clear win;
  break-even at best, and only where the explicit step is set by something other than the
  bond physics.

**For the in-engine design:** keep the struck house's impact windows explicit (the
oracle's integrator, at its certified true step) inside rubble's AVBD step; do not
replace them with AVBD iterations. The levers that do cut the window cost remain the
roadmap's (sliver elimination, intact kernel, per-island stepping, fewer awake
buildings). One lead from this study: on the house the explicit window's step is set by
the penalty contact's stability bound (1.3 µs) rather than the bonds (certified 3.5 µs,
estimated 13 µs, A1); treating only the window's *contacts* implicitly (a per-body block
solve, no network iteration) could lift that limit without touching the bond integrator.

## Method

**Prototype:** `crates/stress-ref/src/avbd.rs` (`AvbdWorld`), driven by
`examples/avbd_compare.rs`. Nothing else uses it; default behaviour and every existing
test are unchanged (the only edit outside it makes five `world.rs` contact helpers
`pub(crate)`).

**One model, three integrators.** The world is the oracle's physics at chunk level, as an
AVBD engine would hold a struck building:
- every active chunk is a 6-DOF rigid body (the oracle's mass and inertia), every
  impactor another;
- every bond is the oracle's joint (`joint.rs`: tension/shear/crush/buckling,
  softening, damage, cracked-joint contact patch) with its **true** stiffness
  (`BondStiffness`, `kn = EA/L` etc., no solver penalty) and its dashpot, through the
  co-rotational kinematics of `bond.rs` (the oracle's small-strain map to first order);
- contact is the **established** sample-point penalty law of `world.rs` (same sample
  points, stiffness shares, dashpot, regularised Coulomb friction, pre-existing-overlap
  memory). The dashpot's stability cap is evaluated at the oracle's own substep for every
  integrator, so the law is one law;
- clusters are connected components of connected bonds; a bond whose chunks land in
  different components stops acting and contact between them starts;
- the world starts from an oracle `World` after its prestress (same equilibrium).

Integrators (`Integrator`):
- `Explicit`: the oracle's symplectic substep at the oracle's substep — the truth,
  replicated, to validate the chunk-level world against the oracle;
- `BackwardEuler`: AVBD as published: per step `h`, block Gauss–Seidel over a graph
  colouring on `1/(2h²)|x − x̃|²_M + E(x)`, one 6×6 Newton step per body per visit
  (Cholesky of the block Hessian: mass, each bond's `Jᵀ K_t J` with the law's frozen-history
  tangent plus dashpot, each contact's point stiffness), warm start from the previous
  acceleration;
- `Newmark`: the same block descent on Newmark's average-acceleration step (`β = ¼`, the
  oracle's `implicit.rs`), also a minimisation (`2/h²|x − x̃|²_M + E(x)`), no numerical
  damping.

**Damage inside the iteration** (`DamageMode::Implicit`, default): on every visit a bond's
law is evaluated at the trial deformation *from its history at the step's start* (so the
force carries the damage the end state implies), with the frozen-history tangent (the
secant in the damage: positive semi-definite) as the block Hessian. Nothing is written
back during the iteration; history is committed once, at the accepted state.
`DamageMode::Frozen` holds damage through the iteration (the oracle's implicit mode).

**Contact inside the solve.** Contacts are found once per step at the pose the step-start
velocities predict for its end (where backward Euler and Newmark evaluate forces), plus a
margin of ¼ of the step's travel, and linearised (fixed lever arms and normals, AVBD's
practice). `ContactMode::Penalty` keeps the oracle's law (stiffness, dashpot, friction)
inside the solve; `ContactMode::Hard` is AVBD's augmented-Lagrangian contact
(`α = 0.95`, ramped penalty, `γ = 0.99`).

**Convergence measure.** The sweep residual (each body's block residual before its
update, force and torque/radius, relative to the forces acting) after each sweep. Runs
are either converged (`--tol 1e-6`, at most 2000 sweeps, a step stops as *stalled* once
its best residual has not fallen 10 % in 40 sweeps) or at a fixed sweep count (AVBD's
practice).

### Two pitfalls found on the way (fixed before any number below)

1. **A speculative contact must not act before it overlaps.** With a margin, a point found
   ahead of the ram still produced a dashpot force (`−c·vₙ` while closing at 36 m/s):
   b5 at 40 m/s then broke 710 bonds instead of 71.
2. **Contact normals must come from the overlap, not the approach.** Found at the step's
   *start* within a margin, a wall chunk beside the ram's leading edge takes the ram's
   front face as its nearest face (it is approaching it); the law (and the explicit
   truth) assigns the nearest face of the actual overlap — the side or bottom face. The
   ram sliding through its hole then plowed the hole's edges instead of grazing them:
   655 broken bonds and 186 chunks out against 71 and 32, *at the explicit step,
   converged*. Detecting at the predicted end pose fixes it (71 broken, 32 out). This is
   a property of AVBD's per-step contact linearisation, not of this prototype: the
   engine's AVBD would need the same care in any stress window.

## Validation of the chunk-level world (explicit replica vs the oracle)

`--solver oracle` runs `World::run` on the same scene, established methods, same machine.

| case | oracle: broken / fragments / out (fast) / ram after / first crack | replica |
|---|---|---|
| b4 sudden, peak reaction | 48.4942148 N (amplification 1.9433) | 48.4942148 N |
| b2 cantilever, f₁ / amplitude ratio over 5 periods | 2.7145 Hz / 1.0078 | 2.7145 Hz / 1.0075 |
| b5 10 m/s | 249 / 24 / 800 (40) / 8.36 / 1.32e-4 | 350 / 50 / 800 (40) / 7.95 / 1.32e-4 |
| b5 15 m/s | 437 / 87 / 800 (54) / 12.42 / 9.21e-5 | 380 / 82 / 800 (52) / 12.46 / 9.21e-5 |
| b5 20 m/s | 541 / 108 / 630 (50) / 17.00 / 7.22e-5 | 483 / 101 / 566 (50) / 17.09 / 7.22e-5 |
| b5 30 m/s | 123 / 5 / 480 (32) / 26.98 / 5.23e-5 | 310 / 48 / 800 (34) / 26.54 / 5.23e-5 |
| b5 40 m/s | 64 / 3 / 32 (32) / 36.00 / 4.23e-5 | 71 / 3 / 32 (32) / 35.85 / 4.23e-5 |
| ram into free wall (0.5 s) | 24 / 6 / 8 (8) / 35.03 / 1.27e-3 | 27 / 6 / 8 (8) / 34.75 / 1.27e-3 |

The replica reproduces the oracle's failure regime, ram exit speed (≤5 %) and first-crack
time (identical) on every case; bond counts differ inside the scenes' own chaotic spread
(v30 sits on the push-over/punch-through boundary: the oracle ends with 480 chunks out,
the replica with 800, both with 32–34 fast).

## b4: dynamic amplification (sudden support loss)

One free chunk between two supports; period T = 0.89 ms; the explicit truth is 1.9433
(damped SDOF 1.9391). Every AVBD step here converges in 1–3 sweeps (a single body), so this
measures the integrators alone.

| step | steps per period | backward Euler | Newmark |
|---|---|---|---|
| 200 µs | 4.5 | 1.179 | 1.778 |
| 50 µs | 18 | 1.555 | 1.919 |
| 20 µs | 45 | 1.756 | 1.936 |
| 5 µs | 180 | 1.890 | 1.940 |
| 1 µs | 890 | 1.930 | 1.940 |

Backward Euler — AVBD's integrator — damps the peak by `≈ exp(−π²/N)`: within 1 % of
the truth needs ~900 steps per period (1 µs here, below the oracle's 24 µs explicit
substep). Newmark matches the oracle's implicit findings (1.73 at T/4.5 there with its
own tangent, 1.92 at T/18). A free-chunk SDOF analysis gives the same table.

## b2: cantilever (20 chunks, bending) — iterations, frequency, statics

**Free vibration** after release (2 s), converged to 1e-9:

| integrator, step | sweeps / step | f₁ | amplitude after 5 periods |
|---|---|---|---|
| explicit 19.5 µs (truth) | — | 2.7145 Hz | 1.0075 |
| Newmark 10 / 25 / 100 µs | 3.0 / 4.0 / 7.0 | 2.7145 | 1.0075 / 1.0075 / 1.0073 |
| Newmark 1 ms | 200 | 2.7151 | 1.0078 |
| backward Euler 10 / 25 / 100 µs | 3.1 / 5.2 / 13.9 | 2.7145 / 2.7144 / 2.7139 | 1.0036 / 0.9985 / 0.9823 |
| backward Euler 1 ms | 832 | 2.7138 | 0.813 |

**Sweeps needed to converge** (Newmark, tol 1e-6, first 0.3 s):

| step | explicit substeps it replaces | sweeps / step | sweeps per replaced substep | with Chebyshev (best ρ) |
|---|---|---|---|---|
| 0.1 ms | 5 | 5.0 | 0.98 | |
| 0.3 ms | 15 | 16.3 | 1.06 | |
| 1 ms | 51 | 124 | 2.4 | 41 (ρ 0.9): 0.8 |
| 3 ms | 154 | 761 | 4.9 | 139 (ρ 0.99): 0.9 |
| 8.3 ms (1/120 s) | 427 | 2,682 | 6.3 | |

**Fixed sweep counts at 1 ms** (51 explicit substeps per step): 10 sweeps — Newmark
diverges by 0.07 s, backward Euler with the previous-acceleration warm start diverges,
with the inertial warm start runs but rings at 1.75 Hz instead of 2.71 (the unconverged
network is too soft); 50 sweeps — correct frequency (Newmark 1.0077, backward Euler its
own 0.83 damping). Outcomes depend on the sweep count until the step converges.

**Statics** (the cantilever from rest under its tip load, backward Euler with a long
step as a static solver): 324 sweeps at 1 ms, 27,137 at 10 ms, residual still 0.15 after
100,000 sweeps at 0.1 s, diverged at 10 s. Load travels one bond per sweep and a
bending chain's conditioning grows like N⁴; AVBD is no static solver for bonded networks
(the oracle's statics use block-Jacobi PCG or a direct solve).

The pattern behind both tables: a Gauss–Seidel sweep moves information one bond, and
the explicit step is the time a wave takes to cross one bond. A converged implicit step
of `n` explicit substeps therefore needs at least ~`n` sweeps once the step resolves the
response (b2 at 0.1–0.3 ms); larger steps need more than one sweep per substep, and
Chebyshev acceleration only brings that back to ~0.8–0.9.

## Impacts: b5 wall 10–40 m/s, ram into a free wall, house-1story rammed

"Out (fast)": chunks detached from the anchored wall (or the main piece), and of those
the ones faster than half the ram's initial speed (the punched plug). Push-over is ~800
out, punch-through 32 out all fast; 20–30 m/s are mixed and sit on the regime boundary.
The truth's own spread is the oracle, the replica at the oracle's substep (2.49 µs) and
the replica at the certified **true** step (8.55 µs, the GPU solver's; `--true-step`):

| case | truth spread: broken / out (fast) / ram after (m/s) |
|---|---|
| b5 10 | 249–350 / 800 (40–47) / 7.95–8.36 |
| b5 15 | 380–437 / 800 (50–54) / 12.42–12.48 |
| b5 20 | 483–541 / 566–664 (48–50) / 17.00–17.09 |
| b5 30 | 123–567 / 480–800 (32–38) / 26.26–26.98 |
| b5 40 | 64–209 / 32–386 (32) / 35.85–36.00 |
| free wall | 24–27 / 8 (8), 6 fragments / 34.69–35.03 |
| house (to 0.04 s) | 155–159 / 43–44 (0) / 2.64–2.86 |

**AVBD converged** (tol 1e-6; ✓ inside the spread, ~ at its edge, ✗ outside):

| integrator, step | b5 10 | b5 15 | b5 20 | b5 30 | b5 40 | free wall | house |
|---|---|---|---|---|---|---|---|
| Newmark 10 µs | ✓ 274 / 800 (39) / 8.37 | ~ 381 / 623 (52) / 12.47 | ~ 358 / 512 (40) / 17.51 | ✓ 121 / 480 (32) / 26.87 | ✓ 79 / 32 (32) / 35.83 | ✗ 192 / 162 (8) / 33.89 | ✓ 152–158 / 43–44 / 2.68–2.72 |
| Newmark 25 µs | ✓ 239 / 800 (48) / 8.31 | ✓ 395 / 800 (50) / 12.45 | ✓ 603 / 662 (48) / 17.00 | ~ 620 / 555 (48) / 25.90 | ✓ 92 / 32 (32) / 35.45 | ✓ 25 / 8 (8) / 34.15 | ✓ 162–178 / 48 / 2.69–2.81 |
| Newmark 100 µs | ✗ diverged | ✗ diverged | ✗ diverged | ✗ 363 / 104 (21) / 25.28 | ✗ 96 / 32 (32) / 33.29 | ✗ 423 / 171 (8) / 31.99 | ✓ 155–165 / 42–48 / 2.67–2.99 |
| backward Euler 10 µs | ✓ 334 / 800 (49) / 7.96 | ✗ 148 / 480 (32) / 13.23 | ✓ 535 / 602 (48) / 17.09 | ✗ 53 / 32 (32) / 27.03 | ✓ 50 / 32 (32) / 36.01 | ✓ 20 / 8 (8) / 34.97 | |
| backward Euler 25 µs | ✓ 334 / 800 (39) / 8.23 | ✗ 129 / 480 (33) / 13.30 | ✗ 64 / 32 (32) / 18.07 | ✗ 55 / 32 (32) / 27.17 | ✓ 64 / 32 (32) / 36.15 | ✓ 20 / 8 (8) / 35.18 | ✓ 141 / 40 / 2.74 |
| backward Euler 100 µs | ✗ 78 / 32 (32) / 9.11 | ✗ 32 / 32 (32) / 13.61 | ✗ 32 / 32 (32) / 18.41 | ✗ 48 / 32 (32) / 27.42 | ✓ 48 / 32 (32) / 36.51 | ✗ 88 / 164 (8) / 35.15 | |

Backward Euler damps the wall's bending response (the push-over), so the punch-through
stays and the far-field failure disappears: every case on or below the boundary becomes
a clean 32-chunk punch-through from 25 µs (10 µs at 15 and 30 m/s). Newmark keeps the
regime to 25 µs; at 100 µs a few stalled steps (below) inject energy and the run diverges.

**Fixed sweep counts** (AVBD's practice), out (fast) / ram:

| config | b5 10 | b5 20 | b5 40 | free wall | house |
|---|---|---|---|---|---|
| Newmark 10 µs × 5 / 10 / 20 | ✓ 800 (36–40) / 8.37–8.40 | ~ 505–518 (40) / 17.48–17.50 | ✓ 32 (32) / 35.83 | | ✓ 44 / 2.68 (× 10, 30) |
| Newmark 25 µs × 5 | ✓ 800 (45) / 8.31 | ✓ 632 (48) / 17.06 | ✗ 796 (46), 743 broken | | |
| Newmark 25 µs × 10, × 20 | ✓ 800 (44–48) / 8.30–8.31 | ✓ 630–660 (48–50) / 17.02–17.03 | ✓ 32 (32) / 35.45 | ✗ × 10: 180 (10), 133 broken | ✓ 38–45 / 2.65–2.68 |
| backward Euler 10 µs × 5 / 10 / 20 | ✓ 800 (41–51) | ✓ 591–609 (49) / 17.08–17.10 | ✓ 32 (32) / 36.0 | | |
| backward Euler 25 µs × 5 / 10 / 20 | ✗ 436 (44–48) at × 5, 10; ✓ at × 20 | ✗ 32 (32) / 17.95–18.07 | ✓ 32 (32) / 36.1 | | |

Within the converged regime the outcome barely depends on the sweep count from ~10
sweeps; below that, or past the step where the converged solve fails, it does.

**Variants** (Newmark 10 µs, converged):

| variant | b5 20 | b5 40 |
|---|---|---|
| damage frozen in the iteration (the oracle's implicit mode) | ✓ 451 / 658 (48) / 16.99 | ✗ 512 / 800 (41) / 35.06: push-over added |
| hard contact (AVBD AL, α 0.95, γ 0.99, β 10) | ✗ diverged, all 1,960 bonds broken | ✗ diverged |
| hard contact, penalty capped at the law's stiffness | | ✗ ram 13.79 m/s at 3 ms (truth 35.84) |

Hard contact drives the contact gap to zero: a rigid ram against the wall, the over-impulse
the README documents for a rigid engine (PhysX: 11 m/s for the free wall). The oracle's
contact compliance is physics, not a solver parameter; inside the solve it must stay the
penalty law. (The AL parameters are this prototype's; tuning them cannot restore a
compliance the formulation removes.)

### Iterations: convergence, stalls

At a step where nothing breaks or switches, the sweep residual falls 1–2 decades per
sweep at first and converges in 2–5 sweeps (b5 40 m/s, 10 µs). At a step where a bond
softens, a cracked patch or a contact switches on/off, Gauss–Seidel settles into a limit
cycle around 1e-3–1e-2 (worst seen 5e-2): the law is non-smooth and the block Hessian is
its secant. Those steps were stopped by the stall test (no 10 % progress in 40 sweeps); they
are 1–14 % of steps on b5 at 10 µs (37–456 of 3,334), up to half on the house. A capped
solver (2000 sweeps) spends almost all its time in them. Chebyshev acceleration (VBD's)
helped the smooth cantilever 3–5× but not the impact (10 µs: 93 vs 78 sweeps per step in
the impact window) and destabilised it at 25 µs (ρ 0.95: 717 broken bonds).

## Cost per simulated 1/120 s

Explicit substeps per 1/120 s: b5 and the free wall 975 at the true step (3,349 at the
oracle's substep); house-1story 6,432 at the true step, which here is the established
contact law's bound (the bonds alone: certified 2,365, estimated 632, A1).

Per-iteration work, measured on the same state, one thread (`--bench`): one explicit
substep's bond pass vs one Gauss–Seidel sweep (all bodies, bonds' laws and tangents, 6×6
solves; contacts excluded):

| scene | bodies / bonds | explicit pass | GS sweep | sweep / pass | colours |
|---|---|---|---|---|---|
| b2 cantilever | 20 / 20 | 5.2 µs | 17.6 µs | 3.4 | 2 |
| b5 wall | 801 / 2,018 | 623 µs | 2,398 µs | 3.8 | 2 at rest, 3–5 with contacts |
| free wall | 221 / 508 | 113 µs | 373 µs | 3.3 | 2–4 |
| house-1story | 300 / 877 | 266 µs | 751 µs | 2.8 | 6 |

Cost of the cheapest configuration that keeps fidelity, against explicit at the true
step (work = sweeps × sweep/pass ratio; GPU = sweeps × colours, against one synchronised
pass per explicit substep — the latency-bound regime A5/A6 measured):

| case, config | sweeps / 1/120 s | explicit substeps | CPU work | CPU wall (1 thread, measured) | GPU passes |
|---|---|---|---|---|---|
| b5 40, Newmark 25 µs × 10 | 3,333 | 975 | 13× | 5.1× (5 ms impact: 3.90 vs 0.76 s) | 13× |
| b5 40, Newmark 25 µs converged | 3,030 | 975 | 12× | 5.5× | 11× |
| b5 20, Newmark 25 µs converged | 8,130 | 975 | 32× | | 30× |
| b5 10 / 15 / 30, Newmark 25 µs converged | 3,870 / 6,070 / 7,200 | 975 | 15× / 24× / 28× | | |
| free wall, Newmark 25 µs converged | 2,300 | 975 | 7.8× | | 7–9× |
| house, Newmark 25 µs × 10 | 3,333 | 6,432 | 1.5× | 0.76× (30 ms: 8.95 vs 11.75 s) | 3.1× |
| house, Newmark 10 µs converged | 5,165 | 6,432 | 2.3× | 0.90× | 4.8× |
| house vs bonds' certified step (contact limit lifted) | 3,333 | 2,365 | 4.0× | | 8.5× |

(The measured wall times include this prototype's contact detection, done every explicit
substep but once per AVBD step, which flatters AVBD; the work column is the
like-for-like kernel cost.)

## Limits

- CPU f64 prototype; no GPU path was built — by the task's own condition ("if promising")
  the CPU result rules it out. The GPU column is a model (colour passes against one
  synchronised pass per explicit substep), not a measurement.
- Chunk-level world: bonds are co-rotational (the oracle's established runs are
  small-strain; identical to first order), contacts are the established sample-point law,
  no crush laws, rebar, blasts, pressure loads or support-removal events (only what the
  gate cases use). The replica matches the oracle as tabled above.
- Single runs per configuration; b5 at 20–30 m/s is chaotic (the truth's own spread is
  wide), so the regime and the ram exit speed are the robust comparisons, bond counts are
  not.
- House: rammed to 0.04 s only (the impact and first failures), not the full second.
- Hard-contact parameters (α, β, γ, cap) are this prototype's choice; the rigid-contact
  outcome does not depend on them, the divergence may.
- Not tried: a direct or multigrid inner solve (the oracle's Newton–PCG implicit mode
  already covers the "implicit at a larger step" question; the IMPROVEMENTS.md findings
  hold here too), AVBD's stiffness ramp for bonds (it is lossy until the ramp completes).

## Reproduce

```sh
cargo run --release -p stress-ref --example avbd_compare -- --case b5_wall_impact_v40 \
    --end 0.0333333333333 --solver nm --step 2.5e-5 --iters 2000 --tol 1e-6
#   --solver oracle | explicit [--true-step] | be | nm;  --iters N (fixed sweeps);
#   --damage frozen; --contact hard [--hard-cap K --beta B]; --chebyshev RHO;
#   --bench (kernel timings); --trace (sweep residuals); --dump probes.json
```
Cases: `b2_cantilever`, `b4_support_loss_sudden`, `b5_wall_impact_vNN` (any speed),
`ram_into_free_wall`, `house_1story_ram`, and any catalogue scene the world supports.
The run logs behind the tables (JSON lines) are not committed.
