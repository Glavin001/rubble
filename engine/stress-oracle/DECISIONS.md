# Decisions log (test targets, tolerances, model fixes)

Running record of the choices made while making every test a trustworthy target
("exact answer, or a derived bound, never a hand-picked number"). Newest last. `User`
marks a choice the user made; `Mine` one I made on my own judgement (the user asked me
to proceed without questions and record them here).

## Rule (User)

When the expected answer is known exactly, the test requires an exact match, or a match
to round-off with the bound derived, not picked. A threshold is allowed only for
genuinely approximate or chaotic results (fracture, oracle comparisons), and each one
states its source: a computed round-off bound, a measured convergence with step size, a
cited textbook factor, or the oracle's own uncertainty.

## Targets

1. **Exact model answers** (User). The expected value is the exact solution of the
   contact model (elastic layer + Coulomb), i.e. the textbook result plus its
   closed-form elastic corrections, not the rigid textbook value with the correction
   folded into a tolerance.
2. **Exact discrete answers where the integrator is the difference** (Mine). Where the
   time integrator itself changes the answer (semi-implicit Euler: a constant-force
   stop at `v^2 / 2a - v dt / 2`), the test checks the simulation against the exact
   discrete value to round-off, and the discrete value against the continuous one
   within the scheme's derived error (`v dt / 2 + a dt^2`, `dt` in time).
3. **Scenes realise the textbook premise** (Mine). Blocks start at the exact contact
   equilibrium of the motion tested (sink and tilt from an independent solution), so
   the textbook answer applies from t = 0 instead of after an unmodelled drop-in
   transient.
4. **Two-field pair target** (User). A cube resting on an anchored chunk is compared
   with the exact equilibrium of the pair law's two-field energy (rim cells, interface
   stiffness jumps, slivers, stiffness-weighted contact point), derived in the test
   (`slab_energy`, `slab_contact_height`) and checked cell by cell against an
   independent semi-analytic integral (bottom cell to 1.2e-13).
5. **Friction acts at the overlap centroid** (User), as the normal force; the targets
   are derived with that convention.

## Round-off bounds (Mine)

6. **Positions and angles at rest**: within `4 ulp` of the stored coordinate (sink), and
   `4 (ulp(z) / 2h + eps)` (tilt). Derivation: stored coordinates carry half an ulp;
   the forces the equilibrium balances come from them along a path of at most eight
   roundings at that scale (coordinate difference, clip interpolation, centroid, arm).
   Observed: 0 to 2.3 units, including the scene lifted 8 m (residuals scale with the
   coordinates, as round-off must).
7. **Settled** = the remaining motion is the contact's vibration at the amplitude of
   that rounding: velocity at most `4 w_v ulp(z)`, spin at most `w_r` times the tilt
   bound, held for a whole rocking period, sampled by frames of 1/8 bounce period (so
   an oscillation shows at least `cos(pi/8)` of its peak, folded into the thresholds).
   A single-frame test was wrong (it caught turning points).
8. **Accelerations**: per frame, the normal force's rounding (`4 ulp(z)` of the sink)
   times `mu g cos t`, plus the velocity's own (`2 ulp(v) / dt`).
9. Run lengths are not tolerances: a run is cut off at ten times the derived decay
   estimate of its slowest mode; failing to settle by then fails the test.

## Model changes made while proving the targets

10. **Rocking damping** (User). The contact dashpot is spread over the area like the
    springs (`c_n / A` per area): relative spin about in-plane axes is resisted with
    `-(c_n/A) Q w`, `Q = int (r x n)(r x n)^T dA`. Before, rocking was undamped (a
    resting heavy block never settled). Contacts now carry the area's in-plane second
    moment tensor, shifted to the force's point of action (pairs and balls act at the
    stiffness-weighted cell centroid; their polar moment had been taken about the
    geometric centroid).
11. **Exact LIFO shear layer** (User). The friction layer is a stack of entry cohorts
    (stiffness, displacement at entry); force `sum k_i (d - d_i)`; growth pushes,
    shrink pops the newest first, slip collapses to one cohort; turned with the contact
    plane by the minimal rotation. Fixes a creep of 8e-14 m/s at a partly lifted
    (near-grazing) contact edge, where round-off makes the area flicker 1e-10 and the
    uniform-strain rule ratcheted the stored force down. Memory grows only while a
    sticking contact's area grows monotonically (accepted, unbounded, User).
12. A contact forming takes all its material as one cohort (it enters together).

## Test scene choices (Mine)

13. **Sticking**: exact settled offset `e = m g sin t / (k_t A)` asserted at 10 degrees
    (cone 2.8 times the load: the start-up step response stays inside, no slip) for
    every mass; no creep (`4 ulp(h)` over the second half of the run) asserted at 10
    and 20 degrees. At 20 degrees (cone 1.37 times the load) the unloaded layer slips
    briefly at the start: a path-dependent offset, not asserted.
14. **Stopping**: one substep per frame; velocities, the stop substep and the stop
    position checked against the exact discrete scheme; the elastic release after the
    stop is not asserted (the friction spring holds exactly `mu N` at the stop, so the
    release can slip a path-dependent amount), only that the block then holds still.

## Workflow (Mine)

15. `scripts/test_fast.sh`: the suite without the panel, hull-pack and showcase runs
    (about 2 min instead of about 25 with `layer_contact`); the full suite gates commits.
16. `.gitignore` (stress-oracle): generated scenes, videos and stray media/log output
    ignored; `golden/` and `golden_mesh/` stay committed (oracle evidence).

## Frame changes (Mine)

17. **Half turns are bit-exact targets** (User asked for bit-exact 180 degrees). A half
    turn about a coordinate axis flips the signs of two vector components (exact) and
    permutes a quaternion's components with signs. Floating point rounds `-a` as `-(a)`
    and pairs commute, so only sums of three or more terms in an order the turn
    changes, and choices made in an arbitrary basis, break it. Fixed (round-off-level
    changes, all runs): quaternion normalisation sums `(w^2 + x^2) + (y^2 + z^2)`; the
    rotation is the explicit matrix with entries grouped `(ww + xx) - (yy + zz)`,
    `2 (xy - wz)`, ... (each maps to an exact sign flip of `D R`); the Hamilton product
    sums each component's terms in pairs (the sequential form reversed its order under a
    right multiplication by a body-frame increment, as the floating frame's drift removal
    does; checked exhaustively on random factors, left and right); a clip's new face
    starts at the hull vertex met first among the segments instead of the extreme of an
    arbitrary in-plane basis. Found by a differential run (both scenes stepped substep
    by substep, every state compared after mapping). **Status:** without fracture, all
    three half turns bit for bit (was 2e-14 m); with fracture, exact until the ball first
    meets a chunk edge (substep 457), where `with_ball` (in-plane basis from
    `any_perpendicular`, absolute angles, the `r^3`/`r^4` cancellation of item 29) gives
    a 4e-9 relative difference in the contact force; the run then differs by 3e-12 to
    1.2e-11 m. Remaining fix: basis-free, cancellation-free ball-cap moments (sector
    moments from the unit vectors of the arc's ends, relative angles only). A covariant
    in-plane basis cannot exist (a half turn about the normal itself would have to fix
    it), so every such computation must be basis-free.
18. **Galilean and generic rotations: a rounding envelope, not a tolerance.** Floating
    point cannot make them exact (`(v_a + d) - (v_b + d)` rounds). The transformed run
    is compared with 19 runs of the original into which, after every substep, each
    stored coordinate of the moving state receives +-1 ulp (fixed pseudo-random signs)
    at the scale the transformed run computes it (positions at the largest coordinate
    reached, linear velocities at the largest speed, other vectors at their own length,
    quaternions at 1): two half-ulp roundings per coordinate per substep. The
    transformed run must deviate no more than the largest of them (if it were just
    rounding, a 1/20 chance of exceeding; seeds fixed, deterministic). Perturbing only
    the initial state was tried and is wrong: a frame change re-rounds every substep
    (it underestimated 25 times). Measured: without fracture 7.6e-14 m against
    2.4e-13; with fracture 2e-10 m against 8e-7 (rounding can flip a damage event at
    substep resolution, so the fracturing envelope is wide; the no-fracture case is the
    sensitive check). Runs pin the substep, which the partition test proves changes
    nothing.
19. **Convergence and frame partition**: monotone decrease of differences over Courant
    safety 0.5, 0.25, 0.125, 0.0625 (no tolerance); frames of 16, 4 and 1 substeps give
    bit-identical states every 16 substeps.
20. **Settled is judged on positions, not velocities**: a velocity whose step `v dt` is
    below half an ulp of the position is rounded away and moves nothing, so a
    "phantom" velocity can persist forever in a block that is exactly still.

## Conservation (Mine)

21. **Linear momentum**: the 1e-9 relative allowance became a derived round-off bound:
    every substep each body's velocity rounds along at most eight half-ulp roundings at
    the largest speed, `4 ulp(speed)` per unit mass, accumulated linearly, plus the
    rounding of measuring the sum (`2 (N + 8) eps sum m |v|`, twice). Measured drift:
    4e-5 to 3.4e-4 of it (the bound is a worst case; rounding walks randomly).
22. **Splits** (fragments inherit `v + w x r`): the 1e-12 became Higham's
    `gamma_{N+10}` of the sums of the terms' magnitudes (three passes: measure, split,
    measure). Measured: 9e-16 against 4.6e-13.
23. **Energy balances converge to zero, judged by the data** (no tolerance): errors at
    Courant safety 0.5, 0.25, 0.125, 0.0625 (two cubes: down to 1/32) must fall in
    magnitude at every halving, and Aitken's extrapolated limit from the last three
    must lie within its own uncertainty (its change from the previous three) of zero.
    A single Aitken estimate was not enough: the two-cube "face" error alternates in sign
    as it converges (1.4e-3, 8.9e-5, -2.2e-5, 1.1e-7, -5.7e-8). A rounding floor
    (`substeps * 4 eps`, decision 6) ends the trend. Replaces 0.01, 0.05, 0.025, 1.001,
    3e-3, 1e-3 and the 0.6 ratio.
24. **Second law of the contact ledger**: exact, no allowance (the 1e-9 e0 is gone).
    With the penalty contact the booked dissipation is a sum of non-negative terms
    (dashpot `c vn^2`, unloading `k depth max(vn, 0)`, friction `|f_t| |v_t|`, crush),
    which floating point keeps exactly monotone. With the elastic layer it is the work
    the contacts took out of the motion net of what they store; not monotone by
    construction (the elastic integration term `-1/2 dx K dx` per substep can be
    negative) but measured never to fall frame to frame at any safety; kept exact.
25. **The penalty contact's energy closure is not tested** (as for the two cubes): it
    books its own dashpot quadrature, which is not the work its non-gradient force does
    (0.33 e0 booked against 0.34 e0 taken out of the motion at safety 1/16; the
    imbalance grows from 0.4% to 2.3% as the substep shrinks). Its ledger monotonicity
    is still tested.
26. (Superseded by 35-36.) **Angular momentum is an exact target** (separate test, derived round-off bound with
    lever arms to the scene's extent). **Known failing**: the small-strain bond model
    balances moments about the undeformed chunk centres and the floating frame couples
    hidden and rigid motion to first order; the drift does not shrink with the
    substep (5.9e-4 of |L0| without fracture, 1.2e-3 to 1.6e-3 with it).
27. (Resolved by 33: the joint law's ledger.) **Fracture energy closure (layer contact) is an exact target. Known failing**: the
    residual converges at first order (ratios 0.48) to about -1.3e-3 e0 (Aitken limits
    -1.31e-3, -1.38e-3, -1.37e-3, -1.26e-3 down to safety 1/128). Located: none of it
    in substeps without contact (2.5e-9), none at splits or bond breaks (those parts
    converge), none without fracture (40 m/s elastic: exactly first order to zero), not
    friction, not the joint-law switches; the bond law alone closes (work = stored +
    dissipated + overshoot, converging, on tension, mixed, cyclic and compression-shear
    paths). After removing the symplectic quadrature term `sum 1/2 m |dv|^2` (first
    order, 3.7e-3 then 0.91e-3) a negative part of -1.82e-3 e0 is the same at safety
    1/8 and 1/32: a loss at a fixed rate while contacts push loose cracked chunks, the
    signature of the same floating-frame coupling as item 26. Fix (future, behind a
    switch): exact rigid/hidden coupling (moments about deformed centres, frame
    accelerations in the hidden dynamics).

## Bugs found by the audit (fixed)

28. **NaN from a sliver contact** (layer contact, safety 1/64, 40 m/s fracture): the
    contact area's second moment was formed about the world origin and shifted to the
    contact point; for a sliver with large, nearly edge-on faces the shift cancelled to
    the size of the true value (polar moment -1.1e-21), the torsional stiffness went
    negative and `sqrt` gave NaN. Fixed: each face's moments are integrated with its
    vertices already shifted to the point and projected (a sum of outer products with
    positive weights), and `LayerContact::about` separates the central moment so the
    shift only adds `a e e^T` (the central polar moment, analytically `>= 0`, is clamped
    at zero against round-off). Regression test
    `a_sliver_contact_has_a_positive_area_moment` (fails on the old formulas: -1.8e-35).
29. **Open (round-off)**: `with_ball` forms a shallow cap's volume and first moment from
    sector formulas of size `r^3`, `r^4` (cancellation): on a 1e-16 m^3 cap the centroid
    is off by about 1e-4 m and the stored energy can come out -2e-9 J. Negligible in
    energy (1e-13 of an impact's), but unphysical; a cap-height formulation would fix it.

## Gradient and geometry checks (Mine)

30. **Force = -grad(energy), without a chosen step or tolerance** (`world.rs` tests,
    replacing 1e-5, 1e-4, the 0.1 torque scale and the fixed steps 1e-6, 1e-9, 1e-10).
    Five-point central differences of the energy over a ladder of steps
    `d = size 2^-k`, each with the classical error bound: rounding
    `N(d) = 1.5 sigma_E / d` (the stencil's weights on energies each off by at most
    `sigma_E`) and truncation `T(d) = (|D(2d) - D(d)| + N(d) + N(2d)) / 15` (error
    `c d^4`); the step minimising `T + N` is used. `sigma_E` and the analytic force's
    own rounding are measured by moving the whole scene rigidly (19 fixed shifts;
    nothing physical changes), each vector's rounding at its own length. Pass: the
    analytic value within `T + N` plus its rounding. A first version took the spread of
    neighbouring differences as the uncertainty and chose the step with the smallest
    spread; that biased it low (one random case in 3000 at 1.01, where the ladder shows
    analytic and differences agreeing to 1e-11). Measured misfits: 0.54 at most on the
    fixed cases, 0.88 the worst of 3000 random checks; a force wrong by 1e-7 relative
    fails at 47 times the uncertainty (old tolerance 1e-5).
31. **Ball-box overlap against an independent integration** (`polytope.rs`, replacing
    the 3000^2 midpoint brute force and its 2e-6): nested Gauss-Legendre over exact
    slices with the rim square roots substituted away and every kink split out (a
    signed-distance slip that missed the kink where a chord enters the box was found by
    the convergence check and fixed). Uncertainty: 64 against 128 points per piece plus
    the worst-case rounding of nested sums, `(N_inner + N_outer + 16) eps` of their
    scale; plus `with_ball`'s rounding by rigid shifts. Agreement 1e-15 to 1e-13
    relative; uncertainty 1e-12 to 1e-9 (was 2e-6).
32. Removed a leftover scratch test in `world.rs` (`zz_slab_energy`, which always
    panicked).

## Second round: physics fixes found by the exact targets (Mine)

33. **Joint law energy ledger (bug, fixed, all runs).** The law's own accounts (the
    softening law's `int U0 kappa^2 dD`, crushing, friction, rebar) are exact only for
    proportional single-mode loading; on general paths they booked up to twice the
    energy the joint actually released (the cracked share still held it: compression,
    rocking, friction-held shear). Found by a new test
    (`the_joint_law_balances_energy_on_any_path`: 40 random smooth paths through every
    mechanism; old law: -32 % at every step size). Now the dissipation is exactly the
    drop of stored energy at the new displacement caused by the internal-variable
    update (`frozen - stored`, zero exactly when nothing changed); the law's share is
    `bond_dissipation` (clipped to that total), the rest `softening_overshoot`. Forces
    and dynamics are bit-identical. Closure converges at second order (trapezoid); no
    update ever releases negative energy. This was the whole fracture-energy residue
    of item 27: the 40 m/s fracture balance now halves with every halving of the
    substep (2.6e-3 at safety 1/32, 1.24e-3 at 1/64) and its test passes.
34. **Half turns bit for bit, with fracture** (item 17 completed): `disc_region`'s
    sector moments from the unit vectors of the arc's ends (relative angles only), not
    absolute angles in an arbitrary in-plane basis. All three axes, with and without
    fracture, `layer_contact` on: position and velocity differences exactly 0.
35. **`exact_frame_transfer` (new switch).** The floating frame's transfer of a fragment's
    hidden rigid motion is linearised (positions move by O(phi^2), velocities by
    O(phi v)), harmless per substep (phi = O(dt)) but not for the finite fold of a new
    fragment; and the split re-expressed velocities with lever arms that left out the
    hidden displacement. Together they changed angular momentum and kinetic energy at
    every split, by amounts that do not vanish with the substep (a deformed beam's
    split: 1.0e-2 in L, 5.1e-2 J created). With the switch, a split keeps every chunk's
    position, orientation, velocity and spin exactly (re-expressed in the child's frame,
    then re-centred so the hidden mean displacement and velocity stay zero, the frame
    taking them, lever-arm change included). Applying the exact transfer every substep
    was tried and rejected: re-deriving the hidden displacement from world positions
    adds rounding at the scale of the positions to a 1e-6 m field (it broke the
    resting-box and momentum round-off targets). Tests: `a_split_of_a_deformed_body
    _keeps_its_motion` (round-off), the angular-momentum ledger below.
36. **Angular momentum: an exact ledger, not a tolerance.** The small-strain bonds act at
    undeformed lever arms (their linear kinematics are invariant only under
    infinitesimal rotations of the undeformed configuration), so internal forces turn
    the system by `sum (u_a - u_b) x f` (about 5e-4 of |L0| in the 40 m/s impact). The
    test integrates this imbalance and requires `L - L0 - imbalance` to converge to
    zero with the substep: nothing else (contacts, splits, frame, rigid update) may
    create angular momentum. Removing the imbalance itself needs co-rotational
    (geometrically exact) bond kinematics with consistent rotation variables:
    `corotational_bonds` (52).
37. **Engine API reported fractures incompletely (bug, fixed).** A fragment that split
    again in the same frame was dropped from its parent's report and its own split
    reported under an id the engine never saw. Each body the engine knew is now
    reported once with every piece it ended as; the test checks the pieces carry all
    of the body's mass.
38. **Mass scaling meets its target exactly (fixed).** The one-shot `(target / dt)^2`
    ignored the dashpots and, with the mass-normalised bound, the neighbours' inertia
    (5 % short). Each chunk's factor is now found by bisection on its own damped stable
    substep (the same bound as `stable_dt`); scaling a neighbour only lowers a chunk's
    rows, so one sweep meets the target everywhere. The test's 0.99 allowance is gone.
39. **`contact_crushing` (new switch): the joint's crushing law on the contact that
    replaces it.** Over-strong arching on b9 (fragments rebounding against the pulse)
    came from contacts with the joint's stiffness but unlimited strength. A perfectly
    plastic crush (indentation where the layer's peak pressure `k''_f s` is `f_c`) was
    tried first: it dissipated 3.1 kJ of the 6.7 kJ input and left fragments 35 % slow.
    The joint law's crushing is energy-limited, so the contact's is too: strength
    `f_c (1 - c / d_u)`, `d_u = 2 G_c / f_c` (concrete: 20 kJ/m^2, 1.33 mm), the
    increment from the cell's linear softening equation, crushing through when softening
    outruns the layer. b9 high against OpenCourant: fragment speed 2.79 vs 3.45 m/s
    (19 %, gate 30 %), peak centre displacement 9 % (established: 22 %).
40. Engine and mass-scaling tests that had failed with `layer_contact` now pass (37,
    38); the candidate set is `layer_contact, scaled_step_bound, exact_frame_transfer,
    contact_crushing`.
41. **The proven methods are the default** (the protocol's "then its default flips"):
    `layer_contact`, `scaled_step_bound`, `exact_frame_transfer`, `contact_crushing`. With
    them the whole suite passes (exact targets, conservation ledgers, gradients, oracle
    goldens, analytic benchmarks); the established methods fail the exact targets
    (contact equilibrium, Coulomb sticking/sliding/stopping, unit rescaling,
    convergence, angular momentum, motion kept through a split) and stay available as
    `STRESS_METHODS=established` (or `-NAME` per switch) for A/B comparison. Not yet
    proven and left off: `exact_joint_patch`, `patch_after_damage`, `exact_rate_filter`,
    `frame_contact_step`. Open before the next round: the catalogue-wide A/B
    (`stress-ref ab`) of the new default, and co-rotational bonds (36).
42. **Catalogue A/B of the new default** (`stress-ref ab scenes golden`, B = every
    proven switch off): gated metrics against the oracles pass A 87/91, B 87/91, the
    same four failing on both sides, all the README's known b5 gaps: at 2 m/s the
    push-over is not converged at the default substep (speed lost A 51.6 %, B 41.2 %
    off); at 40 m/s the oracle itself is not converged (its mesh study: far-field
    cracks keep growing with refinement, no crushing in its elastic chunks) and both
    make a hole where it pushes the wall over (speed lost A 49 %, B 33 % off). No
    metric passes with the established methods and fails with the new default. Cost:
    wall time 339 s against 162 s over the catalogue (the layer contact; the perf plan
    covers it). The b5 speed-lost gap is a lead for the next round: the new default is
    further from the (unconverged) oracle there.
43. **b5 at 2 m/s converges with the proven default** (speed lost 1.441, 1.423, 1.425,
    1.418 m/s at safety 0.5 to 1/16; the README's "not converged" gap was the
    established contact's) and agrees with the rigid-pivot momentum balance (plastic
    limit 1.438 m/s lost). OpenCourant's 0.951 needs the wall to shatter under the ram
    (805 broken cohesive elements at 2 m/s, no crushing in its elastic elements): the
    remaining b5 failures are disagreements with an oracle whose contact stresses are
    uncapped, not convergence or ledger failures of ours. Closing them needs the
    oracle rerun with a crushing law (it runs on Linux; not available here).
44. **Co-rotational bonds: designed, deferred** (User: not now; documented). The
    angular-momentum leak of the small-strain bonds (about 5e-4 of |L0| at 40 m/s,
    item 36) goes only with a bond energy that is invariant under finite rigid rotations
    AND rotation variables used consistently by the hidden integrator; correcting only
    the lever arms (second-order terms) leaves a leak of the same size, because the
    additive rotation vector `th` turns the geometry at `J(th) w`, not `w`. The design:
    (a) each chunk carries an exact hidden orientation `R` (cluster frame) advanced by
    `exp(w dt)` (`w` the spin), `th = log R` kept only for compatibility; (b) the bond
    strain is the gap between the deformed attachment points `g = (u_b + R_b r_b) -
    (u_a + R_a r_a) - (r_b - r_a)` seen from the pair's mean frame `R_m = R_a
    exp(phi/2)`, and the relative rotation `phi = log(R_a^T R_b)`; (c) the loads are the
    exact gradient with respect to displacements and spins: `f = R_m q_lin`,
    `m_a = (R_a r_a) x f + (I - K)^T (g x f) + R_a J_l^-1(phi)^T q_ang`,
    `m_b = -(R_b r_b) x f + K^T (g x f) - R_a J_l^-1(phi)^T q_ang`, with
    `K = R_m J_r(phi/2) J_l^-1(phi) R_a^T / 2` (the mean frame's share of `b`'s spin);
    they balance exactly about any point; (d) the dashpot rate is the same Jacobian
    applied to the hidden velocities and spins; (e) the drift fold turns `R` by
    `exp(-phi)`, the split's exact transfer sets it, the implicit, static and refinement
    paths set `R = exp(th)`. Behind a `corotational_bonds` switch, proven by the
    angular-momentum test (the ledger then zero) and the energy tests. Not started in
    the code (a partial wiring was reverted).
45. **b5 at 2 m/s against an analytic estimate** (User: try it): the rigid push-over by
    momentum alone (the wall turning about its toe, `I = m (h^2 + t^2) / 3`, perfectly
    plastic at 1 m) loses 1.442 m/s; ours 1.418 (1.7 % below: the wall is not rigid),
    OpenCourant 0.951 (34 % below). Added as a report metric
    (`speed_lost_rigid_push`, not gated: an idealisation).
46. **PhysX coupling verified on Linux (Docker, CPU-only SDK).** The PhysX SDK's
    `linux-aarch64-clang-cpu-only` preset builds in Docker (snippets and PVD runtime
    off: they need OpenGL); Blast's `build.rs` now compiles the adapter without CUDA
    when the Linux SDK ships no GPU module (`PxCreateCudaContextManager` was left
    undefined). `stress-physx` tests: 15 passed, 1 ignored. `stress-physx check`: every
    gated metric but the four known b5 ones.
47. **The coupled world takes the standalone world's substep (fix).** Without impact
    islands it took only the stress limit, a larger step than the standalone world's
    under `scaled_step_bound`; b3's reflected tension (1.25 %) and b4's dynamic
    amplification (1.08 %) then failed their 1 % analytic gates in the coupled world
    only. The bar's converged answer is 1006.6 (0.66 %: the chunk lattice); the larger
    step added 0.6 % of time-step error. The stress solution must not depend on who
    owns the rigid motion; with the established bound the two worlds already agreed
    exactly. (The standalone contact limit assumes every chunk pressed on all faces,
    conservative when nothing touches; `frame_contact_step` would lift it in both.)
48. **OpenCourant rerun (x86 Docker, pinned release, hash-verified)** with a new
    optional crushing law for the chunk solids (`--chunk-crushing`: J2 plasticity at
    `f_c`, eroded at the plastic strain dissipating `G_c` per area over one element;
    written as oracle `opencourant_crushing`, off by default so the goldens
    reproduce). b5 at 2 m/s, speed lost: golden (1 element per chunk edge) 0.951;
    Docker K = 1 0.940 (reproduces it); K = 2 1.055; K = 2 with crushing 1.068; ours
    1.418; rigid push-over 1.442. Refining the oracle's mesh moves it toward ours
    (fewer broken bonds: 805, 734, 642); crushing barely matters at 2 m/s. K = 3 and
    40 m/s with crushing: see below.
49. **Oracle mesh study at 2 m/s** (speed lost): 1, 2, 3 elements per chunk edge give
    0.940, 1.055, 1.113 m/s (broken bonds 734, 642, 712): rising with refinement and
    slowing; extrapolation is uncertain (first order 1.17-1.23, a fitted order of 0.3
    gives 1.56); ours 1.418 and the rigid push-over 1.442 lie inside that range. The
    oracle at the golden's mesh (one element per chunk edge) is not converged; 4
    elements per edge is running.
50. **Oracle with crushing at 40 m/s** (`--chunk-crushing`, two elements per edge): the
    J2 elements invert before eroding at this speed (the first run stopped: "infinite
    domain" in the contact); with time-step erosion (`/DT/BRICK/DEL` at a tenth of the
    run's step) it completes, speed lost 4.36 m/s (elastic oracle 5.97, ours 3.04, now
    30.2 % off), still a push-over with a 2.88 m^2 hole region, 1440 broken bonds (ours
    49). But its kinetic energy ends at 3.2 MJ against the ram's 0.8 MJ: the run creates
    energy (crush-and-erode at this speed), so it is not a valid reference. The 40 m/s
    gap remains unresolved: neither the elastic oracle (not mesh-converged, uncapped
    contact stresses) nor this crushing variant can arbitrate it; a proper concrete
    law in the oracle (LAW24 / LAW124 CDPM2) with an energy check is the next step.
51. **Oracle mesh study at 2 m/s, four meshes:** speed lost 0.940, 1.055, 1.113, 1.206
    m/s at 1, 2, 3, 4 elements per chunk edge (broken bonds 734, 642, 712, 768): still
    rising, not converged (the increments do not shrink steadily), every refinement
    toward ours (1.418) and the rigid push-over (1.442). The committed golden is the
    coarsest mesh; the b5 2 m/s gate failure is a property of that golden, not evidence
    against the model. A converged oracle needs finer meshes than this machine can run
    under x86 emulation in reasonable time (4 per edge took 54 min).
52. **Co-rotational bonds (`corotational_bonds`, switch).** Answers 36's open item for
    the explicit integrator. Each chunk turns by the finite rotation of its rotation
    vector. A bond deforms by its patch points' separation
    `(u_b - u_a) + (R_b - I) r_b - (R_a - I) r_a` (formed without cancellation against
    the positions, cf. 35) and by the relative rotation `log(R_a^T R_b)`, both read in
    the mean frame `R_a exp(phi / 2)`.
    - The chunk loads are the exact transpose of the map's variation, with moments
      conjugate to spatial rotation increments. The integrator then composes rotations
      exactly (`R <- exp(w dt) R`, no longer `th += w dt`).
    - Objective: a rigid rotation deforms nothing, so internal forces balance moments
      about the deformed configuration. They are the gradient of the bond energy.
    - Angular momentum on the 40 m/s impact: small strain stays at 5.94e-4 of `|L0|` at
      every substep. Co-rotational: 5.2e-5, 2.6e-5, 1.3e-5, 6.6e-6, converging to zero
      (extrapolated 2.4e-8).
    - The fracture energy closure (27) moves from -2.4e-3 to +2.7e-4 of `e0`.
    - Tests: `corotation_ignores_a_finite_rigid_rotation`,
      `corotation_variation_is_the_derivative_and_loads_its_transpose`,
      `corotational_bonds_conserve_angular_momentum`.
    - The static and implicit solves keep the small-strain map.
53. **`exact_joint_patch`, `patch_after_damage` and `exact_rate_filter` join the proven
    default.** 41 left them off as not yet proven. Each is a correction:
    - the cracked joint's contact layer is integrated exactly over its patch (the
      6 x 6 grid put its outer spring at 0.417 of the width: rocking 17 % low, bending
      2.8 % low);
    - no friction slip accrues on an interface that has not cracked;
    - the strain-rate filter is integrated exactly over each step.

    The bar, measured against the default with explicit windows (b9cadbb27):
    - suite 133/133 with the three on (134/134 as the default, with the shallow
      sphere cap's depth now formed without cancellation);
    - catalogue A/B 87/91 both ways, all three together and each alone, with no metric
      flipping either way;
    - wall time -6 % to +5 %.

    The gate changes 51 of 182 cells, the cracked-joint and rate-effect scenes (b1
    shear, b5, b6, b7, hulls, showcases).
54. **Static equilibrium's energy is booked (`static_discard`).** Adaptive settling,
    settled fatigue, the quasi-static solve and refinement replace the hidden state, and
    the energy they removed went unbooked. Around the frame's static work the ledger
    now books `before + W - after` of mechanical plus dissipated energy. `W` is the work
    of the loads the solve used over the chunks' displacement and rotation: the
    contact-work ledger integrates `f . v` over substeps, and a static displacement has
    no velocity.
    - A static step minimises the total potential, so where it converges the term is
      non-negative: cantilever 0.14 J, frame 15.0 J, b9 8.2e3 J, b4 4.8e-7 J. Test:
      `a_converged_static_solve_never_creates_energy`.
    - **b7_v15 quasi-static books -2.6e6 J**, matched by its softening overshoot
      (+2.595e6 J), net about +7 kJ. Three unconverged frames (the no-equilibrium
      fragment class) leave states holding megajoules, which snap in the damage commit
      and book as overshoot; the discard nets them out.
    - Gate: 182/182 identical apart from the new value, which is reported only in
      quasi-static, adaptive and refining runs.
55. **Adaptive: settled clusters bound the substep.** The frame's substep is chosen at
    its start, and settled clusters did not limit it. A load that woke one mid-frame
    (a removal, debris) then had it integrated explicitly at a step above its stable
    stress step for the rest of the frame. Every bonded cluster now limits the
    adaptive substep; implicit mode keeps limiting it by its windowed clusters only.
    - Gate: only the 14 adaptive cells change. Suite 135/135.
    - Keystone arch, adaptive: 17796 -> 18200 substeps (explicit's count), and 9 -> 8
      broken bonds, inside explicit's own spread under tiny substep changes (4-8).
    - **Resolved: chaos.** Adaptive keystone contact dissipation is about 1550 J
      against explicit's 838-1059 J (14 substep perturbations).
      - It is not the wake rule (wake threshold 0.02, 0.002 or 0, active time 0.5 or
        100 s: identical runs), static fatigue or rate effects (off in both: unchanged).
      - The two runs agree to 1e-6 in mechanical energy until 0.383 s; the first
        difference is the sign of a zero split release (-2e-29 against +3e-29).
      - At 0.383 s they part by 6 J within one frame: a bond at its threshold breaks in
        one and not the other.
      - The active time 1e-4 s run (919 J) is another realisation, not a fix.
56. **Damping study (audit item 9): `rayleigh_damping`.** Measured first.
    - The bond dashpots are stiffness-proportional, the material's `damping_ratio` at
      each bond's axial frequency, so a mode at `w` gets `zeta w / w_b`.
    - The 2.7 Hz cantilever's fundamental mode decays at about 1e-4 (log decrement over
      12 cycles) whether the material says 0, 0.02 or 0.05. A real 2 % beam would lose
      78 % of its amplitude over those cycles.
    - The switch adds a mass-proportional dashpot on the hidden (deformation)
      velocities, `alpha = 2 w1 zeta (1 - w1 / w_b)`, fitted per anchored structure
      from `fundamental_period` and the bonds' mean ratio and frequency. Rigid motion
      is not damped. A single bond (`w1 = w_b`) is unchanged, so b4's damped
      amplification holds.
    - It acts in the explicit integrator and in Newmark (residual and tangent), and its
      work is booked as damping dissipation.
    - Measured: explicit 0.0195 and 0.0488 for material 0.02 and 0.05, implicit 0.0192;
      bond dashpots alone about 1e-4. Test:
      `rayleigh_damping_gives_the_fundamental_mode_its_damping_ratio`.
    - **Found on the way:** `fundamental_period`'s inverse iteration started from a
      vector mixing metres and radians, so the period depended on the units at
      round-off. That broke `rescaling_every_unit_reproduces_the_run` with the switch
      on. The start vector is now in consistent units, and the run is bit-exact again.
    - Switch off: 182/182 identical. On: suite 136/136, catalogue A/B 87/91 both ways,
      no flips, cost unchanged.
57. **`rayleigh_damping` joins the proven default, with the exact two-frequency fit.**
    - The first fit (56) kept the bond dashpots and set `alpha` for `zeta` at `w1`. It
      was exact only for `w1 << w_b` and over-damped a mode at a bond's own frequency by
      `alpha / (2 w_b)`, up to `zeta / 4`. b4's middle chunk has shear and rocking modes
      below its axial one, so its reaction moved 1 %.
    - Now `alpha = 2 zeta w1 w_b / (w1 + w_b)`, and every bond's dashpot is scaled by
      `w_b / (w1 + w_b)`. That gives `zeta` exactly at the fundamental and at each
      bond's axial frequency (exactly for equal bonds).
    - b4's damped amplification is 1.9439 against the analytic 1.9391 (0.25 %; 0.41 %
      without Rayleigh). The cantilever's fundamental: 0.0195 and 0.0488 for 0.02 and
      0.05 explicit, 0.0192 implicit.
    - Suite 136/136 as the default. Catalogue A/B against Rayleigh off: 87/91 both
      ways, no flips, wall +4 %. The gate changes 66 of 182 cells, the damped anchored
      structures.
58. **Rayleigh's mass-proportional term acts on anchored clusters only.**
    - Applied to a free cluster (a fragment of an anchored structure under 57, or a
      free structure once `fundamental_period` took free clusters), it exerts a net
      torque. The hidden field carries a little angular momentum about its deformed
      configuration between drift removals, and the dashpot removes it:
      `angular_momentum_changes_only_by_the_bond_imbalance` and
      `corotational_bonds_conserve_angular_momentum` fail.
    - An inertia-relief projection about the deformed configuration restored both
      (errors converging to zero). But it broke
      `a_rigid_rotation_of_the_scene_rotates_the_result` with fracture: 0.19 m and
      19 m/s apart from 3 ms on, against a rounding envelope of 3e-11.
    - Bisected: Rayleigh off passes; damping without the projection passes; with the
      projection it fails. The frame dependence is not found. Open.
    - So free clusters and fragments keep the stiffness-proportional bond dashpots,
      and the supports of an anchored cluster take the dashpot's net force.
    - `fundamental_period` now also takes free clusters: inverse iteration deflated
      against the rigid modes (inertia relief, the PCG on the complement, rigid part
      removed). Free-free beam 16.36 Hz against Euler-Bernoulli 16.48 Hz (-0.7 %).
      Test: `a_free_cluster_has_the_free_free_bending_period`.
    - Suite 137/137; A/B against Rayleigh off 87/91 both ways, no flips. The gate
      changes the cells whose fragments came off anchored structures (b5, b6, ...).
59. **Ball contact: a sliver cell took the whole disc (bug, fixed).** `ball_cells`
    splits a ball's overlap with a chunk into nearest-face cells. `disc_region`
    integrates a face polygon against the ball's disc. Where the polygon's boundary
    never meets the disc, it took the whole disc when every edge's cross product with
    the foot was non-negative.
    - A cell's face can degenerate to a near-collinear sliver, whose cross products are
      zero or rounding-positive, so the sliver "held" a disc of radius 0.054 m.
    - In the free impact turned by 0.7 rad (the rotation principle test, with Rayleigh
      on free clusters, 58) one cell came out 1.04e-4 m^3 in a 2.1e-7 m^3 overlap, depth
      -0.05 m. The contact stored -2.6e6 J and a chunk got 45 m/s in one 2 us substep.
    - Found by tracing the original and turned runs side by side: bonds and
      `with_ball` overlaps identical, only the impactor contact's stored energy parted.
    - **Fix:** the disc is held only if the polygon also covers at least the disc's
      area (`0.5 sum cross >= pi rho^2`), a necessary condition, so valid
      configurations are unchanged.
    - Tests:
      - `ball_cells_of_a_struck_chunk_add_up_to_its_overlap` (the captured
        configuration, bit for bit; it fails before the fix);
      - `ball_cells_partition_the_overlap_in_every_orientation` (20,000 shallow edge
        and corner contacts: the cells' volumes match the overlap to 4e-16 r^3).
    - Gate 182/182 identical (no catalogue scene met the sliver); suite 139/139.
60. **Rayleigh on free clusters, with `corotational_bonds`.**
    - With the ball-contact fix (59), the deformed-configuration projection of 58
      passes `a_rigid_rotation_of_the_scene_rotates_the_result`: 1.2e-11 m against a
      rounding envelope of 3.2e-11. The 0.19 m divergence was the sliver cell, not the
      projection.
    - With small-strain bonds, free-cluster damping tips
      `fracture_never_gains_energy_and_dissipation_is_accounted`. Its errors halve with
      the substep (4.1e-2 ... 1.9e-3) but extrapolate to -1.0e-3: the known
      small-strain closure offset (27; the current default extrapolates to -2.4e-3 and
      passes only because its estimate moves more). With co-rotational bonds it passes
      (+7.5e-4).
    - The projection balances moments about the deformed configuration, as only the
      co-rotational bonds do. So free clusters and fragments take the mass-proportional
      dashpot when `corotational_bonds` is on; anchored clusters always.
    - Gate 182/182 identical (default). Suite 139/139 as the default, and 139/139 with
      `corotational_bonds`. With co-rotation on,
      `an_impact_converges_as_the_substep_shrinks_and_does_not_depend_on_the_frame`
      now passes at four points: the obstacle to its default (52) is gone.
61. **`corotational_bonds` joins the proven default.**
    - 52 left it off for one failure:
      `an_impact_converges_as_the_substep_shrinks_and_does_not_depend_on_the_frame` at
      four points (1.6 % past its strict bar, within the run's noise). With the
      ball-contact fix (59) and the free-cluster damping (60) it passes unchanged.
    - Suite 139/139 as the default. Catalogue A/B against off: 87/91 both ways, no
      flips; wall 113.5 -> 141.8 s (+25 %). The gate changes 71 of 182 cells, every
      explicit bond evaluation.
    - Angular momentum is conserved to integration error (52). The fracture closure
      offset of the small-strain map (27) falls, and fragments now take Rayleigh
      damping (60).
    - The static and implicit solves keep the small-strain map (their geometric
      tangent is open, see IMPROVEMENTS.md).
