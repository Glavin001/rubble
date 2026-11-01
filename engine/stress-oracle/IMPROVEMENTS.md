# Improvements under proof

Every change to a method of the reference solver goes behind a switch whose default is
the established method, and is adopted (its default flipped) only when its A/B diff over
the scene catalogue and the test suite meets the bar of its class. This file is the
ledger: what each switch replaces, why, the evidence, and its status.

## Protocol

1. **A switch.** A field of `sim.methods` (`scene.rs`, `Methods`), off until proven. It
   is set like a model switch: `--feature NAME=on|off`, `--set sim.methods.NAME=true`,
   or for a whole run of the test suite or the catalogue `STRESS_METHODS` (`NAME` or
   `NAME=on` on, `-NAME` or `NAME=off` off, `established` for every established method;
   it adjusts the default of every scene that does not set `sim.methods` itself; written
   scenes omit `sim.methods` while it is at the built-in default). **Proven switches
   are on by default** (`Methods::proven`): `layer_contact`, `scaled_step_bound`,
   `exact_frame_transfer`, `contact_crushing` (DECISIONS.md 41).
2. **The principles** (`tests/principles.rs`): laws every faithful simulation obeys
   whatever its internals, which a cap, a regularisation or a hand-picked constant
   fails even when every benchmark gate passes: bit-exact reproduction under a change of
   units, invariance under uniform velocity and rotation, Coulomb friction (sticking
   without creep, the sliding acceleration, the stopping distance, all independent of
   mass and substep), a box resting flat with only its elastic compression, and
   convergence with the substep and the frame step.
3. **An A/B run.** `stress-ref ab scenes golden --feature NAME=on [--json out]` runs
   every benchmark scene both ways and prints, per scene, every gated metric against
   its oracles for A and B (a metric passing in A and failing in B is a regression and
   fails the command), every recorded value's change, and wall time and substeps. And
   `cargo test --release -p stress-ref --no-fail-fast` with and without
   `STRESS_METHODS`.
4. **Determinism.** `repeated_runs_are_bit_identical` passes with the switch on.
5. **Shadow mode** for anything that skips work: compute the decision, log it, check the
   full simulation never contradicts it, before acting on it.

| Class | Meaning | Bar |
|---|---|---|
| A, bit-exact | skipped work would have produced identical state | observations byte-identical |
| B, closed form | history advanced by formula | same events (kind, bond); times within one frame; continuous metrics within 1e-6 |
| C, outcome-exact | a proven bound shows no threshold is reached | same events, fragments, momentum; energy ledger closes |
| Step change | a different stable substep | within the refinement study (< 1 % on waves, < 10 % on fracture metrics); every oracle gate passes |
| Model change | the physics differs | every oracle gate passes; the diff report reviewed by a person before adoption |
| Fix | the established method violates a principle | the principle tests pass; every oracle gate passes or the change is explained |

## Ledger

### `layer_contact` (fix + model change)

**Replaces** sample-point penalty contact (`world.rs`, `penalty_force`, `pair_contact`):
box corners pulled in 10 % on all three axes plus face centres; stiffness shared per
point by hand-picked counts (`k / max(n, 10)` between chunks, `k / max(n, 5)` on the
ground, `k / min(n, 5)` for an impactor on the ground); a contact first sampled deeper
than `2 |v_n| dt + 1e-9` m turned into a zero-force offset; friction viscous below
1e-3 m/s and capped with the dashpots at `m_red / (max(n, 10) dt)`.

**Why** (`tests/principles.rs`, established method):

* A box set down flat rests on its single face-centre point (the pulled-in corners sit
  10 % of the half-height above the face): under a 5-degree sideways load it tips 6.7
  degrees, rocks at 1.4 rad/s and sinks 6e-4 m, 20,000 times its elastic compression.
* A block below the friction angle creeps forever (7e-4 m/s, `1e-3 m/s * tan t / mu`).
* A sliding block stops at half the Coulomb distance (it tips onto its leading edge).
* Results depend on the unit of velocity (`FRICTION_REGULARIZATION`).

**Method** (`polytope.rs`, `world.rs` `apply_layer_contacts`, `layer_force`): two bodies
in contact are rigid with an elastic layer between them (the elastic-foundation model).
The normal force is `k'' V` (overlap volume `V` of the two convex shapes, exact by
polytope clipping) at the overlap's centroid, `k'' = 1 / (h_a / E'_a + h_b / E'_b)`:
between two faces this is the stiffness of the joint they shared (`E A / L`, `E I / L`),
so a cracked joint and a contact behave alike. Friction is the shear layer
(`1 / (h_a / G_a + h_b / G_b)` per area) anchored to the contact, with Coulomb's cap by
return mapping (tangential and torsional), so a stuck contact holds with no slip rate.
Dashpots from the coefficient of restitution, uncapped: the contact substep resolves the
damped oscillator. The contact normal is the area-weighted outward normal of the other
body's surface inside the overlap (continuous in the geometry); the overlap's new faces
are chained from the clipped faces' edges (no sorting, tolerance or reference axis).

Found and fixed while proving it: a least-penetration (SAT) normal flips between axes
on round-off where aligned edges overlap (a 90-degree jump in force direction); a
polygon built by sorting in an axis-dependent basis with a 1e-12 tolerance made results
depend on the orientation of the world axes at 1e-8.

**Evidence** (principles, with `scaled_step_bound` also on): all 8 pass. Coulomb
acceleration exact (0.00 %) at three masses and two substeps; sticking creep 1e-20 to
1e-12 m/s; stopping distance 0.1019 m in 0.204 s (Coulomb 0.1019 m, 0.204 s); resting
sink 3.00e-8 m (elastic 3.01e-8); rescaling every unit bit-exact with fracture; uniform
velocity and rotation within round-off with fracture.

**Energy ledger** (fixed while proving it). The contact dissipation is booked from the
work the contact forces actually did on the motion the integrator produced (force at
the start of the substep, velocity at its end) less the change of what the contacts
store, plus the energy joints hand to contacts at a split (`ContactLedger::work`,
`received`). The force law's own first-order tally is kept as a diagnostic
(`modelled`, and per kind of contact in `kinds`). `conservation.rs` checks the second
law for contacts: their net dissipation never falls frame to frame.

Bugs found by the ledger and fixed:

* Friction state stored as a slip that the current stiffness multiplies: when the
  contact area shrank (stiffness to near zero) the slip held the dashpot's share divided
  by that stiffness, so the torsional "twist" reached 160 rad and the booked
  dissipation 6.8e8 J on a 4e4 J impact. The shear layer now stores its elastic force
  (Mindlin-Deresiewicz incremental form). New contact area enters unstrained, and area
  leaving takes its share of the force, so `F^2 / 2k` never grows without work. While
  sliding, Coulomb's slider is in series with the layer (spring and dashpot in
  parallel): `k e + c de/dt = mu F_n`, by backward Euler. This is continuous with
  sticking at the threshold; setting the spring to the capped total was not, and let
  round-off decide which side a contact landed on.
* Sphere impactors: every chunk the ball touched got a full spherical cap against the
  plane at its nearest point, so caps double-counted across chunk edges and the force
  was not the gradient of any energy. It put about 2.1e3 J (5 % of the initial energy)
  into a 40 m/s impact. Replaced by the exact ball-polytope overlap in closed form
  (`LayerContact::with_ball`, divergence theorem over the faces cut by the ball's
  discs). It is cancellation-free for shallow caps: a cap of depth `1e-6 r` is exact to
  round-off. A centre on a face, edge or corner takes the principal value plus the
  singular point's solid-angle share. An empty or full disc region is decided exactly:
  round-off "ghost" contacts with arbitrary normals once gave a 7e7 N kick.
  Tests: equal to the cap on a face; half, quarter and eighth ball at a face, edge and
  corner; additive over a split box; brute-force column integration to 3e-7.

Evidence (free wall hit by a 50 kg ball, both switches as stated; residual =
(mechanical + dissipated - initial) / initial at the end; Courant safety 0.5 / 0.25 /
0.125):

| case | `layer_contact` | `+ scaled_step_bound` | established |
|---|---|---|---|
| 5 m/s, elastic | -4.8e-3 / -2.4e-3 / -1.2e-3 | -8.4e-3 / -4.2e-3 / -2.1e-3 | +6.6e-4 / +3.3e-4 / +1.6e-4 |
| 40 m/s, fracture | -8.1e-3 / -3.4e-3 / -1.3e-3 | -1.6e-2 / -6.9e-3 / -2.8e-3 | -3.5e-3 / -1.3e-2 / -2.2e-2 |

The new contact's residual is a loss and halves with the step (first-order integration
error, no energy created). The established method's residual with fracture grows as
the step is refined (-0.01 % at Courant 1 to -2.3 % at 1/16): its ledger is inconsistent.

**Every elastic layer force is the exact gradient of its energy.** Found by the
per-kind ledger (`ContactLedger::kinds`: the force law's own tally against the work
done, which must converge together with the substep for a conservative law):

* *Pairs* (chunk on chunk): the area-weighted normal tilts by `delta L / A` where a
  face overhangs the other's edge, a tangential force `k delta^2 L` where the energy
  `k delta^2 A_overlap / 2` has gradient `k delta^2 L / 2`. It created 1.8e3 J on the
  40 m/s impact, whatever the substep. Now (`pair_elastic`, `polytope::field_cells`)
  the energy is `(E_a + E_b) / 2`, each `E_x = sum_f k''_f int s dV` over the overlap's
  cells nearest each face `f` of body `x` (`s` the depth below it). By the divergence
  theorem its exact gradient is `k''_f V_f n_f` per cell at the cell's centroid, plus
  the interface integrals where neighbouring cells differ in stiffness, plus the torque
  of the stiffness turning with the bodies. The half-thickness along a direction is
  the smooth `sqrt(sum (n . e_k)^2 half_k^2)`, not the support function, whose kink at
  alignment makes a face-to-face torque jump as it passes through alignment.
* *Balls* (sphere impactors): the same, with the chunk's nearest-face cells cut from
  the ball exactly (`polytope::ball_cells`); interface integrals from the closed-form
  moments of a polygon cut by a disc.
* *Ground*: the torque of the chunk's thickness turning with it.
* *Indentation* (crushed material from a split joint) stays while the pair overlaps
  and is forgotten when they part; lowering it while they touched created energy.

Tests: finite differences of the energy against the force and moment
(`world::tests`), in face, edge and corner configurations of cubes, a non-cubic chunk
and a ball, and over 500 random relative poses (worst 3.5e-6, the differences' own
error). A frictionless, perfectly elastic collision of two free cubes conserves energy
(`an_elastic_collision_of_two_cubes_conserves_energy`): `|dE/E|` 1.4e-3 / 2.2e-5 (face
overhanging an edge), 9e-10 / 2e-12 (tilted onto an edge), 1.1e-3 / 3.0e-4 (edge on
edge) at Courant 0.5 / 0.125.

**Robust polytope clipping.** The planes the cells are cut by are structurally
degenerate: a bisector passes through the edge of the two faces it bisects, and
neighbouring chunks of a grid have coplanar faces. Three fixes, each found by a failing
gradient test:
* a crossing point is interpolated from the inside end, so the two faces sharing an
  edge make it bit for bit and the result stays watertight (one ulp apart, a later
  clip through that edge lost a face: 2/3 of the volume);
* the new face is the convex hull of the crossing points in its plane, not a chain of
  segments (round-off dropped or duplicated a segment; a duplicated vertex then gave
  the disc integrals a zero-length basis edge and an energy jump of 4e4 J on a 1 J
  contact);
* the in-plane basis of a face comes from its normal alone.

**Frame invariance with fracture** (`a_rigid_rotation_of_the_scene_rotates_the_result`)
now holds to round-off: 6.5e-11 m against an envelope of 3.4e-9 m (1.3e-4 m before;
4e-2 m at the start of this work). Found by tracing where the rotated run first
departed: (1) the contact area and normal came from which body owns a face of the
overlap, which round-off decides where the bodies' faces are coplanar; it now comes
from the overlap alone (its shadow along the normal). (2) The torsional lever arm's
quadrature fanned each face from its first vertex, so its error depended on the
orientation; it now fans from the face's centroid.

**Split handover.** A joint broken by a split hands its elastic compression to the
contact (the indentation at which the contact's force equals it), and now also its
shear and torsion to the contact's friction (same stiffness, `k''_t A` and `k''_t J`
being the joint's `G A / L` and `G J / L`). Coulomb's cap applies at the contact's
next step, where any excess slips. It is never more than the joint stored; the rest is
`split_release`. The elastic force (no dashpot) is what is handed over. On the 40 m/s
impact the energy released at splits halves (87 J to 46 J of about 350 J handed over).

**Cost.** Chunk-pair contacts are evaluated in parallel from 16 pairs on
(`par::map_into_heavy`, item order kept: bit-identical for any thread count), and cell
cuts a convex region's vertices show to be idle are skipped (exact). The breaching
pressure panel is still ~12x slower than with the established contact: ~870 pairs per
substep after it breaks, ~45 us of CPU each, mostly allocation in polytope clipping (a
flat-arena polytope is the next lever).

**Suite** (`cargo test --release -p stress-ref --no-fail-fast`; `scripts/test_fast.sh`
skips the panel, hull-pack and showcase runs: ~2 min instead of ~25 with the switch):
default methods pass everything but the principles (by design). With `layer_contact`
alone, unit rescaling also needs `scaled_step_bound` (each fixes a unit dependence), and
the implicit free-vibration test needs `frame_contact_step`: energy is conserved to 7
digits, but the global contact step (with the dashpot factor, though nothing can touch)
shortens the substep and the sampled peaks fall 1.5 % lower. With both switches:
principles 8/8, showcases 7/7; open: elastic impact at the larger step (1.07 %),
convergence, engine-coupled, mass scaling, and the b9 panel (below).

**Open, b9 panel (over-strong arching):** the breaching panel breaks 300 bonds into 16
fragments and barely moves (established: 60 bonds, 4 fragments, matching OpenCourant).
The contact has the joint's stiffness but not its compressive strength: hinges between
fragments never crush. Planned: the joint's crushing criterion on the contact, in closed
form (peak layer pressure `max_f k''_f d_f`, the indentation ratcheted to
`max_f (d_f - f_c / k''_f)`).

**Rocking damping and the exact shear layer** (this round, DECISIONS.md 10-12): the
dashpot spreads over the area like the springs, so relative spin about in-plane axes is
resisted (`-(c_n / A) Q w`); before, a resting heavy block rocked forever. The friction
layer is an exact last-in-first-out stack of entry cohorts; the uniform-strain rule it
replaces made a near-grazing contact edge creep (8e-14 m/s) as round-off flickered its
area. Both are proven by exact targets (`tests/principles.rs`: rest at the exact
two-field equilibrium, sticking at the exact elastic offset with no creep, the exact
discrete Coulomb stop).

**Sliver contacts (bug, fixed):** the contact area's second moment was formed about the
world origin and shifted to the contact point; for a thin sliver the shift cancelled to
the size of the value, the polar moment came out negative and the torsional dashpot's
`sqrt` gave NaN (40 m/s fracture at Courant safety 1/64). Moments are now integrated
about the point and the parallel-axis shift only adds a positive term (DECISIONS.md 28).

**Open, energy closure with fracture:** the balance converges at first order to about
-1.3e-3 e0, not zero; the loss accrues at a fixed rate while contacts push loose cracked
chunks, the signature of the small-strain floating-frame coupling that also keeps
angular momentum from being conserved (DECISIONS.md 26-27). Both are exact targets now
and fail until that coupling is made exact.

**Status:** default on. Catalogue A/B against the established methods: 87/91 gated
metrics both ways, the same four known b5 gaps failing (DECISIONS.md 42).

### `scaled_step_bound` (fix, step change)

**Replaces** the Gershgorin bound of `chunk_stable_dts` / `chunk_frequencies`, whose row
sums add stiffness entries in N/m and N/rad (the kinematic row's L1 norm adds a
dimensionless direction and a lever arm in metres): the critical substep changed by
1.30-1.43x when the unit of length was halved.

**Method:** Gershgorin rows of the mass-normalised `M^-1/2 K M^-1/2` (the same
eigenvalues as `M^-1 K`), each entry divided by the square roots of its degrees of
freedom's inertias: unit-consistent and still an upper bound. Rotations use the smallest
principal moment (conservative).

**Evidence:** substep ratio exactly 1 under rescaling of length, time and mass on the
small impact, the frame and the masonry wall; the "length constant at chunk removal"
(frame b8 diverging when its column is removed) was this bound, recomputed for the new
clusters. The bound is 1.8-3.4x larger on those scenes (the old one was conservative in
metres).

Stability harness (`examples/stability.rs`: every catalogue scene, perturbed at rest,
20,000 substeps at fractions 0.9 to 2 of the bound): the scaled bound is stable at
0.9, 1, 1.1 and 1.5 on every scene, and first blows up at 2 (b2, b5, b6, b9, s_blast,
s_supports): valid and tight within 1.5-2x. The established bound never blows up, even
at 2x: it leaves more than 2x on the table. (The harness first measured nothing: the
bonds' stored energy is their last evaluation's, zero before the first step.)

**Status:** catalogue A/B and test suite running.

### `exact_frame_transfer` (fix)

**Replaces** the linearised transfer of a new fragment's hidden rigid motion into its
frame, and the split's re-expression of chunk velocities with lever arms that left out
the hidden displacement: both moved chunks or changed their velocities at every split
(a deformed beam's split: 1.0e-2 in angular momentum, 5.1e-2 J created).

**Method:** at a split, every chunk's world position, orientation, velocity and spin are
re-expressed exactly in the child's frame, which is then re-centred (hidden mean
displacement and velocity zero, the frame taking them, lever arms included). Per
substep the fold stays linear (O(dt), no cancellation).

**Evidence:** `a_split_of_a_deformed_body_keeps_its_motion` (momentum, angular momentum,
kinetic energy to rounding); angular momentum changes only by the bonds' small-strain
imbalance, converging to zero with the substep. **Status:** default on.

### Rigid drift fold about its centre (fix, no switch, 2026-10-12)

**Replaces** the frame placement of the linear fold (`fold_rigid_drift`, every free
cluster every substep; world.slang `drift_rigid`). It turned the frame by the hidden
field's mean rotation `phi` (exactly, `exp(phi)`) but moved it by `R (t - phi x com)`:
a turn about the body origin compensated to first order, which moved every chunk by
`R (1/2 phi x (phi x c))`, `c` its distance from the body origin. That origin is an
arbitrary choice of the scene (a scene pack placed by moving its chunk centres keeps it
at the world origin), and no observable may depend on it. Per substep `phi ~ 1e-9`
hid it; a split child under the established methods (no `exact_frame_transfer`) keeps
its finite hidden rotation, and its first fold moved it at once: fractured-bridge, a
one-chunk child with 0.09 rad at 23.6 m from the origin jumped 9.50 cm (predicted
9.55 cm) into its neighbour in one substep, and the pair contact fired the overlap as
energy (+1.3 MJ, 35 m/s here; +2.2 MJ, 450 m/s on the GPU).

**Method:** the frame turns about the fold's centre: `R (t - (exp(phi) - I) com)`, the
difference by Rodrigues (`turn_difference`, no cancellation). The rigid part of the
fold is then exact (a one-chunk cluster does not move); what remains is the hidden
correction's first order about `com`, as before. `exact_frame_transfer` re-expresses
every chunk from its recorded world pose, so it was already origin-invariant; it stays
the exact path for multi-chunk children.

**Evidence:** `rigid_drift_fold_is_independent_of_the_body_origin` (a one-chunk and a
3-chunk free cluster, the origin beside them or 25 m away: same world poses to f64
rounding; failed by 1.078e-1 m), stress-gpu `frame_fold.rs` (the GPU's substep against
the reference's), stress-physx `bridge_stays_within_its_energy_{gpu,reference}`
(kinetic plus potential energy never above the start's; failed at frames 49 and 51).
The full stress-ref suite passes unchanged (goldens, showcases, principles). Oracle
shift (`scripts/verify_identical.py`, single-thread cells, both method sets): 61 of 112
cells identical; the 51 that differ are the fracture and contact scenes, chaotic under
any perturbation of rounding at a split: b1_bond_shear 1e-9 relative; b9_panel_high
centre displacement within 0.2 % of its range; s_car_brick car velocity within 0.8 %;
b7_masonry_v04 (established) 181 -> 147 broken bonds; b8_frame_sudden (established)
column force up to 86 % of its range apart. No golden needed updating. **Status:**
adopted (a correctness fix: an observable must not depend on the body origin).

### GPU static solve certifies only a finite, lowered residual (fix, GPU only, 2026-10-12)

world.slang `island_statics` accepted an iterate at f32's floor (STATICS_ULPS of the
iterate's own bond loads) once its residual stopped halving. A diverging secant
iteration (a free fragment without equilibrium: blocks touching only through fully
damaged, contact-only bonds) met that floor at overflow (inf <= inf) and before it (a
finite 5e15 under a floor its 1e21 bond loads raised): adaptive settles came back
"converged" with |u| 1e11 m and split into 1e7-1e9 m/s fragments (house-1story rammed).
It now matches statics.rs: a finite residual, and the floor only for an iterate no larger
than the solve's first; a non-finite residual ends the solve unconverged.
`statics_vs_reference::free_cluster_settle_matches_the_reference` (two captured
settles). **Status:** adopted.

### `contact_crushing` (model change)

**Replaces** contacts of unlimited compressive strength: the contact that takes over a
broken joint had the joint's stiffness but not its strength, so hinges between fragments
never crushed (b9: fragments rebounding against the pulse).

**Method:** the joint's crushing law on the contact: strength `f_c (1 - c / d_u)`,
`d_u = 2 G_c / f_c` (energy-limited like the joint's own crushing; a perfectly plastic
crush dissipated 3.1 kJ of 6.7 kJ on b9), the indentation ratcheting where the layer's
peak pressure `k''_f s` exceeds it.

**Evidence:** b9 high against OpenCourant: fragment speed 2.79 vs 3.45 m/s (gate 30 %),
peak centre displacement 9 % (established 22 %). **Status:** default on.

**Joint law ledger (fix, all runs):** the dissipation booked by a bond is exactly the
drop of its stored energy caused by its internal-variable update; the law's own accounts
had booked up to twice the energy released (DECISIONS.md 33). This closed the fracture
energy balance with `layer_contact`.

### `tangent_statics` (fix)

**Replaces** the verdict of a failed secant solve. The static solve iterates with the
joints' secant stiffness. That fixed-point iteration converges only linearly and can
diverge (b9's residual climbs to 1e13, the villa's to inf). Where it stops short at its
60-step limit, `equilibrate_or_keep` silently keeps the cluster's last state, even where
an equilibrium exists:
- s_arch_keystone_removed in quasi_static: all 600 frames unconverged;
- b9 in quasi_static: the panel never fails;
- the villa (scene pack) under self-weight: no equilibrium found.

**Method:**
- The secant iteration runs unchanged. Every solve it converges is bit-identical.
- Where it fails, the cluster is solved again from the same start by a globalised
  Newton:
  - each step uses the joints' consistent tangent (`JointModel::tangent`, as the
    implicit solve), else the secant (an upper bound of the nonsmooth patch's
    tangents);
  - the step is halved until the residual falls by the Armijo fraction (1e-4 alpha);
  - where neither direction lowers it, the solve stops unconverged and the cluster
    keeps its state as before.
- A solve is accepted only by the same residual test, so every accepted state is an
  equilibrium.
- A secant residual that turns non-finite ends the secant at once, since every later
  iterate is NaN. Its state is discarded for the re-solve anyway, and the gate shows no
  observation changes (182/182 against the build without the early stop).

**Rejected (both measured, b9 quasi_static):**
- **Tangent first in every solve** (6126ca847..4184fdffc). It changes solves the secant
  already converges, at round-off. Those differences decide whether plain tangent
  Newton converges on a nonsmooth contact: at t = 0.002 (cascade pass 2) it converges
  in 14 steps from one start and cycles for 60 from a start 5.6e-15 m away. So b9's
  breach hung on round-off. It also changed default dynamic runs chaotically
  (house_settlement broken bonds 161 -> 214).
- **Fallback re-solve without the line search:** b9 never breaches (20 broken bonds),
  the same round-off failure.

With the line search, both starts reach the same equilibrium (umax 1.488699e-2 m) and
both runs breach.

**Evidence (against cd112933c):**

`verify_identical.py`: 176/182 cells identical. Every cell in the scenes' own modes is
identical, at 1 thread and at all threads. The six differing cells are all forced
quasi_static, at 1 thread:

| Cell | Unconverged frames | Outcome |
|---|---|---|
| s_arch_keystone_removed default | 600 -> 1 | 6 bonds break, 8 fragments |
| b9_panel_high default | 59 -> 16 | breaches (168 bonds, 36 fragments) |
| b7_masonry_v15 default | 6 -> 4 | broken bonds 66 -> 191 |
| b2_cantilever, both method sets | (direct solve, above) | (direct solve, above) |
| s_arch_keystone_removed established | (direct solve, above) | 3e-21 m |

The three default quasi_static cells give identical observations at 16 threads.

**Quasi_static solves converged:**

| Scene | Secant | Re-solve |
|---|---|---|
| s_arch_keystone_removed | 109/1309 | 2087/2089 |
| b7_masonry_v15 | 17/34 | 2813/3030 |
| b9_panel_high | 3/62 | 714/739 |

b2_cantilever is unchanged (121/121).

**Catalogue A/B** (`STRESS_METHODS=-tangent_statics ... ab --feature tangent_statics=on`):
- **scenes' own modes:** 87/91 gated metrics both ways;
- **quasi_static forced:** 44/91 -> 53/91, no regression. The b5_wall_impact v02 and v10
  failure mode (push_over) and hole area, and b9_panel_high's breach, fragment speed,
  first failure and peak displacement now pass against OpenCourant.

**Cost** follows the fracture the secant missed: quasi_static A/B wall 190 -> 566 s
(b5_wall_impact_v02 55 -> 245 s, b7_masonry_v15 2.8 -> 73 s).

**Test:** `hulls::scene_packs_import_and_stand` expected the villa to have no static
equilibrium. With the re-solve every frame converges, with 5 of 2464 joints cracked and
one fragment, the criteria the other three assets meet. The test now expects the villa to
stand, except with the switch off.

**Re-solve residual scale:** the re-solve measures its residual against the larger of
the load and the starting out-of-balance force, as the implicit solve does. A free
fragment under rigid loads alone (falling, or accelerating with its cluster) has a
deforming load that is round-off once the rigid modes are projected out (1e-15 N in
b5_wall_impact_v40). Its equilibrium is its bonds relaxed, and a residual relative to
that round-off load could never accept it.

Every solve the secant converges is still unchanged. Against the build before this
change (310a2c14f), the gate differs only in the three default quasi_static cells
already changed above:

| Cell | Unconverged frames | Outcome | CPU |
|---|---|---|---|
| b7_masonry_v15 | 4 -> 2 | 191 -> 190 broken bonds | 77.1 -> 10.6 s |
| b9_panel_high | 16 -> 6 | still breaches; 168 -> 154 broken bonds | 46.3 -> 48.0 s |
| s_arch_keystone_removed | 1 -> 0 | | 1.7 -> 2.2 s |

The quasi_static A/B stays at 53/91 with no regression. b9's fragment speed moves from
25.6 % to 18.7 % off OpenCourant, and its peak displacement from 11.2 % to 11.9 %.

**Secant steps before the re-solve:** the secant iteration now gets 20 steps
(`SECANT_STEPS`: the PCG's 10, then 10 direct) before the re-solve, not 60. Traced on b9
quasi_static, the secant solves that converge almost all do so within 12 steps (1068 at
0, 52 at 1-5, 50 at 10-12, then single solves at 19 and 52). The 580 that fail spend
34,800 of the 39,336 Newton steps running out the 60.

Where the re-solve fails too, the secant resumes at step 20, its remaining steps
exactly as before (no history is committed in between). So every solve it converged
still converges. The only difference: a solve the secant would converge after step 20
takes the re-solve's equilibrium instead. Both pass the same residual test; the first
such b9 solve gives the same displacement to 7 digits, with residual 5.6e-14 against
9.0e-11.

Gate against 7499bee20: only the three default quasi_static cells change.

| Cell | CPU | Change |
|---|---|---|
| b9_panel_high | 48.2 -> 28.1 s | same broken bonds; unconverged frames 6 -> 15 |
| s_arch_keystone_removed | 2.1 -> 1.15 s | broken bonds 6 -> 4 |
| b7_masonry_v15 | 11.0 -> 9.6 s | broken bonds 190 -> 191 |

The extra b9 unconverged frames come from the run's divergence after that first
solve. They are 7- and 9-chunk free fragments under 1e-3 N loads, the open class below;
the uncapped run's failures are the same class. Quasi_static A/B: 53/91 with the same
nine flips and no regression; B wall 505 -> 394 s. Villa (3 quasi_static frames):
1079 -> 706 s, observation identical.

**Equilibrium at working precision (re-solve only).** b5_wall_impact_v40 in quasi_static
failed 115 of 308 solves, all on free fragments. Two kinds:

- **At rest within round-off.** Bond forces of about 1e-15 N under a 1e-15 N load: no
  residual relative to a load of round-off can certify them.
  - The re-solve now also accepts an iterate whose every residual component is within
    the rounding error of computing it (`residual_floor`, derived per operation):
    - a bond's relative displacement in local axes errs by at most `gamma_7 |D|`, with
      `D = |u_a| + |u_b| + |th_a| x |r_a| + |th_b| x |r_b|`: large rigid motions cancel;
    - its rotation by `gamma_4`;
    - the force by the stiffness times that, plus `gamma_20 |q|` for the joint law;
    - loads and moments by `gamma_3` plus the propagated errors;
    - the sums over a chunk's bonds by `gamma_(n+1)`;
    - a free cluster's rigid-mode projection by `gamma_(N+10)` of the absolute sum.
  - Such an iterate is an equilibrium to working precision; no f64 iterate does
    better. Scale: a fragment 3.6 m from home on 1e10 N/m bonds has a floor of about
    3e-5 N, 3e-7 m/s^2 on a 100 kg chunk.
- **Converging linearly, cut off.** A traced 13-chunk fragment falls from 1 to 5.7e-9
  at about x0.7 per step, against 1e-10.
  - A finite-difference check of every joint at the last iterate finds them heavily
    cracked (D 0.65-1) and opened about 1e-13 m: the no-tension patch's kink, faces just
    touching. The force law is not differentiable there, so Newton goes linear.
  - The re-solve's step limit is now 3 x `max_newton` (`RESOLVE_STEPS_FACTOR`). Where
    no descent remains it still stops at once, so a mechanism costs nothing extra.

Effect:
- b5_v40 quasi_static: converged 193/308 -> 372/396 (84, then 158, at round-off); the
  run diverges after the first changed solve, so solve counts differ. Newton iterations
  20,741 -> 12,157; push-over and hole area unchanged.
- Gate against 41a88904d:
  - b9 quasi_static: unconverged frames 15 -> 8, displacements within 7e-14 m;
  - b7_v15 quasi_static: one more solve converges, then the run diverges (broken bonds
    191 -> 182), CPU 9.2 -> 8.1 s;
  - all other cells identical.
- Quasi_static A/B: 53/91, same flips, no regression.

`equilibrate_or_keep` still keeps the previous state for what fails: 6 % of b5_v40's
solves, fragments converging linearly past even the larger limit. Two causes were
tested and ruled out on b5_v40 quasi_static:
- **The kink's Jacobian element** (semismooth Newton). Counting a contact spring
  active up to an opening of 1e-12, 1e-10 or 1e-8 m, rather than by the opening's sign,
  converged fewer solves (321/413, 277/396, 1001/1435, against 372/396) in 2.5-8x the
  Newton iterations.
- **The tangent's positivity floor.** At 1e-9, 1e-4 and 1e-3 x k (against 1e-6),
  96.1 %, 94.2 % and 93.5 % converged (against 93.9 %), with no trend.
**The friction cap's dependence on the normal force (fixed).** A sliding patch carries
`mu nc` in shear and `mu nc r_f` in torsion, so those rows depend on the normal and
rocking displacements through `nc`. `JointModel::tangent_full` adds that coupling
(`nc`'s derivative from the compressed springs, or the exact patch's region). It is not
symmetric, so the re-solve's tangent step for a cluster of at most 64 chunks is now a
dense LU of the full tangent, with a free cluster's rigid modes fixed by the
constraints `project_displacement` imposes (`[K G^T; G 0]`).

Test `full_tangent_is_the_derivative_of_a_sliding_cracked_joint`: central differences
in the linear columns match within round-off, and the symmetric tangent fails them.

Effect:
- b9 quasi_static: every solve converges (2763/2763, against 1765/1774).
- b5_v40: 375/396 (against 372/396).
- Gate against 995bd44a9: the three default quasi_static cells. b9's unconverged frames
  go 8 -> 0, b7's 3 -> 2, and the arch's collapse diverges (broken bonds 4 -> 8).
- Quasi_static A/B: 53/91, no regression (b9 fragment speed 18.6 % off, peak
  displacement 11.6 %).
- Suite 125/125.

The dense limit is 128 chunks (`DENSE_MAX_CHUNKS`). At 64, the 102- and 116-chunk b5
fragments kept the symmetric tangent and relaxed slowly. At 128, b5_v40 quasi_static
converges 308/314 in 14 s (against 375/396 in 22 s), and b9 is unchanged (2763/2763).
At 256, b9's run diverges onto a path with 7 failures. Gate against 178dca904: only b7_v15
quasi_static changes. Its larger clusters now take the consistent tangent, and the
cascade follows other equilibria: broken bonds 190 -> 346, fragments 44 -> 90 (its
explicit run: 82), unconverged frames unchanged at 2. No quasi_static reference exists for
it. A/B 53/91, suite 125/125.

What still fails in b5 is the no-equilibrium kind. A two-chunk fragment on one cracked
joint cannot carry its load: the re-solve finds no descent at step 0, with 1e-11 N of
internal force against 8-44 N of load, or the joint saturates (564 N against 598 N).
Keeping the previous state is the right verdict there.

**Rejected (exact, measured):** computing a cracked joint's spring patch once per
`evaluate`. It is the second hottest function in a sampled b5_v10 run, and the frozen
energy, the crushing probe and the final force evaluate it at the same crush. The
gate is 182/182, but CPU is 16.3 -> 16.2 s on b5_v10 and 9.6 -> 9.6 s on b7_v15: most
damaged joints never take the probe paths. Not worth the code.

**Tangent consistency:** `JointModel::tangent` against central differences of
`evaluate`:
- **Rebar (fixed):** the tangent used the bars' elastic stiffness even where
  `return_map` caps the force (yielded in tension, slipped in dowel), whose derivative
  there is zero. Now the derivative of the return map. Bars exist only in imported
  scene packs (reinforced concrete), so the catalogue is unchanged (182/182 against
  15853356b). The villa's outcome agrees to 1e-12 and its time is unchanged. Test:
  `rebar_tangent_is_the_derivative_of_the_force`; the old tangent fails it (6.854e10
  against 6.763e10 for a yielded dowel).
- **Open:** sliding friction, `cap = mu nc`. The sliding shear and torsion forces scale
  with the patch's normal force, a coupling the tangent omits. It is non-symmetric, so
  neither the PCG nor the Cholesky can take it; a non-symmetric solver (GMRES, or LU of
  the band) would. The tangent's positivity floor `1e-6 k` keeps the matrix definite.
  It is not in the force, and makes Newton linear wherever it dominates (fully opened
  joints).

**Finding corrected:** "Finding for the solver" below concluded that b9's failing solves
have no equilibrium (a mechanism), because a line search on the secant held the
residual at 4 to 75. The tangent re-solve finds equilibria for most of them. The secant
direction, not the physics, was the obstacle.

**Status:** default on. `STRESS_METHODS=-tangent_statics` restores the secant verdict.

### `patch_after_damage` (model change, switch off)

**Replaces** evaluating the cracked-joint contact patch on every joint: today an
uncracked joint (damage 0) runs the patch's 36 springs and commits friction slip on an
interface that does not exist until the joint cracks.

**Why now:** the GPU's cold-bond path (REALTIME_ROADMAP.md, Part A6) can be exact only if
an intact joint carries no patch state.

**A/B** (`stress-ref ab scenes golden --feature patch_after_damage=on`, 2026-10-11, base
`c84658baf` plus the pack-import fix):
- Gated metrics 87/91 both ways (the same four long-standing failures), no regression.
- Elastic benchmarks unchanged to the printed digits.
- Catalogue wall time 178 -> 156 s.
- Broken bonds: b5 v02 +0.8 %, v40 +2 %; b7 v04 +4 % (fragments 25 -> 29), v15 -0.9 %;
  **b8_frame_sudden 23 -> 11 (fragments 22 -> 11)**. b8's sudden collapse is chaotic: the
  GPU gate's reference ensemble already admits 17-29 broken bonds there.

**Status:** evidence recorded. Adoption is a model change and needs a person's review
(the b8 difference judged against its ensemble spread); then `scripts/fidelity.py`.

### Engine shapes follow the chunks' deformation (coupling fix, switch, default on pending review, 2026-10-24)

**Switch.** `World::engine_shapes_deformed` (`GpuWorld::set_engine_shapes_deformed`;
`STRESS_ENGINE_SHAPES=undeformed` turns it off), in the host-coupled frame and the
GPU-resident one. Coupled runs only: the standalone oracle and the GPU accuracy gate
build no engine shapes.

**What it replaces.** The coupling built a cluster's engine shapes at its chunks'
undeformed centres, once per chunk set. Now each shape sits where the stress solve has
its chunk (hidden displacement `u` and rotation `th`, as `World::chunk_box`), refreshed
every frame (`World::shape_poses`, carried as `EngineBody::poses`). The shapes
themselves stay undeformed, and every body is built from them (as the GPU's split
builds its bodies from the undeformed part table). A body whose shapes moved keeps
its identity: rubble places its colliders in place (`RubbleEngine::move_shapes`),
keeping its contacts and their warm starts. Its mass properties stay the build's
(small strains). A settled structure's poses do not change, so nothing is uploaded
for it.

**GPU-resident frame.** One kernel before the engine steps (`stress-gpu` `shapes.rs`,
`shaders/shapes.slang`): for each engine collider of a chunk, its chunk from the
coupling's collider table, that chunk's `u` and `th` from the state the last static
solve left, and the collider's undeformed placement (`ResidentEngine::collider_rest`,
uploaded only for colliders added since). It writes rubble's collider buffer in
place; nothing is read back. A GPU split's fragments are moved from the frame after
the host follows the split (README "Known gaps").

**Why (b8_frame_sudden, coupled).** rubble's broadphase used to drop dynamic-static
pairs whose static had the lower collider index. b8's anchored frame is created first,
so its freed pieces fell through it. Keeping those pairs (PR #52) exposed this:
- 11-40 broken bonds at 0.25 s over the gate's input perturbations (median 29),
  against the oracle's 7-14.
- Runs with and without those pairs break the same bonds until 84 ms.
- Then a freed 2.7 m top-beam piece sits between the frame's anchored halves. In the
  stress solve it has a gap at both ends, widening from 0.2 to 0.9 mm in 4 ms as the
  frame springs apart after the column loss. In rubble, with undeformed shapes, it was
  flush against both faces.
- Turning (0.08 rad/s), it wedged: equal and opposite end impulses of 11, 84, then
  121 N s per 1 ms frame, about its angular momentum over the lever arm. It broke in
  three, and the cascade followed.

**Evidence.**

| b8, 7 runs over the gate's perturbations | broken bonds |
|---|---|
| oracle ensemble | 7-14 |
| undeformed shapes (before) | 11-40, median 29 |
| without the lower-index pairs (before PR #52) | 10-14 |
| this switch | 13-17, median 14 |

- *Ruled out.*
  - Dropping the engine's contact loads from the stress solve (on pieces, on
    anchored clusters, or both): median 24-28.
  - rubble's friction at 0: median 34.
  - Shapes deformed only when a body is built (at the split), not refreshed: median 28.
  - Rebuilding the anchored bodies every frame, undeformed: median 13, but two runs
    still cascade (37, 30). The rebuild resets the contacts' warm starts, which acts
    like compliance.
- *Isolated cases* match the oracle either way: a beam landing on an anchored block,
  and a beam freed between two undeformed anchored walls (`stress-rubble` tests
  `integrated::a_beam_*`).
- *Resident against host-coupled.* `resident_gpu::resident_frame_moves_shapes_as_the_fast_frame`:
  with the switch on, the rammed house's damage (which differs from its undeformed
  run's) and the cantilever's split and fall match bit for bit; the ram's speed
  agrees to rounding (f32 on the GPU, f64 on the host).
- *Suite.* `stress-rubble` (every test binary) and `stress-gpu` pass with the switch
  on.

**Cost at scale.** `stress-rubble --example shape_deform_cost`, Apple M3 Max: 10,000
bodies of 4 chunks (40,000 colliders), 40 frames, quasi-static. Two scenes: free beams
dropped on the ground (every body active, the engine stepping; resting free beams
barely deform, so no pose changes), and cantilevers fixed at one end (every body's
chunks displaced, every pose new each frame while they settle).

| per frame, median | free beams | cantilevers |
|---|---|---|
| resident: shape pass, submit to done (`STRESS_RESIDENT_PHASES`) | 0.30 ms | 0.27 ms |
| resident: its host encode | 0.06 ms | 0.06 ms |
| host-coupled: body sync, undeformed | 2.05 ms | 2.48 ms |
| host-coupled: body sync, deformed | 5.47 ms | 5.68 ms |

- The resident pass does the same work whatever moves (one thread per collider) and
  reads nothing back. Its time includes the submission's round trip, so it bounds the
  kernel from above. Resident frame totals (13-19 ms free, 4-7 ms cantilevers) vary
  more between repeats than the pass costs, so they cannot resolve it.
- The host-coupled frame pays about 3.3 ms on the host at this size: each cluster's
  poses are computed and compared every frame (O(chunks)), uploads only for bodies
  whose poses changed. Lossless options if that matters: skip clusters whose chunk
  state did not change, or the GPU pass in that frame too.
- A resident frame's split write-back keeps the poses last given
  (`World::engine_poses_held`): the host does not recompute them there.

**Status:** default on in both frames, pending review (a coupling model change).

## Runtime (bit-exact, no switch)

Speedups that change no result take no switch. Each must reproduce the established
build's observations byte for byte (`wall_seconds` aside). The bar does not lean on the
physics tests (some were being reworked at the time), so it holds whatever they say.

**Gate:** `scripts/verify_identical.py --base OLD --cand NEW --out DIR`. It covers
every scene (catalogue and showcases):
- under the default methods and under `layer_contact,scaled_step_bound`;
- at `RAYON_NUM_THREADS=1` and with every thread;
- on a subset, with each `sim.solve_mode` forced (adaptive, quasi_static, implicit).

That is 182 cells. The baseline checked against itself is 182/182 identical, so the
gate is deterministic. `--bench --repeat K` gives the minimum CPU time per scene at one
thread.

**Profile:** `stress-ref run SCENE --profile` (or `STRESS_PROFILE=1`) prints stage
times and work counters to stderr; nothing goes into the observation. It covers the
substep stages, contact broadphase and pair stages, bond responses, splits, statics,
plus counters for bonds evaluated, pair candidates and overlaps, polytope clips,
field-cell calls, splits, and which limit set each frame's step.

| Commit | Change | Gate |
|---|---|---|
| 53895036b | Profiling and gate. Pool dispatch only with more than one worker and from 64 heavy items (a frame collapse spent 11 s of system time waking 16 workers for ~20 pairs). Clip side buffer | 182/182 |
| 19e5cf7e0 | Polytopes as one vertex list with face spans; clips into reused thread-local buffers | 182/182 |
| 186477277 | x sweep for chunk pairs (superseded below: it re-sorted per cluster pair) | 182/182 |
| 9b6971c10 | Broadphase: exact sweep over cluster boxes, then each cluster's chunk intervals sorted once per substep along its widest axis. Padded intervals (`m = 16 eps (max r + max |x|)`, bound in `chunk_pairs`) reject before the unchanged sphere test; pairs restored to loop order (randomised test against the all-pairs loops, mutation-checked) | 182/182 |
| 0a64daf59 | `field_cells` in thread-local buffers; `Polytope::clone_from` reuses capacity (the derived one reallocated) | 182/182 |
| 454220656 | Each pair's polytopes built once (they were built twice, and on every bisection step of the split handover); `intersect` clips `self` directly | 182/182 |
| 1f74bd25d | Cluster index in two dimensions: columns along the widest axis, sorted across within each. A wall was searched along a slab; window tests on b5_wall_impact_v02 fall 3.56e9 -> 2.5e8, broadphase 16.8 -> 9.4 s | 182/182 |
| c4d31ba78 | On one thread, each bond response is applied as it is computed. Previously every substep collected all responses, each holding the whole joint state, in a fresh list. A response reads only its bond and the chunks' motion, which applying the others does not change | 182/182 |
| 5161fb0aa | The statics layout's chunk-to-local index is a dense array, not a SipHash map (about a fifth of the static and implicit solves) | 182/182 |
| fd78110ea | Clip caps: bit-equal repeats of crossing points left out before sorting, then a stable insertion sort on integer keys (a stable sort's order is unique). Randomized test against the original, mutation-checked. Floor drop with layer contact: 65.0 -> 60.8 s | 182/182 |
| 771fc64ac | No second copy of an uncut clip. Static-solve counters (solves, converged, Newton and CG iterations) | 182/182 |
| 8fe879496 | A cluster is indexed only when its queries in the substep repay it (about ten); otherwise it is scanned directly. One 664-chunk structure against one fallen chunk paid 13 us per substep for an index used once. Settlement, default methods: -6.3 % cycles | 182/182 |
| bad18bcb5 | The contact patch's 36 springs in their own non-inlined function (same arithmetic): evaluate compiles better. Bond-response micro-bench 105.6 -> ~100 ns per bond | 182/182 |
| f4d514c62 | Merge of fix/oracle-accuracy (e6fc3ade9), its polytope changes ported to the flat representation with the same arithmetic | 182/182 vs e6fc3ade9 |
| 05e58f430 | Blast decay fit (`fit_decay`) stops at its bisection's fixed point (each step depends only on `(lo, hi)`); no per-substep clone of the blast faces. s_house_blast: 21.9 -> 15.9 s | 182/182 vs e6fc3ade9 |
| a3ddf33c8 | Blast shadowing tests a face's line of sight only against boxes reaching nearer the charge than the face (every point of the line is that near); boxes sorted once per rebuild. 869 M segment tests in s_blast_two_walls; 24.7 -> 21.7 s | 182/182 vs e6fc3ade9 |
| ab72210b0 | Merge of fix/oracle-accuracy (cd112933c: proven methods default, exact frame transfer, contact crushing); the gate now runs the default and `established` methods | 182/182 vs cd112933c |
| 604c1e4a5 | Split data in dense per-chunk arrays (SipHash maps that were only looked up); splits stage 0.07 -> 0.05 s over the catalogue | 182/182 vs cd112933c |
| c002276eb, e328b485d | A static solve unconverged after 10 Newton steps takes its remaining steps by a banded Cholesky of the same tangent (anchored cluster, band <= 600 DOFs), not the PCG. b9 quasi_static 48.5 -> 31.1 s. **Not bit-exact: 178/182; the 4 quasi_static cells of b2_cantilever and s_arch_keystone_removed differ, more accurate.** See below | 178/182 vs cd112933c, suite 121/121 |
| 310a2c14f | The direct static solve takes on a cluster whose layout band is too wide in reverse Cuthill-McKee order (the villa scene pack: band 2766 -> 690 DOFs), with cost limits raised to a 1200-DOF band and 64 MB; the Cholesky goes column by column (each entry the same inner product as row by row, so the same factor) with a column's rows in parallel. Villa quasi_static self-weight (3 frames, hulls test): 2545 -> 1190 s, 30.6 M -> 7.1 M CG iterations; its observation changes only at round-off (bond dissipation 1e-12 relative; 5 broken bonds, 1 fragment). Test: the reordered solve meets banded Cholesky's backward-error bound (mutation-checked); hulls tests 1962 -> 1496 s | 182/182 vs 32f92fbd6 |
| c347352ea | The contact layer's cohorts: about 3 passes instead of 14, no clones, the same sums in the same order (held energy with the stiffness, load with energy, the carried load only to slide). Floor drop with layer contact: 80.3 -> 69.4 s | 182/182 vs e6fc3ade9 |

Small exact changes are now judged by cycle counts (`/usr/bin/time -l`, about 0.1 %
between repeats on a quiet machine, up to 1-2 % under load). Changes to one function
are judged by an in-process micro-bench on captured or mid-run inputs, which is
repeatable to about 0.3 %.

CPU seconds at one thread, from the gate run of 5161fb0aa (both builds run 8 at a
time on the same 16-core machine):

| Cell | Base 0c71ffe13 | 5161fb0aa | Speedup |
|---|---|---|---|
| s_house_settlement, layer | 372.7 | 165.7 | 2.2x |
| s_house_car, layer | 252.3 | 88.0 | 2.9x |
| b9_panel_high, layer | 227.1 | 69.6 | 3.3x |
| s_floor_drop, layer | 203.5 | 65.1 | 3.1x |
| b5_wall_impact_v02, layer | 161.4 | 73.3 | 2.2x |
| s_house_settlement, default | 115.5 | 111.8 | 1.0x |
| s_house_blast, layer | 83.6 | 37.3 | 2.2x |
| b5_wall_impact_v02, default | 80.8 | 62.5 | 1.3x |
| b9_panel_high, quasi_static | 70.8 | 47.6 | 1.5x |
| All 91 one-thread cells | 2353 | 1178 | 2.0x |

After merging fix/oracle-accuracy's e6fc3ade9 (its layer contact does more per contact:
the LIFO shear layer, rocking damping, area moments), `verify_identical.py --bench`,
one thread, sequential, minimum of one run each, base e6fc3ade9 against df60bfba4 (the
same code plus every runtime commit):

| Cell | e6fc3ade9 | df60bfba4 | Speedup |
|---|---|---|---|
| s_house_car, layer | 312.6 | 94.3 | 3.3x |
| b9_panel_high, layer | 205.8 | 59.3 | 3.5x |
| s_floor_drop, layer | 196.4 | 64.9 | 3.0x |
| b5_wall_impact_v10, layer | 33.6 | 18.6 | 1.8x |
| b5_wall_impact_v10, default | 24.8 | 18.4 | 1.3x |
| s_house_car, default | 14.9 | 13.8 | 1.1x |
| b8_frame_sudden, layer | 9.0 | 3.5 | 2.6x |
| s_arch_keystone_removed, layer | 6.9 | 2.3 | 3.1x |
| s_car_brick, layer | 6.9 | 2.2 | 3.1x |
| All 22 bench cells | 830.6 | 296.6 | 2.8x |

("layer" here is `layer_contact,scaled_step_bound`, which cd112933c made part of the
default; the gate now runs the default methods and `STRESS_METHODS=established`.)

Final state, HEAD's code (ab72210b0 plus documentation) against cd112933c, the upstream it
reproduces bit for bit (182/182): `verify_identical.py --bench`, one thread, default
(proven) and established methods:

| Cell | cd112933c | HEAD | Speedup |
|---|---|---|---|
| s_house_car, default | 298.8 | 97.0 | 3.1x |
| s_floor_drop, default | 81.5 | 23.2 | 3.5x |
| b5_wall_impact_v10, default | 35.1 | 16.8 | 2.1x |
| b9_panel_high, default | 34.2 | 10.3 | 3.3x |
| s_car_brick, default | 11.1 | 3.2 | 3.5x |
| s_arch_keystone_removed, default | 4.3 | 1.2 | 3.6x |
| b8_frame_sudden, default | 4.2 | 2.1 | 2.0x |
| b5_wall_impact_v10, established | 25.7 | 19.9 | 1.3x |
| All 22 bench cells | 531.6 | 209.8 | 2.5x |

Where the time goes now: every scene, default methods, one thread, profile stages summed
(578 s): pair geometry 61 % (field cells 23 %, intersection 13 %, contact geometry 12 %,
force law 10 %), bond responses 20 %, broadphase 5.5 %, scripted loads 3 %, chunk frame
loads 2.7 %, integration 1.8 %. The plan items not done (cluster-local Gershgorin rows,
split speedups, probe buffers) act on stages that together take under 0.3 % (step dt
0.2 s, splits 0.07 s, split handover 0.7 s, probes 0.6 s, events 0.1 s). The
replacement-load index is bounded by the whole of the frame loads, 2.7 %. A change
cannot save more than its stage takes: even at zero cost these items would gain at most
0.03 % (step dt, both Gershgorin passes), 0.01 % (splits) and 0.1 % (probes). That is
below the 0.1 % repeat noise of cycle counts, so no build of them could show a gain.

The direct sparse solve for statics (plan Phase 2), sized on b9's quasi-static solves:
- 440 chunks, so 2640 unknowns; a grid ordering gives a band of about 6 x 40 = 240.
- A banded Cholesky is about 2640 x 240^2 = 1.5e8 flops, roughly 50 ms.
- The converged solves take about 100 CG iterations at 27 us, about 2.7 ms.
- The failing ones take about 520 per Newton step, about 14 ms, with a singular or
  indefinite tangent (a mechanism) that Cholesky cannot factor.

So it would be slower as well as round-off-inexact. Not built.

The default methods are dominated by bond responses (about 130 ns per evaluation).
What is left of them is fixed-order arithmetic; the contact patch's 36-spring loop,
for one, is bound by the latency of its four ordered sums. Little more is to be had
there exactly.

Measured and rejected (exact, but no faster):
- Skipping the intact joint's patch moments and energy when `D = 0`. They only enter
  multiplied by `D`, so the skip is exact for nonzero intact moments and finite sums;
  shadow-checked, but 11 % slower (16.6 s vs 15.0 s, settlement intact phase).
- A multiply-rotate hasher for the contact-state maps: no measurable change.
- A branchless contact-patch loop: exact (an open spring adds `+0`, and no sum is ever
  `-0`) but no faster.
- The cluster rotation hoisted out of the per-chunk frame loads, and one flat list of
  cluster members in the broadphase: within noise.
- Replacement loads indexed by chunk (plan item): there are only a few (support or
  chunk removal), and expired ones still add a signed zero, so they cannot simply be
  dropped.
- Micro-bench of bond responses: the branchless loop (128 ns against 100) and the
  intact-joint skip (118 ns) are slower. The compression-only failure indices of the
  crushing test (no `hypot(0, 0)`, no tension or shear indices) are within noise.
- A depth table in `field_cells` (each vertex's depth below each face evaluated once):
  7.7 % more cycles on floor drop.
- The four quadrature distances of `mean_axis_distance` computed as an array: +-2 %
  depending on the scene, within noise.
- The bond's world position in `record_bond_events` computed only for an event: 0.8 %
  fewer instructions but no fewer cycles.
- Impactor-chunk layers evaluated in the pool: rarely more than a few chunks, only
  more CPU.
- Bond responses in the pool below 4096 bonds (b5_wall_impact_v02, 1900 bonds, 16
  threads): 1024/256 at best about as fast in wall time for 2-4x the CPU; finer is
  slower.

**The direct static solve (c002276eb) is the one runtime change that is not bit-exact.**
Its commit message says "exact on the catalogue", which the gate disproved. It changes
two scenes in quasi_static mode, and both changes are physically better or round-off:
- **b2_cantilever:** the 30 N tip load is released at 0.05 s, with no gravity, so the
  quasi-static equilibrium from then on is the beam at rest. The PCG-Newton cannot meet
  its tolerance there (relative to a zero load). 116 frames ended unconverged, keeping
  a stale deflection of up to 3.6e-6 m; only 5 of 121 solves converged, in 6847 Newton
  steps. With the direct solve all 121 converge (17 Newton steps) and the beam ends at
  rest. Tip deflection, its discrete value and the root moment are unchanged
  (-9.609e-3, -58.5).
- **s_arch_keystone_removed:** displacements differ by about 1e-21 m.

The direct solution agrees with the PCG's to about 1e-12 wherever both converge. The full
suite passes with it (121/121).

Excluded (cannot be shown exact or within round-off without the physics tests):
- rows 1, 4, 5, 7, 8, 9 and 2a below (step change, closed form, certificate,
  iterative tolerance, f32, model changes);
- the volume-weighted mean radius of an overlap (`mean_axis_distance`, about 10 % of
  layer contact). Only friction torque that may slide needs it, but no cheap bound
  holds for its quadrature value.

Finding for the model (not a runtime change): the LIFO shear layer's cohort lists grow
long. On s_floor_drop with layer contact, about 1.3 M of the 5.5 M contact evaluations
carried 256 to 1023 cohorts, and 0.4 M carried 1024 or more. Every substep each contact goes
over all of them, and the force law was 22 % of the run before c347352ea. Most cohorts
are slivers of area (round-off flicker of a contact edge, a cohort pushed whenever the
area grows after the displacement moved). Merging cohorts would change rounding, so
it is not a runtime change. Cohorts are only physically one when their anchor
displacements are equal, and `resize` already merges that case (growth at the top while
the displacement has not moved). Any other bound replaces real cohorts with an
approximation. A tolerance for "the same displacement" would be a model decision. Under
the proven default (with `contact_crushing`), the force law is 17 s of 154 s on the
settlement, against 111 s of 318 s with layer contact alone.

Finding for the solver (not a runtime change): in `quasi_static` mode, b9_panel_high's
static solves mostly fail. Over its first 10 frames, 9 of 12 diverge: the plain
Newton loop's relative residual grows from about 10 to 1e21 over its 60 iterations
(about 520 CG iterations each). `equilibrate_or_keep` then keeps the previous state,
which is right if there is truly no equilibrium (a mechanism). Ending the loop early
would not be exact, since one solve climbs to 5e5 and comes back down. A globalised
Newton (backtracking on the residual, the full step kept whenever it already
decreases it) would leave bit-identical every solve whose residual falls at every
step. It would change the others, mainly the ones that fail now: they would converge
where an equilibrium exists, or fail sooner where none does. That is a method change
for a `sim.methods` switch, and nearly all of that cell's 47 s goes to these
divergent iterations.

Tried (experiment, not committed): a backtracking line search (step halved until the
residual falls, down to 1e-6). The same 9 of 12 solves still fail, now with the
relative residual held at 4 to 75 instead of growing to 1e27. The out-of-balance force
stays several times the applied load: the panel has no static equilibrium under that
pressure (a mechanism, the breach of the dynamic run). Keeping the last equilibrium is
the physically right verdict, and globalising Newton changes nothing but the cost. Not
adopted. A direct sparse solve would only make these failing iterations cheaper, in a
mode three tests use once each, at round-off cost to exactness. Not pursued. (Corrected: the
globalised *tangent* Newton of `tangent_statics` finds equilibria for most of these
solves; it was the secant direction that failed, not the existence of an equilibrium.)

### Contact neighbour list (bit-exact, switch on, 2026-10-16)

DEM and molecular-dynamics codes do not search for contact pairs every step. They
keep a Verlet list of the pairs within a skin and search again only when something
has moved far enough to enter it (`findings/stress-literature-review.md`). The oracle
ran `chunk_pairs` every substep, and inside impact windows that is every substep of
the run.

**Method** (`world.rs`, `NeighbourList`): the list is `chunk_pairs` with every bounding
radius grown by `s / 2`, at the centres of the build. Skin `s` = 0.5 x the smallest
active chunk's bounding radius. Each substep keeps the listed pairs that pass
`chunk_pairs`' own tests (cluster boxes, `admit`, padded intervals, spheres) at the
current centres. The tests are the same and so is the order (lexicographic in
`(cluster a, cluster b, chunk a, chunk b)`), so the result is `chunk_pairs`' own, pair
for pair. The list is rebuilt when:
- the active chunks, their clusters or their radii change;
- a cluster's anchored or driven flag changes;
- the two largest centre displacements since the build sum to more than
  `s - 256 eps C`.

That last condition is sufficient: a pair `chunk_pairs` returns now was within
`ra + rb + s` minus far more than round-off at the build, so the grown search found it.
The test reads only centres and fixed radii, so rotation cannot move a pair in or out.
Every motion is counted, because the displacement is measured from the world centres
every substep: hidden deformation, frame folds, rewinds and splits included. Pair
evaluation (overlap, force law, friction state) still runs every substep on every pair
whose spheres touch.

**Switch:** `STRESS_REF_CONTACT_SKIN` (the skin as a fraction of the smallest radius,
default 0.5; `0` or `off` runs `chunk_pairs` every substep). It is a runtime switch, not
a `sim.methods` field: it changes no result. Non-finite centres, or a skin below
`1e-9 (max |x| + max r)`, fall back to `chunk_pairs`.

**Gate:** `verify_identical.py` with base 35e1421 gives 182/182 identical (both method
sets, one thread and all threads, forced solve modes). The test
`the_neighbour_list_matches_chunk_pairs_every_step` compares the list with
`chunk_pairs` at every step. It covers moving bodies, approaching pairs, exact ties on a
grid, hidden jumps, splits, cluster changes, flag flips and NaN centres. It is
mutation-checked: it fails if only the largest displacement is counted, if the flags
or cluster identity are left out of the build's identity, if the displacement check is
skipped, or if the radii grow by `s / 4`.

**Skin** (b5_wall_impact_v10, proven methods, pair search):

| skin / r_min | builds | list pairs per substep | pair search |
|---|---|---|---|
| off | | | 0.75 s |
| 0.25 | 132 | 224 | 0.23 s |
| 0.5 | 67 | 278 | 0.20 s |
| 1 | 35 | 636 | 0.27 s |
| 2 | 19 | 1322 | 0.43 s |

**Profile** (`--profile`, one thread, scenes from `examples/rubble_references`, proven
methods unless marked established). `base` is 35e1421, `off` is this build with the
list off, `on` is this build. All 30 runs were concurrent on a shared 16-core machine,
so wall time is loaded. User CPU is the steadier measure, and the stage times are
per-substep work. Every observation is identical across base, off and on.
house_1story_ram (proven methods) is not in the table: after 110 CPU-minutes per
variant it had not finished.

| Case | substeps | list builds | contacts, us/substep (base / off / on) | pair search, us/substep (off / on) | broadphase stage, s (off / on) | wall s, base / on (user s) |
|---|---|---|---|---|---|---|
| b5_wall_impact_v10 | 24162 | 67 | 747 / 720 / 693 | 57.5 / 16.5 | 2.45 / 1.53 | 43.0 / 42.6 (26.3 / 25.9) |
| b5_wall_impact_v15 | 24159 | 92 | 5556 / 5529 / 5383 | 194.9 / 43.6 | 10.41 / 6.58 | 158.5 / 155.1 (112.4 / 109.6) |
| b5_wall_impact_v20 | 8058 | 32 | 781 / 733 / 811 | 55.2 / 19.6 | 0.74 / 0.48 | 16.8 / 17.0 (8.2 / 8.3) |
| b5_wall_impact_v30 | 8058 | 46 | 670 / 658 / 642 | 40.7 / 12.2 | 0.57 / 0.40 | 15.5 / 15.3 (7.5 / 7.4) |
| b5_wall_impact_v40 | 8058 | 58 | 1902 / 1847 / 1875 | 27.3 / 15.0 | 0.69 / 0.65 | 25.3 / 25.2 (13.2 / 13.0) |
| s_house_car | 43100 | 185 | 3072 / 3083 / 2988 | 130.8 / 23.3 | 12.25 / 7.26 | 164.0 / 160.2 (117.5 / 113.8) |
| s_floor_drop | 104598 | 58 | 350 / 356 / 337 | 23.6 / 8.2 | 5.80 / 4.04 | 58.9 / 56.0 (33.9 / 32.6) |
| b5_wall_impact_v10, established | 39927 | 57 | 307 / 310 / 227 | 111.0 / 30.8 | 6.26 / 2.90 | 41.8 / 39.2 (25.2 / 23.0) |
| s_house_car, established | 51300 | 137 | 156 / 156 / 93 | 69.1 / 11.3 | 4.20 / 1.28 | 33.0 / 30.4 (18.3 / 16.5) |

Effect:
- Pair search falls 2-6x per substep. Its time is deterministic work; the counts
  (builds, list pairs) reproduce exactly.
- Under the proven methods pair evaluation dominates contacts, and that is unchanged.
  The run saves 1-4 % of CPU (house hit by a car 117.5 -> 113.8 s; b5 v20 +0.1 s,
  within noise).
- Under the established methods (sample-point contact, cheaper per pair) contact per
  substep falls 26-40 % and CPU 9-10 %.
- What remains of the broadphase stage is the oriented boxes and polytopes of the
  candidates, built every substep for evaluation.

**Impactor and ground paths: no list.** Their detection is one test per active chunk
per substep: a centre distance against the impactor, or a height against the ground.
A list cannot be cheaper than that, because checking a chunk's displacement costs the
same test. The time in those stages (`impactors` 2.5 s on b5 v10) is the evaluation of
the chunks in reach (ball and box overlaps), which must run every substep. The tests
themselves are about 20 M per path on b5 v10. At a few ns each that is about 0.05 s of
26 s, an estimate rather than a measurement.

**Porting to stress-gpu's islands.** stress-gpu already reuses its candidates across
substeps (`contacts::plan`), with per-chunk travel budgets that the GPU checks
(`travel_check`, re-planning when a chunk outruns its budget). In effect that is a
neighbour list with a budget in place of a skin. Three things differ:
1. The plan's chunk pairs are an all-pairs loop over chunks of different clusters
   (`contacts.rs`). At 10,000 bodies that is 5e7 tests per plan. The sweep build of
   `chunk_pairs` (cluster boxes, then the two-axis cluster index) would replace it, with
   the plan's budgets as the growth of each radius.
2. A plan lasts a fixed `PLAN_STEPS = 4` steps and each re-plan rebuilds the device
   buffers (`solver.rs`, `step`). Keeping the plan until a travel check fails (or the
   island's membership or flags change, as this list does) would make re-plans
   event-driven. That needs the candidate buffers re-uploadable without a full
   re-layout.
3. Each candidate's exact sphere test should run in the shader before its contact is
   evaluated, so a longer-lived (larger) list costs a test per pair, not an
   evaluation. A filtered list is bit-identical to a fresh plan only if the contact
   loads are summed in an order that does not depend on the list (here, candidate
   order with the filter preserving it). Not checked for the GPU's accumulation.

Item 1 is CPU work in Rust, reusing `chunk_pairs` from stress-ref. Items 2-3 change the
plan lifetime and a shader filter. Pair state is already keyed by chunk identity
(`PairKey`), so a list that persists across re-plans keeps it as it is.

## Queue (from the real-time performance analysis, 2026-10-10)

Each oracle-applicable idea, with its class. Exact ones go in directly. Step and model
changes go behind `sim.methods` switches, off by default, each with measured fidelity:
- the catalogue A/B against the goldens;
- discrete outcomes (fails or not, where, when, how, pieces log-binned, debris speed);
- momentum and energy ledgers;
- runtime.

| Idea | Class | Status |
|---|---|---|
| True stable step: a certified bound on the highest frequency of the intact network (every fragment's is lower) | Step change | switch `true_step` (off); see below |
| Gameplay-fidelity comparison: fails or not, where, when, pieces log-binned, debris speed +-30 % | Tooling | `scripts/fidelity.py` |
| True combined step: bonds plus the contacts that can engage, warm-started each frame (contact sets the step in every catalogue scene) | Step change | `lattice_contact_step` covers the densest network (below). Beyond that, research. A certified contact block needs a bound over unknown contact points (lever arm up to the radius) and over pairs engaging within the frame. The default global bound (`6 k / m`, every chunk on six faces) ignores rotation, so it is not certified either, and a certified bound may come out *below* it. `frame_contact_step`'s Gershgorin sum is 7-20x below it |
| Multirate (subcycling) explicit: per-chunk steps in powers of two (2.5-3.5x on authored packs, none on uniform grids) | Integration change | not pursued for the oracle. No catalogue scene gains (uniform grids, and contact sets the step), and partitioned explicit integrators need interface treatment (Belytschko-Smolinski-Liu) whose error the ground truth would carry. It belongs to the GPU's impact windows |
| Influence-line wake test for the adaptive mode: a load change accepted without a solve when `utilisation + influence x load < threshold - margin` | Skip decision | no catalogue scene runs adaptive (only the gate's forced cells), so the oracle has no idle cost to save; a GPU concern |
| Hierarchy import (L0/L1/L2 with parent links) | Exact (L2 unchanged) | the packs carry one level: `nodePieces` (authored parts; villa 243 over 1045 chunks, house-2story 135 over 599), and `nodeGroups` is trivial. No L1 level |
| L0 members as statically condensed super-elements (exact for statics), exact recovery of each L2 bond | Exact for statics | low payoff on these packs: pieces average 4-5 chunks, so most chunks are interface DOFs and condensation barely shrinks the system. The banded direct solve (310a2c14f) already takes the villa |
| Cold-bond fast path with closed-form fatigue | Approximation | exact variant rejected (slower). Bond responses are 39 % of b5_v10 at one thread (137 ns per bond-substep). The approximate path would skip the strain-rate filter, whose history sets the dynamic increase factor (strengths up to 2-3x under impact): not adopted without an exact filter update |
| Reduction cadence for free islands (rigid acceleration, drift removal) | Approximation | not pursued for the oracle: on b5_v10 at one thread, rigid accelerations are below 0.1 % of the run and `fold_rigid_drift` about 1.5 % of samples. Their cost is the GPU's per-substep barriers, which the oracle does not have |
| Regime map (contact time against wave transit and fundamental period) | Analysis | `examples/regime_map` (below) |
| Explicit windows in implicit mode: impacts and failure in the explicit integrator, the rest implicit (the analysis's tier 2 + tier 3 windows) | Integration choice (reference inside windows) | switches `implicit_contact_windows`, `implicit_failure_windows` (default on); implicit A/B 72/91 -> 80/91 (below) |

### `true_step` (switch, off)

**Which bound sets the step.** In every catalogue scene measured (b5, b7, b8, b9,
s_house_car), the contact bound is below the stress bound, and the two combine as
`1/dt^2 = 1/dt_s^2 + 1/dt_c^2`. The analysis's 5-41x was measured against the
unscaled Gershgorin step; the default scaled bound is 2.5-8x below the true step.
`frame_contact_step` (the contact bound from the contacts that can engage) gives 7-20x
*more* substeps in impact runs (b5_v10 11,327 -> 216,555): not a speed lever.

**A Ritz value is not a bound.** Power iteration's Ritz value plus its residual bounds
only the eigenvalue nearest it. Taken as a bound on the top one, it gave 1.37e8 on the
cantilever against the true 1.99e8 (dense Jacobi), and the stability harness blew up at
safety 0.9 on nearly every scene. A fixed iteration count with a margin (the GPU's
`true_stable_dt`, 3 %) converges here, about 300 iterations to the top mode. But it is
an estimate, not a certificate.

**Certified bound.**
- `A = L^-1 K L^-T` is similar to `M^-1 K`, and `lambda_max(A) <= rho(|A|) <= max_i
  (|A| v)_i / v_i` for any positive `v` (Collatz-Wielandt). Every power-iteration iterate
  on `|A|` is therefore a bound, starting from the row sums.
- On networks bonded face to face, `|A|` is `A` up to a diagonal sign similarity, so the
  bound is the eigenvalue itself: 1.988831e8 on the cantilever, against 3.18e8 for
  Gershgorin. Test: `true_step_bounds_the_highest_frequency`.
- Damping enters at the structure's largest dashpot-to-stiffness ratio.
- A cluster takes the larger of this step and its own Gershgorin step.

**Measured.**
- **Stability harness** (with `frame_contact_step`, so the stress bound sets the step):
  stable at safety 0.9 on every scene; most blow up from 1.1, so the bound is close to
  the real limit.
- **Switch off:** 182/182 against aef673ef7.
- **Switch on:**
  - `scripts/fidelity.py`: 34/35 scenes pass every gameplay metric.
    s_arch_keystone_removed's collapse differs: 8 -> 14 fragments, debris 0.29 -> 0.82 m/s.
  - Substeps 0.74-1.0x; catalogue CPU 557 -> 506 s (-9 %).
  - A/B: 87/91 both ways.

**Status:** off by default. It is a step change for 9 %; the larger lever is the
contact bound (queued).

### Contact step on the densest network: `lattice_contact_step` (switch, off)

**The global contact bound leaves out the tangential layers.** The default bound is
`w^2 = 6 k / m` per chunk ("buried in rubble on six faces", normal stiffness only). In
the densest network, a lattice of equal cubes in face contact, each cube in the
alternating mode feels, along an axis:
- its two faces' normal layers (`4 k`);
- its four other faces' tangential layers (`8 k_t`, with `k_t / k = G / E' = (1 - nu) / 2`).

**Bloch waves.** Using the code's own layer moduli at the face centres (normal,
tangential, rocking `k'' I_A`, torsion `k_t'' J`), the top eigenvalue is
7.48 k/m at nu = 0.15 (brick, mortar), 7.27 at 0.2 (concrete), 6.85 at 0.3 (steel),
8.10 at 0. Safety 0.9 absorbs up to 7.41 k/m.

**Measured** (`examples/contact_lattice`: unbonded cubes pressed together with the
outer shell held, no damping, random velocities):

| Lattice | Predicted critical safety | Observed |
|---|---|---|
| 6^3 brick | 0.904 | stable at 0.9, blows up at 1.0 |
| 12^3 brick | 0.898 | stable at 0.9 for 40,000 substeps, blows up at 0.95 |

So the default holds at 0.9 (the friction caps and finite size give a little), but
its margin on fully broken masonry is under about 5 %, not the 10 % that safety 0.9 is
meant to give.

**Switch.** `w^2 = 1.02 max_d (4 k_d + 4 sum_(e != d) k_t,e) / m`, each chunk's own faces
along each axis. The 2 % is the rotation coupling (Bloch: at most 1.25 %, at nu = 0).
Measured:
- 6^3 brick and concrete: stable at 1.0, blows up at 1.1 (the bound is tight);
- switch off: 182/182 against 96ef867e6;
- switch on: fidelity 34/35 (b8_frame_sudden's collapse differs: 8 -> 22 fragments),
  catalogue CPU +1 %, A/B 87/91 both ways.

**Status: default on** (proven set, after the change below). As the default, the
gate changes 80 cells (every contact-bound explicit run); the measurements above (A/B
87/91 both ways, fidelity 34/35, CPU +1 %) are its evidence.
`STRESS_METHODS=-lattice_contact_step` restores the `6 k / m` bound.

**The fracture energy test, made robust.** With four halvings,
`fracture_never_gains_energy_and_dissipation_is_accounted` judged the extrapolated limit
from three noisy points. Fracture events land on different substeps, so the halving
ratios scatter (default: 2.31, 1.95, 2.30, 2.29). With the switch it failed narrowly:
Aitken's limit 1.9e-4 against a spread of 1.0e-4. A fifth halving (safety 0.03125)
settles both:

| Run | Errors | Aitken limit | Spread |
|---|---|---|---|
| default | ..., 4.82e-3, 2.10e-3 | 1.6e-5 | 4.8e-3 |
| switch | ..., 4.91e-3, 1.93e-3 | -2.4e-3 | 2.7e-3 |

No tolerance is chosen; the test takes about 30 s longer. Suite 125/125 with the
switch on.

**Other contact geometries** (`examples/contact_lattice MATERIAL N VARIANT`, 6^3, brick
and concrete, safeties 0.8-1.1):
- **`rotated`** (cubes turned +-10 degrees in a checkerboard, corner to face): stable
  at every safety under both bounds (energy growth at most 1.02 with the lattice bound,
  1.23 with the old one). Corner contacts are softer than faces.
- **`layers`** (cubes alternating with plates a quarter as thick, every contact the
  same overlap): stable at every safety under both bounds. A thin plate between thick
  blocks sees a layer of `h_thin + h_thick`; the bound assumes an equal neighbour
  (`2 h_thin`), here 2.5x stiffer.
The densest equal-cube face packing stays the critical case, and the per-chunk lattice
bound covers it (stable at 1.0, above).

**Stability harness fixed.** `examples/stability` took its reference energy after the
first frame, so a run that blew up within that frame read as stable. It now evaluates
the bonds' stored energy at the perturbed state before any step. Re-run:
- the default bounds hold at 0.9-2.0 on almost every scene (contact binds the step);
- `frame_contact_step` holds at 0.9;
- the certified `true_step` holds at 0.9 and blows up at 1.0-1.1 on many scenes, as a
  tight bound should.

### Regime map (`examples/regime_map`)

Per scene, the three times that decide a solver tier (the real-time analysis, section 8.2):
- `t_c`: an impactor's Hertz contact time, a load's rise or pulse time, or a removal's
  duration;
- `T_w`: the wave transit across the structure's largest extent;
- `T_1`: the fundamental period, by inverse iteration with the banded direct solve
  (`ReferenceSolver::fundamental_period`; the cantilever's is Euler-Bernoulli's first
  mode to 0.1 %, test `fundamental_period_is_the_first_bending_mode`).

Selected rows (tiers by the analysis's rules, `>>` read as 10x):

| Scene | t_c | T_w | T_1 | Tier |
|---|---|---|---|---|
| b5_wall_impact_v02 / v10 / v40 | 3.7 / 2.6 / 2.0 ms | 0.59 ms | 37 ms | 2 |
| b7_masonry_v04 / v15 | 1.6 / 1.3 ms | 1.2 ms | 119 ms | 2, at the edge of 3 |
| b9_panel (pressure) | 10 ms | 0.62 ms | 12 ms | 2 |
| s_car_brick, s_house_car | 5.9, 5.7 ms | 1.4, 1.8 ms | 84, 61 ms | 2 |
| b8_frame_gradual, s_house_settlement | 1 s | 2.6, 1.8 ms | 373, 61 ms | 2, 1 |
| sudden removals (b4, b8, keystone) | 0 | | | 3 |

Packs, T_1 only (no loading): house-2story 65 ms, house-1story 43 ms, brick-building
36 ms, villa-savoye 1.28 s.

Impacts also print `T_w,member`, the transit across the structure's extent along the
impact (the wall's thickness): b5's Hertz time is 35-65 of them, b7's 20-27.

**Tier-2 calibration** (`scripts/fidelity.py --b-set 'sim.solve_mode="implicit"'`).
Every scene explicit (the truth) against the oracle's implicit solve (Newmark with
Newton on the joints' tangent). In that mode contacts still set the substep, so this is
implicit bond integration at about the explicit substep: tier 2's best case. The
implicit step alone (1/60 s against the scene's frame step) changes nothing.

Results:
- **17/35 scenes pass every gameplay metric:** analytic benchmarks, cantilevers, bar
  waves, support loss (gradual and sudden), elastic frames, the arch standing,
  overhangs, the static floor and house.
- **Fast-impact failures are missed:**
  - b5_wall_impact_v40 does not breach: failure flags differ, 32 -> 0 chunks detached;
  - b6_spall does not spall: 720 -> 0 detached.
- **The rest differ in pieces and debris speed:** the slower impacts, b7, the cars, and
  the collapses (keystone, house settlement, house car).

So the oracle supports the analysis's intent: punch-through and spall need explicit
windows. But the stated rule (`t_c <= T_w` for tier 3) puts b5_v40 in tier 2 with either
transit time. A breach is local punching and bending, not a through-thickness wave.
The criterion needs another measure. A candidate, now in the regime map: the elastic
Hertz peak pressure over the target's compressive strength.

| Scene | `p0 / f_c` |
|---|---|
| b5_v40 (implicit misses the breach) | 214 |
| b5_v10 | 123 |
| b5_v02 | 65 |
| cars | 49-54 |
| b7 | 26-44 |

It ranks the one failure first. A speed sweep of the same 1000 kg box ram (b5's scene
at 10-40 m/s, explicit against implicit, fidelity's judge) places the threshold:

| Speed | Explicit (the truth) | Implicit |
|---|---|---|
| 10, 15 m/s | push-over: the whole wall falls (800 chunks) | same failure mode; pieces differ (58 -> 4 at 15 m/s) |
| 20, 25, 30, 35, 40 m/s | punch-through: 32 chunks out at about the ram's speed | nothing detaches; the collapse flag is lost |

So tier 2 fails exactly where punch-through begins, between 15 and 20 m/s.

**More rams and walls:**

| Ram | Explicit punch-through starts | `p0 / f_c` across the boundary |
|---|---|---|
| 500 kg on b5's wall | 20 m/s | 126 -> 141 |
| 1000 kg | 20 m/s | about 145 -> 160 |
| 2000 kg | 20 m/s | 166 -> 187 |
| b7's brick wall, 30 kg ram, 4-40 m/s | never: the whole wall collapses | implicit matches the failure at every speed (pieces differ) |

A fourfold change in ram mass leaves the boundary at the same speed, while the
pressure ratio across it spans 126-187. So for this wall the punch-through criterion is
the impact speed, not the contact pressure. The wave ratios barely move with speed
either (`t_c` goes as `v^-1/5`).

At 500 kg and 20 m/s, implicit gives the wrong mode: the whole wall pushes over (800
chunks) where the truth punches 32 out. Its failure flags agree, so fidelity's
`fails` metric misses it; `where` catches it.

**Walls** (`examples/wall_variants`: b5's ram against walls 1-4 chunks of 0.1 m thick,
concrete and brick, 10-60 m/s). "Partial" is chunks detached short of a push-over (for
the thicker walls, large partial failures):

| Wall | Explicit (the truth) | Implicit agrees at |
|---|---|---|
| concrete 0.1 m | a 16-chunk hole at every speed, 10-60 m/s | no speed |
| concrete 0.2 m (b5) | push-over to 15 m/s, punch-through from 20 | 10-15 m/s |
| concrete 0.3, 0.4 m | push-over at 10 m/s, partial failure (760-1120 chunks) from 15 | 10 m/s |
| brick 0.2 m | push-over at 10 m/s, punch-through from 15 | 10 m/s |
| brick 0.3 m | partial failure from 10 m/s | no speed |

The critical speed depends on the wall: under 10 m/s for 0.1 m concrete and 0.3 m
brick, 10-15 m/s for 0.2 m brick and 0.3-0.4 m concrete, 15-20 m/s for 0.2 m
concrete. It is not monotonic in thickness. Implicit at the frame step reproduces
only push-over.

For the GPU tiering: the rule `t_c <= T_w` does not find these regimes. A critical
impact speed per wall type, calibrated by this sweep, does; without a calibration,
hard impacts above about 10 m/s belong in explicit windows.

**Superseded in the oracle.** Explicit windows (default on in implicit mode, below)
take every impact in the oracle's implicit mode in the explicit integrator, so the
oracle needs no critical speed per wall. The sweep stays as the record for a GPU tier
selector that cannot afford windows everywhere.

**Caveats:**
- In the oracle's implicit mode, contacts act every substep but bonds advance once per
  `implicit_dt` (here the 1/60 s frame), and `implicit_predict` moves the hidden DOFs
  under the contact loads with no bond resistance in between. During a hard impact
  that coupling creates energy: 0.3 m brick at 15 m/s reached chunks at 1e18 m/s,
  contact dissipation -3e7 J; b5_v40 -3.1e7 J; b7_v15 -1.55e6 J. **Fixed by
  `implicit_contact_steps` (default on, below).**
- The analysis's tier 2 takes impacts as velocity conditions
  (`ImpactModel::VelocityCondition`, momentum exact). That model lives only in the
  engine API (`api.rs`), so this calibration measures tier 2 with explicit contacts.
  Calibrating with velocity conditions needs a driver through that API: done below.

**Tier 2 with velocity conditions, through PhysX** (`stress-physx` example
`tier2_velocity`, PhysX CPU from a macOS install via `PHYSX_ROOT`). The same walls and
ram as above, 6 frames of 1/60 s, three ways:
- the standalone world, explicit (the truth);
- PhysX with Hertz pulses and the explicit solve;
- PhysX with velocity conditions and the implicit solve (the analysis's tier 2).

Both PhysX runs allow at most one correction. "Detached" counts chunks off the anchored
body; "fast" counts those moving at over half the ram's initial speed.

| Wall, speed | Truth | PhysX explicit | PhysX tier 2 | Ram after (truth / PhysX) |
|---|---|---|---|---|
| concrete 0.2 m, 10 m/s | push-over (800) | push-over | push-over | 8.4 / 0.8-0.9 m/s |
| concrete 0.2 m, 20 m/s | punch-through: 32 out, all fast | 800, none fast | 800, none fast | 18.1 / 1.0 |
| concrete 0.2 m, 40 m/s | punch-through: 32 fast | 800, none fast | 800, none fast | 37.0 / 1.2-11.6 |
| brick 0.2 m, 15 m/s | punch-through: 32 fast | 800, none fast | 800, 19 fast | 14.0 / 0.3-0.7 |
| brick 0.2 m, 30 m/s | 719 out, 40 fast | 61, none fast | 800, 149 fast | 27.4 / 2.0-10.9 |

No PhysX configuration punches through at any speed, explicit or tier 2, with 0, 1 or
4 corrections. PhysX resolves the ram against the rigid anchored wall and stops it
within a frame; the stress solver then receives the whole ram's momentum, and the wall
shatters. That is the handover `stress-physx`'s own documentation names: a correction
pass cannot shorten a rigid contact. So through the engine API, tier 2 cannot be
calibrated apart from the coupling: the coupling's error is larger than the tier's at
every speed. The coupled world (`World::with_engine`) avoids it by solving impacts in
impact islands with the world's own contacts. That is the route for a tiered pipeline,
and the oracle's explicit-against-implicit sweep above remains the tier calibration.

**Implicit steps that do not converge (fixed).** `newmark_step` committed its best
state when Newton stalled, with relative residuals up to 0.52 in the gate's implicit
cells. `newmark_step_subdivided` now rejects such a step: the cluster returns to the
step's start and takes two half steps under the same averaged loads, recursively, at
most 6 deep (1/64). Every implicit cell's worst residual is now about 1e-9; a step
that converges is unchanged.

| Cell | Worst residual | Broken bonds |
|---|---|---|
| b8_frame_sudden established | 0.52 -> 1e-9 | 53 -> 28 |
| b7_v15 | 0.30 -> 1e-9 | 280 -> 285 |
| b9_panel_high (default) | 5e-4 -> 1e-9 | unchanged |

**`implicit_contact_steps` (default on).** A bonded cluster takes its implicit step at
the end of every substep in which a contact acts on it. That is the mode's own
condition that `implicit_dt` resolve the dynamics that matter; a contact's do while it
lasts.

| Case | Before | After |
|---|---|---|
| 0.3 m brick at 15 m/s | chunks at 1e18 m/s, contact dissipation -3.0e7 J | at most 26.8 m/s, +2.3e4 J |
| b5_v40 implicit | -3.1e7 J | +5.4e4 J |
| b7_v15 implicit (gate cell) | -1.55e6 J | -2.0e4 J |

- Switch off: 182/182 identical.
- Implicit A/B: 69/91 -> 72/91, no regression. b5's failure mode and hole area and
  b6's back-face velocity now pass against OpenCourant.
- As the default it changes only the four default-method implicit cells. Suite 126/126.

**Closed by explicit windows (below); this paragraph is the history.** b7_v15 implicit booked -2.0e4 J. A substep trace finds 17 substeps with
a falling contact ledger. The largest, -1.49e5 J at 9.62 ms, is a contact substep that
also took an implicit step: more than the impact's input (about 3.4 kJ), and booked
back later. Traced further: no split, handover or split release in that substep. The
contact's stored energy rose 1.466e5 J as its dissipation fell 1.488e5 J. The implicit
step at the substep's end relocated the chunks after their contact force was evaluated,
so the overlap deepened without the contact doing that work, and the ledger books the
stored energy as negative dissipation. Closing it needs the step's displacement inside
the contact's work: the contact evaluated against the implicit solution, or the
contact load inside the Newton solve (its stiffness in the tangent). That is a redesign
of how implicit mode couples bonds and contacts; open.

It is one event, not a steady error. At the scene's safety (0.5) b7_v15 implicit books
-2.0e4 J; at 0.9, 0.45 and 0.225 it books +2026, +1656 and +1271 J. It is
thread-independent: the forced implicit, quasi_static and adaptive cells of b4, b7, b8
and b9 are identical at 1 and 8 threads (the gate checks the forced modes only at 1).
**Closed by explicit windows (below):** with `implicit_contact_windows` b7_v15 implicit
books +2089 J, as explicit does.

### Explicit windows in implicit mode (switches, default on)

The analysis's tiering — implicit at the frame step, with explicit windows for
impacts and failure — as two switches. A windowed cluster takes the frame's substeps
in the explicit integrator (the reference), through the same per-cluster split adaptive
mode uses; the others stay implicit.

**How it works.**
- **Hand-over at frame boundaries.** Every frame ends in an implicit step, so a window
  opens on a consistent state. Leaving a window, Newmark takes `a_n` from the explicit
  integrator's last hidden acceleration.
- **The substep** honours the windowed clusters' stable stress step.
- **Fragments** of a windowed cluster stay windowed to the frame's end.

**`implicit_contact_windows`.** At a frame's start, every body is bounded by a sphere
(chunk circumspheres). A pair can close at most at its relative speed, plus rim speed
from spin and hidden motion, plus `|g| T`. A window opens when:
- the bodies close faster than the engine's resting band `|g| T` (slower is a resting
  contact, which the implicit step resolves); and
- the gap is under that bound's travel in one frame.

**`implicit_failure_windows`.** A cluster takes the frame explicitly in two cases:
- its most utilised intact bond is failing (failure index 1), or reaches failure within
  two frames at its rate over the last frame;
- a bond starts to fail inside an implicit step anyway (an event, or an intact bond at
  index 1). Then the frame is solved again from its start with that cluster windowed,
  until no implicit step meets a failure. This is the coupled world's redo pattern;
  windows only grow, so it ends. The count is `redone_frames`.

Fracture runs on a chunk's wave transit. The implicit step commits damage once per step
and spreads a cascade over steps.

**Evidence** (implicit with windows against the explicit truth):

| Case | Explicit | Implicit | Implicit + windows |
|---|---|---|---|
| b5's wall, 20 m/s: chunks out (fast) | 32 (32) | 800 (52): push-over | 32 (32) |
| b5's wall, 20 m/s: contact dissipation | 16040 J | 25261 J | 16040 J |
| b7_v15: contact dissipation | +2089 J | -2.0e4 J | +2089 J |
| b5_v40: broken bonds | 48 | 1079 | 48 |
| b9_panel_high: broken bonds, fragment speed | 60, 2.792 m/s | 220, 2.519 | 60, 2.790 |
| b9_panel_low (elastic), b3 bar wave | - | - | no window; bit-identical to implicit |

**Hand-over.** An undamped stack in free vibration, with windows every 2nd or 3rd frame,
keeps its energy within 1.4x the explicit integrator's own frame-end fluctuation. That
fluctuation is 17 % of the vibration energy, against 3e-10 for the implicit step
alone.

**Pulse windows rejected.** An explicit window while a pulse load acts (Friedlander,
half-sine, blast) fixed b9 but broke b3's wave speed (0.2 % -> 7.1 %). b3's and b9's
pulses each span about ten implicit steps, so the pulse's length does not separate
them. What implicit misses in b9 is the fracture, not the load.

**Results.**
- Switches off: 182/182 identical.
- Implicit A/B with both on: gated metrics 72/91 -> 80/91. Now passing: spall
  (b6, both oracles), back-face velocity, ball speed lost and debris speed, speed lost,
  two failure forces.
- The only regressions are b5_v40's failure mode and hole area against OpenCourant.
  There the explicit solve gives the same (a 0.32 m^2 hole where OpenCourant pushes the
  wall over): implicit had passed by the wrong mechanism.
- Tests (`solve_modes.rs`):
  - `contact_windows_resolve_punch_through_as_the_explicit_solve`
  - `contact_window_closes_when_the_body_is_out_of_reach`
  - `failure_windows_resolve_the_breach_as_the_explicit_solve`
  - `window_hand_over_keeps_the_energy_of_free_vibration`

**Cost** (8 threads, wall seconds, broken bonds):

| Scene | Explicit | Implicit | Implicit + windows |
|---|---|---|---|
| b9_panel_low (elastic, no impact) | 1.8 s | 0.8 s | 0.9 s, no window |
| b8_frame_sudden_elastic | 1.2 s | 1.4 s | 1.4 s |
| b4_support_loss_sudden | 0.0 s | 0.0 s | 0.0 s, no window |
| b8_frame_sudden | 3.4 s, 23 | 7.0 s, 77 | 2.7 s, 16 (1 frame redone) |
| b9_panel_high | 4.7 s, 60 | 11.5 s, 224 | 4.8 s, 60 (1 redone) |
| b7_masonry_v04 / v15 | 4.1 / 5.5 s | 5.7 / 3.7 s | 4.1 / 5.4 s, as explicit |
| b5_wall_impact_v10 / v40 | 15.4 / 4.6 s | 42.8 / 18.0 s | 15.5 / 4.7 s, as explicit |
| b6_spall | 1.3 s, 432 | 1.6 s, 0 | 1.4 s, 432 |

Windows never cost more than the explicit solve. Where something fails they are 1.6-3.8x
faster than plain implicit, whose Newton iterations through fracture are expensive.
Without impact or failure they cost what implicit costs. In the catalogue's impact
scenes the window covers the whole run, so there implicit mode is the explicit solve.

**b8_frame_sudden** is the one case where windows and explicit differ:
- The elastic phase before failure runs implicit, and the first crack comes 0.24 ms
  later (64.12 against 63.88 ms).
- The collapse that follows is chaotic: 16 bonds and fragments against explicit's 23 and
  22, and contact dissipation 9974 against 3018 J (more pieces strike each other).
- No energy is injected: the balance residual of the two runs grows alike, from the
  column removal's external work (32.6 against 33.3 kJ at 0.65 s).

**Default on.** Suite 130/130 with both switches on. Against the previous default the
gate changes only 7 implicit cells; every explicit, adaptive and quasi-static cell is
identical (175/182):
- b4_support_loss_sudden gains only the `redone_frames` value.
- b2_cantilever and s_floor_static take a few frames in failure windows (the rate rule
  fires as the load comes on: +81 and +24 substeps) and agree to round-off.
- s_arch_keystone_removed now matches explicit: 6 bonds and 8 fragments, contact
  dissipation 924.6 against 927.5 J. Implicit alone breaks 14 into 16.
- b7_masonry_v15, b8_frame_sudden and b9_panel_high change as in the tables above.

### `corotational_bonds` (switch, default on: DECISIONS.md 61)

The small-strain bond map is invariant under infinitesimal rotations only. Its forces
act at undeformed lever arms and turn the system by `sum (u_a - u_b) x f`
(DECISIONS.md 36). The co-rotational map (`bond::CoRotation`, DECISIONS.md 52) works
like this:
- **Deformation:** each chunk turns by the finite rotation of its rotation vector. The
  bond reads its patch points' separation and relative rotation in the chunks' mean
  frame.
- **Loads:** the exact transpose of the map's variation, with spatial moments.
- **Rotations:** the explicit integrator composes them exactly.

| Measure | Small strain | Co-rotational |
|---|---|---|
| Angular momentum drift, 40 m/s impact, safety 0.5 -> 0.0625 | 5.94e-4 of `|L0|` at every substep | 5.2e-5 -> 6.6e-6, converging to 2.4e-8 |
| Fracture energy closure (extrapolated; chaotic runs, estimates move by ~5e-3) | -2.4e-3 `e0` | +2.7e-4 to +8.4e-4 `e0` |
| Elastic impact energy closure (extrapolated) | 4.3e-6 | 2.1e-6 |
| Finite rigid rotation (0.43 rad) read as deformation | 1.7e-2 m | 4e-17 m |

- **Switch off:** 182/182 identical.
- **Switch on:** catalogue A/B 87/91 both ways, no metric flips. Explicit wall time
  +32 % (120 -> 159 s): a quaternion and a few 3x3 products per bond and substep.
- **Cost reduced:** each chunk's rotation matrix, quaternion and Rodrigues
  coefficients are now computed once per substep (`bond::ChunkTurn`), not once per
  bond. The quaternion log takes a series for small rotations (to round-off). On
  b5_wall_impact_v10 at 6 threads (off 15.8 s) the overhead falls from +42 % (22.5 s)
  to +27 % (20.0 s), with the same substeps and broken bonds. Switch-off outputs are
  bit-identical.
- Every conservation test passes except the small-strain ledger test, now pinned to
  the small-strain map.
- The static and implicit solves keep the small-strain map. Their tangents assume it,
  and at their rotations the difference is second order.

**Suite with it on: 132/133.** The failure is
`an_impact_converges_as_the_substep_shrinks_and_does_not_depend_on_the_frame`. Its
check is strict: over safeties 0.5 to 0.0625, each change in the ram's final velocity
must be smaller than the one before.

| Sequence | Changes (m/s) |
|---|---|
| Co-rotational, with fracture | 0.0186, 0.00432, 0.00439 (fails by 1.6 %) |
| Small strain, with fracture | 0.0110, 0.0068, 0.0044 |
| Small strain, no fracture | not monotonic either |

- The measure is noisy at this level in both models.
- One more halving (safety 0.03125) settles the co-rotational run: 2.7620 -> 2.7618 m/s,
  a change of 0.00023. It converges, with broken bonds 108 throughout.
- **Kept off by default.** Turning it on needs the test's sequence extended to five
  points (a test-design decision), and it costs 32 % in the explicit solve.

**Into the static and implicit solves (tried, reverted).** The co-rotational
deformation and loads went into the residual, the damage commit, the dashpots and the
refinement, with the small-strain stiffness kept as Newton's tangent. Switch off:
182/182 identical. Switch on:
- **The cantilever's exact-linear targets fail by design.** The tip moves from
  -9.609000e-3 to -9.608772e-3 m and the root moment by 1.4e-5. That is the bending's
  geometric nonlinearity at 1e-2 rad, against targets that are the linear discrete
  solution.
- **The static solve loses convergence at round-off.** Gravity prestress stalls at a
  relative residual of 1.8e-10, and the villa has 3 unconverged frames.
  - The tangent lacks the geometric stiffness (the derivative of the co-rotational
    transpose).
  - `residual_floor` models rounding for the small-strain map.
- Implicit A/B 80 -> 73/91 and quasi-static 53 -> 39/91. Among the regressions: b5's
  failure mode and b7's breach in quasi-static, a consequence of the unconverged solves.

What it needs: the consistent tangent (geometric stiffness of `CoRotation`) and a
co-rotational rounding model. Exact-linear targets then need tolerances for the
geometric nonlinearity. Open; the patch is not committed.

**Contact bound on hulls** (`contact_lattice ... hull | hull_rotated`): equal cubes as
convex hulls, through the hull contact path, behave as the boxes do. Aligned: stable to
safety 1.0, blows up at 1.1. Rotated: stable throughout.
Blast loads and dropped bodies have no `t_c` here yet.

GPU-only items are out of scope for the CPU oracle: shared-memory windows, cooperative
launch, temporal blocking, Morton order, CUDA graphs.

## Open (from the audit and `CPU Oracle Improvements to Prove First`)

| # | Change | Class | Status |
|---|---|---|---|
| 0 | Instrumentation: work counters, binding limit of the step, adaptive decisions | - | done (`--profile`; adaptive decisions not yet counted) |
| 1 | True stable step (power iteration on the explicit update itself) and a per-frame contact step from the contacts that can engage; stability harness (`examples/stability.rs`) | Step change | done: `true_step` (certified Collatz-Wielandt bound, switch), `frame_contact_step`, `lattice_contact_step` (default) |
| 2a | Joint contact patch only after first damage | Model change | default on: `patch_after_damage` (DECISIONS.md 53) |
| 2b | Cold-bond fast path (exact variant first) | A, then bounded | exact variant measured slower, rejected (see Runtime) |
| 3 | Direct (sparse Cholesky) solve for intact islands | Exact | done: banded Cholesky with RCM reordering, dense LU for the full tangent (`tangent_statics`) |
| 4 | Sleeping with closed-form fatigue | B | to do |
| 5 | Energy certificate for settling and waking, shadow first | C | to do |
| 6 | Spanning-forest split test; split, drift and rigid-acceleration cadences; per-island substeps | A to bounded | to do |
| 7 | Multigrid statics prototype | exact to tolerance | to do |
| 8 | f32 mirror of the hot kernels (not the whole crate) | bounded | to do |
| 9 | Damping study (measure first) | Model change | done: global modes were almost undamped (stiffness-proportional only); `rayleigh_damping` default (DECISIONS.md 56-57) |
| - | Strain-rate filter exact (`1 - exp(-dt / tau)`) | Fix | default on: `exact_rate_filter` (DECISIONS.md 53) |
| - | Joint contact springs at the patch edges (outer spring at 0.417 w, rocking 17 % low) | Fix | default on: `exact_joint_patch` (DECISIONS.md 53) |
| - | Adaptive wake on per-bond utilisation, not summed force norms | Fix | not needed on the evidence: adaptive matches explicit on the gate's adaptive cells, insensitive to the wake threshold (0-0.02); the keystone gap is chaos (DECISIONS.md 55) |
| - | Settle and refinement energy booked, not discarded | Fix | done: `static_discard`, net of the loads' work (DECISIONS.md 54) |
| - | `split_release`: hand the joint's compression, shear and torsion to the contact (with `layer_contact`) | Fix | done |
| - | Test targets audited: exact answers or derived bounds, no picked tolerances (DECISIONS.md) | - | done for principles, conservation, gradients, ball overlap |
| - | Exact rigid/hidden coupling of the floating frame (angular momentum, fracture energy closure) | Model change | `exact_frame_transfer` (splits, default) and `corotational_bonds` (bonds, switch): angular momentum converges to zero; fracture closure -2.4e-3 -> about +5e-4 `e0` |
| - | Half turns bit for bit (covariant quaternion formulas or matrix orientation) | Fix | done: `a_half_turn_of_the_scene_turns_the_result_bit_for_bit` passes about x, y and z, with and without fracture |
| - | Shallow ball caps without cancellation (centroid off ~1e-4 m on a 1e-16 m^3 cap) | Fix | done: the cap depth in closed form (`shallow_sphere_cap_depth_is_exact`); the centroid is `centre - n from_center`, rounding relative to the radius |
