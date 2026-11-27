# Intersection-free contact for node-beam bodies (design, 2026-11-26; revised from the literature)

> **Revision (after `contact-literature.md`):** the guarantee comes from Planar Divide and Truncate
> (Chen, Hsu, Ayman, Macklin, SIGGRAPH 2026, arXiv:2604.15513; in NVIDIA's open-source Newton engine),
> not from CCD rounds and impact zones. Per step:
> 1. Proximity contact as today (the dissipative law), with OGC's feature regions for the normal of
>    each point-triangle and edge-edge pair (the `EDGE_CONE=1` rule, exact once nothing crosses).
> 2. For every pair near enough to meet this step (query radius about 7-8 cm at 2 kHz), a separating
>    plane at the start of the step, between its closest points, its room shared by each side's
>    approach.
> 3. After integration, each node's step is scaled by the smallest fraction its planes allow: only
>    motion toward a surface is cut, tangential motion is free. One parallel pass, no iteration, a
>    minimum (order independent, so bit exact).
> This is a hard non-penetration constraint applied as a projection, an inelastic contact at the
> plane; it removes momentum toward the surface where it binds and never adds energy. Counters
> report how often it binds and the momentum it removes. Runtime CCD, its rounds and impact zones are
> dropped; ACCD stays as a CPU test oracle. Unverified: the arXiv plane-placement formula (followed
> Newton's code), planes moving with a pair's common motion (our extension), the cost (to measure).
> The sections below are the first draft, kept for the record.


**Goal (the owner's bar):** in any crash, car against car, a car against itself, or a rigid body
into a car, no two surfaces ever pass through each other, not even briefly. It must run in real
time on the GPU, create no energy, and be guarded by tests and authoring checks. It goes beyond
BeamNG, which accepts clipping. It is a runtime option beside today's contact, which stays as it is.

## The principle: never let surfaces cross

Today's contact (BeamNG's model) looks at positions only at each step. In a hard hit a panel
passes through another within one step, and the solver then tries to undo the overlap. Undoing
it has no single right answer: which side a node belongs on, which way crossed edges should be
pushed, how deep to follow them. Every symptom we have chased (cars locked together, jitter,
panels left inside each other) lives there.

The principled alternative, used in cloth and deformable-body simulation (Bridson et al. 2002;
Harmon et al. 2008; IPC, Li et al. 2020), keeps one invariant: **surfaces start apart and stay
apart.** Every step checks the motion over the whole step, not only its end. Contact then only
ever happens between separated surfaces, so it is well defined:

- the push direction is the line between the two closest points;
- the gap is a real distance;
- nothing is "inside", so nothing has to be undone.

## The primitives

A triangle mesh against a triangle mesh can only touch in two ways. Both are needed, and nothing
more is.

1. **Point against triangle:** a node against another surface's triangle.
2. **Edge against edge:** two edges, between their nodes.

Edge against face reduces to these two. Face against face does too: it is a set of point-face
and edge-edge contacts.

The surface is the **contact surface**: the collision triangles pushed out to the paint
(`contact_offset`), so what you see is what touches. The same machinery serves a car against
another car and a car against itself. Self pairs that share a node are excluded; nothing else is.

## Each step (explicit, 2 kHz)

1. **Predict.** Integrate forces as today, giving each node a start `x0` and an end `x1`.

2. **Proximity contact (resting and sliding).** Pairs closer than a thickness `d` get a contact
   along the closest-point direction. This is the existing dissipative law (§37), unchanged.
   `d` is a property of the contact surface: the visible gap at rest. It must be large enough
   that the solver does not try to keep the surfaces too close.

3. **Continuous collision detection (CCD).**
   - **Broadphase:** the swept box of every primitive over the step.
   - **Narrowphase:** each candidate point-triangle and edge-edge pair is tested for a time of
     impact `t*` in [0, 1] at which the gap reaches `d`. That is a gap of `d`, never zero, so the
     surfaces never touch exactly and floating point cannot let one through.
   - **Method:** a conservative CCD such as additive CCD (Li et al. 2021). It may report an
     impact too early, never too late. No impact can be missed: that is a guarantee, not a
     tolerance.

4. **Resolution (inelastic).** Every pair that hits gets an impulse along its normal at `t*`
   that stops the approach: it removes approach speed and never adds any, the same principle as
   §37. Friction is limited by the normal impulse. Impulses are shared over the primitive's nodes
   by their weights and masses, Jacobi with mass splitting on the GPU. The CCD is then repeated
   over the corrected motion, for a few rounds.

5. **Failsafe: impact zones** (Bridson 2002, Harmon 2008). Pairs still colliding after the
   rounds form an impact zone, the connected group of nodes involved. Each zone's motion over the
   step is replaced by its common motion (momentum and angular momentum kept), so nothing in it
   can pass through anything else in it.
   - It only removes relative motion, so it creates no energy.
   - It is rare and local: a hard hit's few fast nodes for one step.
   - It is what turns "no crossing" from a likelihood into a guarantee.
   - The rounds and the zone size are an explicit engine-wide setting. Exceeding the setting is
     reported as an error, never silently degraded (AGENTS.md).

6. **Rigid bodies** take the same CCD against their convex shapes (surface contact already has
   swept casts).

## What stays, what goes

- **Kept:** beams, the dissipative contact law, contact offsets and the paint-depth measures.
  So is today's discrete contact, as the other option.
- **Not used in this mode:** anti-clip's side memory, the crossing depth (`EDGE_DEPTH`), the
  edge-normal-cone question and the contact band's role as a last line of defence. They exist
  only to recover from crossings, and this mode does not cross.

## Prerequisites (authoring)

The invariant has to hold at the start. The generated sedan's collision skin crosses itself 60
times at rest (the vehicle lint's `skin_self_crossing`), and the other designs fail too. Those
must be zero before self-contact CCD can work. The lint's rule becomes a hard gate for any car
that uses this mode. Tearing must keep the mesh free of crossings: a broken triangle is removed,
which is fine.

## The runtime option

`contact: discrete | continuous`, an engine-wide setting (`ExplicitBeamsWorld`, the World's
vehicle definition, a lab control).

- Switching takes effect at a build or restart, not mid-run: a run in discrete mode may already
  have crossings, which continuous mode cannot accept.
- Both modes share the broadphase and the contact law. Continuous adds its own passes.

## Proof, by stage (each stage committed; nothing default until it passes)

1. **The design and the start state.** This document. The lint's crossings at rest go to zero
   for the generated sedan.
2. **CPU prototype, car against car.**
   - Tests: no edge passes through a triangle at any step, in every acceptance scene. That is
     the depth measure at every sample, not only once settled.
   - No energy rise; no jitter (the `JITTER` measure); cars part and slide as today.
   - Pair counts and cost measured early.
3. **CPU, a car against itself, and rigid bodies.** The same invariant tests, self versions.
4. **GPU in Slang.** GPU == CPU bit for bit with bit-exact on; the scenario matrix A/B on CUDA;
   Chrome check of every module.
5. **Performance:** to real time in the lab's scenes, then the option offered in the lab.
   Default only by the owner's decision.

## Cost, honestly

- **CCD per pair:** a few dozen floating-point operations per test, iterated.
- **The broadphase:** close to today's, plus swept boxes.
- **Impact zones:** rare.
- **Estimate:** contact cost about 2-3 times today's in a crash, and about the same at rest.
  This will be measured at stage 2, before any GPU work.

## Stage 2 results (CPU prototype, car against car, 2026-11-26)

`CONTACT_MODE=continuous` (`explicit_contact/dat.rs`). Each step, on every point-triangle and
edge-edge pair between two vehicles within `r_q` = 8 cm at the step's start (collision
triangles, the node mesh):

1. **Speculative inelastic contact** on velocities, Jacobi with mass splitting, 8 rounds
   (`DAT_ITERS`): each pair's closing speed is cut so its gap ends at its paint thickness
   (the two sides' contact offsets at the closest points), or at its start gap when already
   closer. It removes approach and never adds separation, and passes momentum to the struck
   side.
2. **Planar-DAT** as the failsafe (γ = 0.85). A pair whose plane cannot be certified (gap
   within rounding) stops only its nodes moving toward the other side; nodes moving away keep
   their step.

What was learnt on the way:

- **Truncation alone fails.** It stops a node at 85 % of its room each step, so a pushed pair's
  gap decays geometrically. Head 60 km/h: 20,574 pairs went uncertified, the cars froze into
  each other, and 25 mm of paint was still inside at the end. The response must hold the gap;
  truncation only guarantees it.
- **A position-level speculative pass created energy** (+3.9 MJ at head 100), because a step's
  displacement includes the contact's position shifts. The velocity-level form only removes
  energy.
- **Freezing every node of an uncertified pair locks cars together for good** (hood nodes
  frozen for 2,900 of 3,000 steps). Freezing only the approaching nodes lets them part.

Generated sedan, 1 run each, 1.5 s, edge contact on. Depth in mm (worst / after 0.3 s);
crossings are the most edges of one car through the other's triangles at any check (every
20 ms).

| crash | continuous depth | crossings | jitter m/s | discrete depth | crossings | jitter |
|---|---|---|---|---|---|---|
| head 100 | 9 / 0 | 0 | 0.24 | 0 / 0 | 52 | 2.58 |
| head 60 pressed | 5 / 0 | 0 | 0.79 | 22 / 0 | 56 | 2.72 |
| side 60 pressed | 0 / 0 | 0 | 1.40 | 11 / 11 | 0 | 1.70 |
| side 90 | 16 / 0 | 0 | 1.31 | 23 / 4 | 0 | 1.05 |
| head 60 | 0 / 0 | 0 | 0.66 | | | |
| head 30 pressed | 0 / 0 | 0 | 0.18 | | | |
| side 30 | 0 / 0 | 0 | 0.14 | | | |
| side 60 | 0 / 0 | 0 | 1.19 | | | |

Every row passes the adopted bar; the speculative pass removed between 0.8 and 122 kJ in
each. The side rows' jitter includes the struck car's spin, since the measure is about its
centre of mass. Pairs: about 16 a step at head 60, a few a step in side hits. CPU wall time,
head 60 for 1 s: 24 s continuous against 19 s discrete. The prototype rebuilds hash grids
every step and is unoptimised.

Open:

- 8 runs per crash (the bar's count).
- Uncertified pairs still occur at head 100 (981 node freezes).
- A car against itself: its own paint inside its interior solids, up to 52 mm at head 100.
  Stage 3.
- Rigid bodies.
- A test in the suite (the mode as a parameter, not the environment).
- The GPU port.

## Stage 2-3 revised: CCD with inelastic impulses (2026-11-26)

The owner ruled truncation out: it scales a node's step and speed, so momentum is lost where
it binds, and it cut 3,002 node steps of a car simply driving. The Planar-DAT planes remain
only as a detector (`DAT_ENFORCE=1` applies them, a measurement).

The guarantee now comes from physics (`explicit_contact/ccd.rs`, after an expert review):

1. **Contact pass.** The speculative inelastic contact holds each pair at its thickness: its
   paint between cars, 2 mm within a car. It uses mass splitting (Tonge 2012), so every
   pair's impulse is equal and opposite.
2. **CCD response.** Additive CCD (Li 2021, 32 iterations at most) tests every listed
   point-triangle and edge-edge pair's straight paths for the first time its gap closes to
   half its thickness. Each impact gets an inelastic impulse along its normal, mass split.
   Its velocity part cancels at most the velocities' own closing, and the rest moves
   positions only. This repeats for up to 8 rounds.
3. **Impact zones.** Pairs still colliding after the rounds become rigid impact zones
   (Bridson 2002), which keep their linear and angular momentum. The velocity projection
   cannot add kinetic energy. Pairs inside one zone are not re-tested. Pairs still unresolved
   after 16 zone rounds are counted.

**A car against itself.** Every one of a car's own pairs that share no node takes part; none
is left out (owner, 2026-11-26: no seam exclusion, the authoring is checked instead). A car
whose own pairs are under 2 mm apart at spawn cannot run the mode: the CPU reference stops
with an error, and the GPU hosts' `set_continuous(true)` returns one, listing the pairs. The
skin authoring keeps them at 2 mm or more (`lint::SKIN_CLEARANCE`): seam bridges, hole caps
and belly faces are rejected closer, and the doors' edge columns sit 3 cm inside the door's
own outline at each row's height (the rear door's slanted rear edge had put `door_R?_o12`
1.2 mm over the quarter panel's skin). The lint's hard rule `skin_clearance`
(`generated_vehicle_lint`) fails a generated design with any such pair; the sedan has none
(14 before the skin rules, 2 before the door fix).

**Results.** `tests/continuous_contact.rs`, 8 runs each, generated sedan. Every crash passes
the bar with 0 crossings, between the cars and within a car, at every 20 ms check:

| crash | worst (mm) | after 0.3 s (mm) | jitter (m/s) |
|---|---|---|---|
| side 30 | 0 | 0 | 0.30 |
| side 60 | 0 | 0 | 0.48 |
| side 90 | 0 | 0 | 0.57 |
| side 60 pressed | 0 | 0 | 0.26 |
| head 30 pressed | 0 | 0 | 0.63 |
| head 60 | 5 | 0 | 0.88 |
| head 60 pressed | 11 | 0 | 0.17 |
| head 100 | 13 | 0 | 0.72 |
| head 130 | 34 | 1 | 0.72 |
| offset 100 (40 % overlap) | 41 | 0 | 1.12 |

**Conservation, 1 run each.**

- Unresolved impacts: 0.
- Momentum: the largest change in one step of the total linear momentum is at most
  0.0003 N·s, which is rounding.
- Kinetic energy: the response only removed it, 2.4 to 385 kJ per crash. Driving alone it
  removed nothing.

**Compromises, explicit, for the owner's approval.**

- Contact is perfectly inelastic: no bounce.
- Impact zones remove the relative motion inside a zone for that step. Momentum is kept. At
  head 100 there were 3,288 zones over 2,000 steps.
- ~~Seam neighbours under 2 mm at spawn are left out.~~ Removed (owner, 2026-11-26): no pair
  is left out; the authoring keeps them 2 mm apart and the lint and the runtime check it.
- The thicknesses: paint between cars, 2 mm within a car.

**Cost.** The CPU prototype takes about 100 s per simulated second in a hard crash. It
rebuilds hash grids every step and runs ACCD on every listed pair.

**Open.**

- Head 130 leaves 54 mm of own paint inside the interior solids, though the collision surface
  never crosses itself. Those solids' surfaces are not all collision triangles.
- Rigid bodies: reuse `explicit_surface::cast_toi` with impulses in the same solve.
- The Slang GPU port, bit exact.
- The cost on CUDA and in Chrome.
- The lab control.

## Non-rigid impact zones, compared (2026-11-26)

Rigid zones fired 1-2 times a step in hard crashes (5,220 over 2,000 steps at head 100), so
they were no longer a rare failsafe: a frozen zone stiffens the crumpling metal for that step
and removes its relative motion. The non-rigid alternative is Harmon et al. 2008's inelastic
projection (ArcSim's non-rigid zones): `ccd::soft_project`, `DAT_SOFT=N` rounds before the
rigid ones.

- Every pair still colliding after the CCD rounds becomes a constraint row, kept for the
  following rounds so a resolved pair stays resolved (re-linearised when it is hit again).
- Each round projects the step's end positions so that no row's pair is closer along its
  normal than at its impact, and the velocities so that none closes: the least change by
  mass, by projected Gauss-Seidel (32 sweeps at most, rows in order).
- Each row's push is equal and opposite along the gap's direction: momentum and angular
  momentum are kept. The velocities' set is a cone through zero, so every Gauss-Seidel step
  lowers kinetic energy: it cannot create any.
- What is still colliding after the non-rigid rounds goes to the rigid zones (the guarantee).

One run each, generated sedan, the door fix and no seam exclusion; depth = worst paint depth
(mm), zones = rigid zones over the run, energy = what the response removed (kJ):

| crash | rigid only: depth / zones / energy | 8 non-rigid rounds | 16 non-rigid rounds |
|---|---|---|---|
| head 60 | 0 / 1,002 / 9.0 | 0 / 848 / 6.0 | 0 / 1,029 / 6.1 |
| head 100 | 4.6 / 5,220 / 130 | 0.4 / 2,440 / 126 | 0.4 / 1,753 / 123 |
| head 130 | 16.4 / 6,630 / 248 | 0.0 / 2,230 / 269 | 0.0 / 7,078 / 277 |
| offset 100 | 24.0 / 2,629 / 151 | 27.2 / 1,094 / 71 | 28.1 / 738 / 38 |
| side 60 | 0 / 99 / 0.07 | 0 / 41 / 0.07 | 0 / 41 / 0.07 |
| ball 100 | 16.3 / 5 / 0.7 | 19.8 / 14 / 1.2 | 19.8 / 17 / 2.7 |

Every run: 0 crossings (car to car and own), 0 unresolved, 0 from 0.3 s after the hit, the
response only removes energy, momentum changes by at most 0.002 N·s a step (rounding).

- 8 non-rigid rounds cut the rigid zones 2 to 3 times in the hard crashes and remove the
  head-on depth (4.6 -> 0.4 mm at 100, 16.4 -> 0 mm at 130). The offset hit removes half the
  energy (71 against 151 kJ): the rigid zones were damping it. Depth there and in the ball
  hit is 3 mm worse, both within the bar.
- 16 rounds are not better: the rows grow faster than 32 sweeps converge them (head 130's
  rigid zones rise to 7,078).
- CPU time, head 100 alone in the same conditions: 41 s against 48 s rigid only (fewer
  rigid zone rounds).
- Not adopted as the default yet: the GPU's response is rigid only, and GPU == CPU must hold.
  The port is next (one workgroup a group, the rows' Gauss-Seidel in order, as the CPU).

## Offset 100 during the hit: paint thickness, not skinning (2026-11-26)

`continuous_offset_100` (8 runs) fails at 56 mm worst during the hit (bar 50; 0 mm from 0.3 s,
0 crossings). Traced with `PAINT_WHERE=2`:

- Around the front of the sedan the drawn paint sits 3-6 cm outside the collision triangles
  (the contact surface's per-node gaps, 0.027-0.063 m).
- The CCD stops a pair at half its thickness (`CcdPair::xi`); the speculative pass aims for
  the full thickness but a violent hit gets past it. Two cars' collision surfaces then sit
  about half their summed paint thickness apart and their 3-6 cm paint shells overlap.
- Tried and reverted: paint bound to the collision triangle under it (BeamNG's flexbody
  practice, `SKIN_TRI`). The deepest vertex was already on its own triangle; worst rose to
  82 mm in one run.
- Measured (`DAT_XI=0.9`, the CCD at 0.9 of the thickness): 40 mm worst, passes; but the
  response removes 180-765 kJ a run against 21-253 kJ at 0.5: contact, not crumpling,
  absorbs much of the crash. Not adopted: the thickness is one of the owner's compromises.
- The authoring fix: a collision skin within 1-2 cm of the paint (thin paint shells), so the
  thickness the contact must hold is small.

## Continuous contact damps the crash (measured 2026-11-26, open)

Two generated sedans head on at 100 km/h each, 0.6 s, CPU, `vehicle_film c2c` (its SUMMARY
line; FILM_EDGE=1 the run without continuous contact, edge contact on):

| | without | rigid zones | non-rigid zones (8) |
|---|---|---|---|
| beams' permanent set (sum of rest length change) | 44.5 m | 36.6 m | 38.5 m |
| beams bent past 1 mm | 1,447 | 1,401 | 1,428 |
| beams broken | 62 + 63 | 97 + 93 | 85 + 81 |
| removed by the contact response | - | 156 kJ | 173 kJ |
| kinetic energy left | 18.5 kJ | 3.4 kJ | 11.2 kJ |

The response removes about 15 % of the crash's energy and the metal bends 14-18 % less: the
inelastic impact deletes the struck nodes' closing kinetic energy, which in metal goes into
bending the panel round the impact. Against the owner's rule (AGENTS.md, "Contact never
stands in for metal"): a bug to fix, not a compromise. (The run without continuous contact
has 52 crossings, so it is not a perfect reference.)

## Moreau's impact law, e = 1 (CPU, `DAT_E=1`, measured 2026-11-26)

`dat::restitute`: the response's velocity change Δ = v_p − v⁻ (equal and opposite impulses) is
given back e times, v⁺ = v_p + θΔ, θ = e unless the vehicles' kinetic energy would rise above
its value before the pass (an exact quadratic guard). Positions stay non-penetrating.

| | inelastic (today) | e = 1 |
|---|---|---|
| removed by contact, head 100 (2 runs) | 26 % | 2.5-3.1 % |
| removed by contact, side 60 | 8.5 % | 0.1 % |
| head 100, 8 runs | pass, 20 mm | pass, 21 mm |
| side 60, 8 runs | pass | pass, 0 mm |
| offset 100, 8 runs | fail, 56 mm | pass, 36 mm |
| jitter head 100 / offset 100 (m/s) | 0.64 / 0.73 | 0.81 / 1.01 |

0 crossings, momentum to rounding. Off by default until the GPU has it (GPU == CPU). Left: the
guard is one sum over the world (per group on the GPU); rigid proxies (the ball) are not in
Δ yet, so e > 0 there would not keep momentum; the head-on's remaining 3 %.

## Impact law on by default, CPU and GPU (2026-11-26)

`CONTACT_RESTITUTION = 1` (dat.rs; the GPU's tables carry it, `ccRestitute`): the guard per group
of vehicles the listing's near test joins, its sums in 64 fixed stripes then in order, so
GPU == CPU bit for bit (stepping tests, 80 steps). The suite, 8 runs (energy tests 2):

| | inelastic | e = 1 |
|---|---|---|
| energy removed, head 100 / side 60 / ball 100 | 26 % / 8.5 % / - | 2.9 % / 0.2 % / 1.5 % |
| head 130 | 18 mm, pass | 66 mm worst, fail |
| offset 100 | 56 mm worst, fail | 39 mm worst, 25 mm from 0.3 s, fail |
| side 90 | pass | 49 mm, pass |
| the other 9 | pass | pass |

0 crossings, momentum to rounding. The bounce squeezes the 5-12 cm paint shells harder in the
two hardest hits: the generator's paint gaps (CONTACT_OFFSET_LOG) are the next fix.
Found on the way and fixed: the CPU flagged a torn triangle in the next beam pass, the kernel
reads its beams' break marks, which a beam broken by its own load keeps pending until the next
beam update and a break group's members get at their own update (after its master in beam
order: this one; before: the next). The beam pass now records that (`explicit_beams_torn.rs`,
`ExplicitBeams::tri_torn`) and the continuous pass's surface reads it: the listing is bit for bit
at the step a triangle tears (`continuous_pairs_list_like_the_cpu`).

## The acceptance bar (owner, 2026-11-26)

- **Hard: the collision mesh never overlaps.** No edge of one car through the other's
  collision triangles, none through its own, at every 20 ms check of every run (8 runs a crash).
- **Paint: a guard.** The drawn paint is a flexbody over the coarser node-beam mesh and may stand
  a little into the other car while the collision meshes are apart (judged on close-ups of the
  head 130). It is held at or under the worst measured, so it cannot quietly get worse: 7 cm
  during a hit, 3 cm from 0.3 s after it, over 1 cm for at most 0.2 s
  (`tests/continuous_contact.rs`, `PAINT_DURING`, `PAINT_AFTER`).

All 11 acceptance crashes pass it (head 130: 66 mm during, 0 after; offset 100: 39 mm during,
25 mm after; 0 crossings everywhere).

## The impact law's cap above 1, measured (2026-11-26, not adopted)

The response's velocity change Δ is not an exact projection (mass splitting, zones), so θ = 1
leaves energy out: the head 100 loses 2.9 %. The guard's θ is the largest that keeps kinetic
energy at or under its value before the pass; letting θ go above 1 up to a cap recovers the
rest and never creates energy (`DAT_E` = the cap). 8 runs (energy 2):

| cap | energy head 100 | ball 100 | head 130 paint, during | others checked |
|---|---|---|---|---|
| 1 (default) | 2.8-2.9 % | 1.3-1.5 % | 66 mm | pass |
| 1.1 | 1.4-1.7 % | | 42 mm | |
| 1.15 | 1.0-1.1 % | | 3 mm | |
| 1.2 | 0.8-1.0 % | | 75 mm | |
| 1.25 | 0.6-0.7 % | 0.3 % | 75 mm (guard 70) | pass, offset 100 0 mm after |

0 crossings in every run. The head 130's paint depth is chaotic in the cap (3 to 75 mm), not a
trend. 1.25 is the smallest cap that puts every energy test under 1 %; it needs the paint guard
during a hit at 8 cm: the owner's decision.

## Rows with accumulated impulses, CPU reference, first measurements (2026-11-26)

`explicit_contact/ccd_rows.rs`, `DAT_RESPONSE=rows` (off by default): the CCD rounds replaced by
rows (design C of `continuous-response-parallel-design.md`), mass-split Jacobi to a fixed point
a round, e = 1, zones unchanged. The speculative pass still runs before it, apart.

| | today | rows, position row at the impact gap | rows, position row on the path |
|---|---|---|---|
| energy removed, head 100 (2 runs) | 2.8-2.9 % | 1.4-1.5 % | 3.8-4.1 % |
| zones, head 100 | ~700-1,700 | ~3,800-4,700 | ~1,200-1,900 |
| 8-run crashes (head 100/130, offset 100, side 90) | pass | pass, 0 crossings | not run |

- More iterations change nothing (128 and 512 a round: 1.3-1.6 %): not convergence.
- Holding a pair at its impact gap leaves its path closing, so it is hit again every round
  and ends in a zone; removing the path's closing (as today's rounds) keeps zones rare but the
  guard binds in two thirds of the steps: the velocity change is still not one projection,
  because the speculative pass's inelastic changes are in it.
- The speculative pass as rows too (`speculative_rows`, accumulated λ to a fixed point): 4.2-4.8 %.
  Its rows carry a bias b > 0 (closing allowed down to the thickness), and against them giving
  the change back once returns KE(v⁻) − 4·λᵀb, not KE(v⁻): exact projections do not make e = 1
  energy-neutral for speculative constraints.
- Energy removed, head 100 (ball 100), 2 runs each, by what runs (e = 1, cap 1):

  | | head 100 | ball 100 |
  |---|---|---|
  | today (speculative, CCD rounds, rigid zones) | 2.8-2.9 % | 1.3-1.5 % |
  | today without speculative (`DAT_ITERS=0`) | 1.8 % | |
  | rows without speculative | 2.6-2.7 % | |
  | without speculative, 8 non-rigid zone rounds | 1.8-1.9 % | 2.7 % |
  | speculative, 8 non-rigid zone rounds | 2.6 % | 1.3-1.5 % |

  No exact-projection variant reaches 1 %: the loss is spread over the speculative pass (about
  a third), the rigid zones and the projections' remaining inexactness. The energy-exact θ per
  group (the cap above 1, measured above) is what removes it.

## The accurate path, first steps (2026-11-26, CPU, behind switches)

Owner: no cap above 1; validate against physics, not today's solver. Measured (head 100,
e = 1): the rows rebuild every node's change from all rows' whole impulses each iteration
(no rounding build-up: linear momentum to 0.0002 N·s a step); rows, no speculative pass
(`DAT_ITERS=0`), 8 non-rigid zone rounds, 256 iterations:

- energy removed: head 100 1.1-1.9 %, ball 100 0.0 %; rigid zones 190-340 (from 560-760);
- 8-run head 100, head 130, side 60: 0 crossings, within the paint guard;
- angular momentum: 2.1-2.5 N·m·s a step (bar 2): a row's impulse acts along its impact
  normal on nodes that moved on to the step's end, so its two sides are not collinear;
- rigid zones switched off (`DAT_ZONES=0`, a measurement) remove more (3.1-3.4 %): the zones
  are not the main loss.

## Energy against physics, not against ourselves (2026-11-26)

- **Known solution:** two elastic bars head on (`tests/contact_known_solutions.rs`). Elastic
  theory: energy kept exactly. e = 0 keeps it to 0.6 J; **e = 1 as written created 403 J on
  275 J of kinetic energy** (a pressed contact bounced every step): the default is back to 0.
- **The right measure** is the pass's change of the mechanical energy (kinetic, gravitational,
  stored in the beams; `DAT_ENERGY=1`), not its kinetic change. At e = 0: head 100 −18 %, side
  60 **+10 %**, ball 100 **+8 %** of the start's kinetic energy. The pass both removes and
  creates energy: its position-only steps move nodes with no force doing the work, storing or
  releasing energy in their beams.
- So the fault is the response's form, not a tuning: contact must act through impulses on
  velocities with positions following them (x1 = x0 + h·v), or as forces in the node update
  (explicit crash codes' penalty contact). Either is energy-consistent by construction.

## Penalty contact at 2 kHz, measured (2026-11-26, `DAT_PENALTY`, off)

Explicit crash codes' soft-constraint contact: a spring k = scale·m/h² for each pair under its
thickness, forces in the beam pass, the CCD only as a backstop at `min_gap`, no speculative pass.
On the elastic bars (`tests/contact_known_solutions.rs`, exact answer: energy kept):

| scale | energy kept | CCD impacts (backstop) | zones |
|---|---|---|---|
| 0.05 | 1.024 (created) | 439 | 6 |
| 0.1 | 0.972 | 209 | 1 |
| 0.5 | unstable (a node in several pairs: their springs add past the step's bound) | | |

The springs do not stop the bars before the backstop does, and the backstop is the old
position-stepping response, so its energy errors return. The limit is the step: an explicit
spring is stable up to about m/h², so at h = 0.5 ms a contact spring lets surfaces sink about
1 cm at 10 m/s and several cm at crash speeds. Crash codes (LS-DYNA) run h ≈ 1 µs, which allows
springs ~10⁵ times stiffer. At 2 kHz penalty contact needs centimetres of sinking (a cushion,
ruled out) or the backstop doing the work.

## Subcycled penalty contact, first version (2026-11-26, `DAT_PENALTY`, `DAT_SUBSTEPS`, off)

The nodes of the listed pairs move over the step again in sub-steps (exact pair distance each
sub-step, k = scale·m/h_s², kick then drift), the beams' forces held from the step. The elastic
bars with continuous contact the only contact (`BARS_BARE=1`; with the bars' self-collision and
discrete surface and edge contact on, bars at rest gain 147-195 J with no continuous contact at
all, a separate fault):

| bars 5 m/s | energy kept (exact 1) | rebound (exact ~1) | backstop impacts |
|---|---|---|---|
| impulse response, e = 0 | 0.977 | 0.83 | 2000 |
| penalty, 32 sub-steps | 1.101 | 0.85 | 4 |
| penalty, 64 sub-steps | 1.045 | 0.86 | 0 |
| penalty, 32, bars at 20 m/s | 0.648 | 0.48 | 29 |

The contact pass alone adds 636 J (32 sub-steps). Sub-stepping only the contact nodes with their
beams' forces frozen is a splitting that is only energy-consistent when the frozen forces are
slow, and a crashing node's beams are not. Next: the contact nodes and their beams sub-stepped
together, the neighbours' positions interpolated over the sub-steps (Belytschko's mixed-time
integration, crash codes' subcycling).

Second version, measured: contact nodes sub-stepped with their linear beams (the change of
their forces since the step's start), neighbours on their straight path taking back the
beams' averaged change (momentum kept). Elastic bars, 5 m/s, continuous contact alone:

| sub-steps | 1 | 2 | 4 | 32 | 64 |
|---|---|---|---|---|---|
| energy kept (exact 1) | 0.978 | 1.017 | 1.063 | 1.751 | 1.201 |

At one sub-step it is the plain penalty step (the bookkeeping holds); the energy error grows
with the springs' stiffness over the step's: the resonance instability of multiple-time-step
integration with a stiff fast force (Ma, Izaguirre and Skeel 2003). Crash codes run contact at
their global small step rather than sub-stepping it against a long structural step. At 2 kHz
that leaves: the whole of a crashing group's step at the small step during contact (the
solver's cost times the sub-steps), or impulse time-stepping (Moreau-Jean).

## Velocity-only impulse contact (Moreau-Jean), first measurements (2026-11-26, `DAT_RESPONSE=vel`, off)

Fable's recommendation (and the owner's rule: created energy is a bug): impulses on velocities
only, positions following (x1 += h·Δv); each CCD impact a row Σ w n·v ≥ −max(s₀ − ξ, 0)/h, solved
with accumulated impulses to a fixed point; these rows take the speculative pass's place;
Moreau's law e·σ per row, σ the share of its closing that was there before the step's node
update (a pressed contact cannot bounce), the return capped so kinetic energy stays at most its
value before the solve. Elastic bars, continuous contact alone (exact: energy kept 1, rebound 1):

| | at rest | 5 m/s | 20 m/s |
|---|---|---|---|
| e = 0 | 1.0003 | 0.955, rebound 0.65 | 0.441, rebound 0.33, stuck |
| e = 1, true impacts only | 1.0003 | 0.963, rebound 0.71 | 0.469, rebound 0, stuck (826 zones) |

No energy created in any run. The loss at speed is the rigid zones gluing the bars: pairs within
their gap keep being flagged until the rounds run out. Next: why pairs within their gap stay
flagged (their rows are linearised at the impact, the closing test is not), then the zones.

## Velocity contact made exact (2026-11-27, `DAT_RESPONSE=vel`, off; `explicit_contact/ccd_vel.rs`)

What was wrong with the first version, in the order found:

1. **Pairs stayed flagged.** A pair resting within its gap was listed again every round as a
   new row, and the solve's stop test (no λ changes at all) is never met in floats. So every
   step ran its 8 rounds and fell to the rigid zones. Fixes: one row per pair, kept with its λ.
   The solve is settled when no row closes past 1 % of its gap over the step, and no pushing row
   parts faster than its bound by more than 10⁻⁴ m/s (a push past its bound does work).
2. **The measure.** Raw kinetic + elastic energy at the step's end swings with the bars'
   ringing, because ωh ≈ 1. The bars test now judges the integrator's own conserved energy
   (symplectic Euler, kick then drift): E = KE + V + (h/2)·Σ v·F(x). Without contact it is flat
   to 0.1 J over 0.2 s. For the cars, the pass's work on that energy is
   `contact_work` = Σ m·Δv·(v⁻ + v⁺)/2, v⁻ the velocity before the kick. A pressed contact does
   none; an inelastic impact does −½μv². `contact_de` also counts the beams' energy a held node
   keeps, which is not work.
3. **The solve.** Mass-splitting Jacobi needed ~2000 iterations on the bars' 100 redundant rows
   (8 end nodes). Fable's pick, APGD (Mazhar et al. 2015, Chrono): the same sweep with Nesterov
   momentum, the mass-splitting diagonal as preconditioner, a fixed restart every 16 iterations
   (no reduction, so deterministic). With the impact law below it takes about 1.5 iterations a
   round.
4. **The impact law.** With e = 0 the bars lose the end slices' inelastic impact,
   ½·μ·v_rel² = 50 J at 5 m/s (measured 52.6). That loss is inherent to lumped mass.
   - Fable first proposed Cirak–West's momentum reflection, v⁺ = v_f + (1+e)M⁻¹Jᵀλ. It created
     +95 J: v_f is the kicked velocity, so a node its beams press bounces at hF/m every step,
     which does work m·u² on the conserved energy.
   - The law adopted is Newton's on the closing before the kick: J v⁺ ≥ −e·min(J v⁻, 0), as a
     bound in the same rows (the larger of it and the gap's). Its work per row is
     ½(u⁺ − u_f)(1 − e)u⁻: zero at e = 1 and ≤ 0 below. A pressed contact (u⁻ = 0) gets no bounce.

Elastic bars, this contact alone (`VV_RANGE=0 BARS_BARE=1`), e = 1. Exact answer: kept 1, rebound 1.

| | 0.3 m/s | 1 m/s | 2 m/s | 5 m/s | 10 m/s | 20 m/s |
|---|---|---|---|---|---|---|
| energy kept (conserved E over KE₀) | 0.953 | 0.988 | 0.994 | 0.996 | 0.993 | 0.137 |
| rebound | 0.91 | 0.88 | 0.91 | 0.90 | 0.85 | 0.32 |

None is created at any speed.
- At 5 m/s the loss is the integrator's drift without contact: −1.75 J against −1.67 J.
- At 0.3 m/s the 0.09 J lost is that drift on 2 J of kinetic energy.
- At 20 m/s the bars close at 40 m/s, faster than their 37 m/s wave speed, so they are outside
  the elastic regime: they crush and stick.
- The rebound below 1 is the 11-slice bar's dispersion: the energy kept as ringing.

**Two other contact passes create energy on the bars** (pre-existing; they act within a band, a
cushion under AGENTS.md):
- The discrete node-against-triangle vehicle pass adds +0.7 J in one step to two bars at rest
  1 cm apart (`VV_RANGE=0` turns it off).
- Each bar's own node-against-triangle self collision (`candidates`) fights the rows. Together
  with the surface and edge passes, the bars at 5 m/s go from kept 0.996 to 2.89.

Fable's fix for these: retire them where the velocity rows cover the same contact, or bound
each hit's impulse to 0 ≤ J ≤ 2·m_eff·max(0, −u), which gives no impulse at relative rest.

## Velocity contact in the cars (2026-11-27, `DAT_RESPONSE=vel`, off)

Bugs found and fixed, each one a crossing or a cushion:
- **Signed end test.** A row's "ends far enough apart" test was unsigned. An engine node passed
  through the front apron within one step, and 1.1 mm on the far side passed the test. The end
  separation is now measured along the row's own normal.
- **Re-listing.** Impulses can carry a node farther than the reach its pairs and records were
  listed for. Records and pairs are now listed again after the response (`relist_velocity`).
- **Newton's law on true impacts only.** Every row with e > 0 used to get b ≤ 0, so a pair
  that wasn't closing was held where it stood, up to 3 mm apart: a cushion.

Generated sedan at e = 1, after these fixes:

| Scene | Crossings | Zones |
|---|---|---|
| Side 60 | 0 | 2,750 |
| Head 100 | 0 | 4,784 |

Momentum is kept to 1e-3 N·s. The zones are the rounds not finishing.

**Why the rounds don't finish: centred damping.** The node update moves positions with the
velocity before centred damping's correction, then the correction changes the velocity. So the
end positions are not x0 + h·v. A row's bound taken from the step start, (s0 − ξ)/h, then misses
by the correction's share. The pair ends short of 99 % of its gap, is linearised again, and runs
out of rounds.

**Measured, not adopted:** the bound taken from the rounds' start positions, (sb − min(ξ, s0))/h − J·vb.
- Side 60: zones 2,750 → 33, 0 crossings.
- Head 100: the rows must also undo the correction, tens of m/s in the crush. APGD's iterates
  reach 170–270 m/s, the fixed-point velocity sums overflow, and momentum is lost (850 N·s).
  The cars cross 7 times.

Next: reconcile centred damping with the contact pass, for example by applying the correction
before the drift or by passing the correction to the contact pass explicitly. That changes how
the integrator steps, so it is a design decision, measured on the elastic bars and the crash
suite before adoption. Also: a bound on the fixed-point sums' range, so an overflow is an error,
not wrapped garbage.

## Velocity contact: the band passes off, warm starts, a safe preconditioner (2026-11-27, later)

**Correction to the section above.** The mismatch between end positions and velocities was not
centred damping: its correction moves positions too. It was the older contact passes'
position-only steps, `shift_applied` (§37 dissipative contact).

What changed:
- With velocity contact on, the band passes it covers are off: vehicle contact, and each car's
  self collision. Kept on with their own band passes: surface contact, ground, static colliders,
  rigid proxies. Without their position steps, the band passes' forces only dissipate: the
  elastic bars kept 12 % of their energy.
- Warm starts: each pressed pair starts the step from last step's impulse. Impacts are not
  seeded: their impulse seeded into the next step overflowed the sums.
- APGD's preconditioner is Gershgorin's bound, so D ⪰ A. Mass splitting fell under A with
  barycentric weights, and the accelerated steps diverged on 744 rows.
- Impulse sums are 64-bit fixed point. The 32-bit sums wrapped 902 times at head-on 130.
- Over-push is judged by its work, λ·excess ≤ 10 µJ.

Elastic bars with every pass on, velocity contact at e = 1: kept 0.989 at 1 m/s, 0.996 at 5 m/s,
none created; rebound 0.89 / 0.90.

Generated sedan, e = 1:

| Scene | Crossings | Zones | Momentum (N·s) | Contact's work |
|---|---|---|---|---|
| Side 60 | 0 | 737 | 1e-4 | −2.9 % |
| Head 100 | 0 | 2,910 | 7e-4 | −11.5 % |
| Head 130 | 0 | 5,219 | 2e-3 | |
| Offset 100 | 0 | 1,277 | 1.5e-3 | |
| Ball 100 | 0 | 119 | | |

Open:
- The rows on a crushed front converge slowly. One impact row shared its nodes with 88 others
  and overshot its bound by 0.07 % after 1,000 iterations. The rigid zones that catch these
  steps are the energy sink: −11.5 % at head 100, against ≤ 5 %. Next: a solver for dense,
  redundant clusters (a block solve per node cluster, or removing exactly duplicate rows).
- Ball 100: the ball's centre comes 136 mm inside the door's paint, against 17 mm in the
  default mode. Next: surface contact's position steps against the velocity rows.

## Velocity contact: the full paint gap, spheres as bodies, the suite (2026-11-27, later still)

**Changes:**
- **Full paint gap.** Each pair's gap is now its whole paint thickness (`ccd_pairs`'s share 1.0).
  The default mode's speculative pass held the half the CCD's half-gap left. Velocity contact
  has no speculative pass, so with the half gap the paint stood 107 mm into the other car at
  head 100. With the full gap it is 0.8 mm.
- **Spheres as bodies.** A rigid proxy's sphere joins the solve as a body of its own: its centre
  against the car's collision triangles, the gap grown by its radius. The node pass's and
  surface contact's rules for it are off. Ball 100: paint 139 → 44 mm during the hit, 0 after.

**Acceptance suite** (8 runs a scene, generated sedan, e = 1). Every scene has 0 crossings.

| Result | Scenes |
|---|---|
| Pass | side 30, side 60, head 30 pressed, head 60 pressed, head 60, side 90, ball 100 |
| Paint passes, angular momentum fails | head 100 (2.06 N·m·s a step), head 130 (2.3), offset 100 (3.5); bound 2.0 |
| Paint fails | side 60 pressed (52 mm, lasting) |

Linear momentum is kept to 2e-3 N·s in every scene.

**Angular momentum.** A row's impulse acts along its impact normal, but at the nodes' positions
at the step's end, where the pair has turned. The two impulses then make a couple, larger with
a wider gap. Taking the normal along the pair's separation at the rounds' start positions brought
it under the bound (1.4–1.8), but head 100 crossed 10 times: **not adopted**.

**Side 60 pressed.** The lasting paint is car B's front-left door near the pillar. That paint
vertex is skinned partly to `door_FL_i12`, an inner node in no collision triangle. Contact never
holds an inner node, so paint that follows one goes where the skin does not. The fix is in the
generator's skin binding, which is visual only.

## Velocity contact: where the zones come from (2026-11-27, measured)

Head 100, e = 1, full paint gap.

**The zones are the energy sink.** Contact's work is −172 kJ of 1,117 kJ (15.4 %). With the zones
off (`DAT_ZONES=0`, a measurement) the rows alone do −57.9 kJ (5.2 %), but 7 pairs then cross.

**Why the rounds don't finish.** The stuck solves hold rows on shared nodes with nearly
opposite normals: different pairs (no duplicates: 0 or 1 a solve), whose constraints are nearly
parallel. One row over-pushes (λ ≈ 12 N·s while the pair parts 0.3–1.6 m/s faster than its
bound); its neighbour stops just short. An ill-conditioned system like this converges slowly
under every iterative method.

| Variant | Zones | Notes |
|---|---|---|
| APGD (default) | 1,972 | |
| Projected Gauss–Seidel (`DAT_PGS=1`), 32 / 128 / 512 iterations | 1,833 / 2,185 / 1,820 | |
| Over-push bound relative to the row's speed, 1e-5 (`DAT_PUSH_SHARE`) | (above) | kept: the work bound asked a 55 m/s row for 1e-6 m/s, below f32 resolution |
| Rounds stopped when nothing new collides, settled or not, APGD | 1,298 | paint 31 mm |
| Same, Gauss–Seidel | 260 | paint 82 mm, angular momentum 19 N·m·s a step |
| e = 0 (Gauss–Seidel) | 5,033 | not restitution's multi-impact paradox |

Next: a solve exact for these near-parallel clusters. Candidates: a block solve per small node
cluster (a direct LCP of tens of rows); or merging nearly parallel rows that share nodes into
one, which is lossless only if their constraint is the same.

## Velocity contact: two CCD holes, the rounds made to converge, an exact cluster solve (2026-11-27, evening)

Head 100 unless named; one run of 0.3 s for counts, the energy tests' 2 runs of 0.6 s for energy.

**What the unsettled solves are.** `DAT_CLUSTERS=1` prints, for every solve APGD leaves
unsettled, the clusters of rows joined through shared nodes. Every unsettled row there keeps its
gap. What fails is over-push: about 1e-3 J of work past the bound on near-parallel rows, in
clusters of 116–192 rows at first and up to 1,100 rows later in the crush.

**An exact solve of those clusters** (`ccd_exact.rs`, `DAT_EXACT=1`, off). It is the cluster's
LCP in f64, dense A = J M⁻¹ Jᵀ, solved by Lawson–Hanson's active set. It starts from APGD's free
set; a row whose Cholesky pivot is under 1e-6 of its diagonal is dependent and held bound. Its
answer is kept only when the whole solve settles. Sums run in node order (a BTreeMap: with a
HashMap, two runs differed). Head 100's zones fall from 5,187 to 435 in 0.3 s. It is off by
default for what it lets be seen: panels deform where the rigid zones held them, and the paint
doesn't follow the collision skin there yet. Ball 100 paint goes from 42 to 133 mm, side 90
from 0 to 86 mm, side 60 from 0 to 37 mm. Side 60's case is the door top's paint vertex, bound
to `sh_d1_ml`, `door_FL_o21` and `ebody34` across a half-metre triangle, 37 mm into the other
car's hood. It comes back on once the generator fits the paint to the skin there.

**Two CCD holes**, found with `DAT_MISS=1`, which lists every node that ends a step through one of
its own car's triangles and whether that pair was a record and a CCD pair:
- A pair starting inside its gap and parting at t = 0 was never reported, even when it turned
  and passed through later in the step. The closing test ran only at the time of first reach,
  which is 0 for a pair already inside its gap. A node 0.05 mm from its own triangle ended
  0.1 mm through it. Now such a pair is also tested against half its start distance (velocity
  contact only; the position response's kernels keep their test bit for bit).
- A listed row was skipped while its separation along its own normal stayed positive. That
  normal and those weights are from the impact, and a feature that turned since let a node
  0.5 mm through with the rounds "resolved". Now the pair's true closest gap at the end, along
  the row's normal, must pass the same test.
Head 100 went from 0 crossings to 2–12 as trajectories changed under other edits, and back to 0
with both fixes. The engine block's nodes through the front crossmember (`eng_*` through `bf_o02`,
`whefr_f3`, `whefl_f3`) were the first case.

**The rounds made to converge.**
- Two tests disagreed. The settle test allowed closing 1 % of the gap; the re-linearisation
  allowed 1 % of the start distance. A pair inside its gap settled where it was linearised again
  every round, and the rounds ran out into zones ("fresh 1, relin 1, settled true" for 8 rounds,
  `DAT_VDBG`). Now the settle test takes 1 % of min(start distance, gap), and re-linearisation
  waits for twice that (`DAT_RELIN` 0.98).
- A re-linearised row now takes its features at the step's end, where it is tested, while the
  end is on the row's side. Linearised at the end with the node already through, its normal
  pointed through: 8 crossings, found the same way.
- `DAT_CCD` now sets the rounds (`Dat::new` had 8 fixed). 32 rounds changed nothing: the rounds
  had stopped converging, they were not too few.

**Rounding.** A pushing row is also settled when its excess is within its nodes' storage. That
is 4 units in the last place of each node's largest velocity component, plus the fixed point's
unit for each row added there (`Ctx::precision`). Not the impulses' own f32 rounding (ε·Σ|λw/m|):
that grows with λ, so a solve of huge opposing impulses passed by its own size, and their lever
arms turned head 100's cars by 16 N·m·s in a step.

**Energy, measured right.** Velocity contact's measure is the contact's work on the conserved
energy, m·(v⁺ − v_f)·(v⁻ + v⁺)/2. The mechanical energy also counts the stiff beams' share of
the h·Δv position change: +265 kJ against −151 kJ of work in one head 100 run. The energy tests
assert on the work in velocity mode: none created, at most 2 % removed (the owner's target, rule
8). `CcdStats::work_rows` and `work_zones` split it, each phase's change of
g(v) = ½m(v·(v⁻ − v_f) + |v|²).

| e = 0 (default) | Contact's work, of the start's kinetic energy | Rows | Zones |
|---|---|---|---|
| Ball 100 | −2.9 kJ of 43 kJ (6.7 %) | −2.9 | 0 |
| Side 60 | −5.2 kJ of 201 kJ (2.6 %) | −5.2 | 0 |
| Head 100, run 0 / run 1 (exact solve on) | −151.5 / −239.6 kJ of 1,118 kJ (13.6 / 21 %) | −80 / −52 | −71 / −188 |

(Ball and side 60 above were with the exact solve on too; their zones were 0.)

**Restitution, tried and not adopted** (`DAT_E`; `restitute::VELOCITY_RESTITUTION` = 0 and why):
- Newton's law in every row: the bars kept 0.997 (0.905 at e = 0), but coupled rows created
  +591 kJ in a head 100 run.
- Poisson's form (inelastic solve, then e·λ_c on the impacts' rows), with a guard that solves the
  pass again with a smaller e when its rows did positive work: no energy created, but the guard
  fired in hundreds to thousands of passes and head 100 removed more than at e = 0. Impacts were
  first "closing before the kick" (a crush's pressed nodes keep closing), then "newly closing"
  (no row the step before).

**The suite with these fixes, the exact solve off** (one run each; `out/suite/run.sh`, the probe):
every crash 0 crossings and no fixed-point overflow. The paint is 0 mm during and after in all of
them but ball 100, which is 42 mm during and 0 after; it was 54 mm, and side 90 46 mm. Angular
momentum per step is at most 1.1 N·m·s (head 130), where head 100, head 130 and offset 100 failed
the 2 N·m·s bar this morning (2.06, 2.3, 3.5). Side 60 pressed, a lasting-paint failure, passes.

**Why the exact solve stays off.**
- With it, head 100's angular momentum per step rose to 3.6 N·m·s, past the bar (the owner's
  hard rule 2). A row's impulse acts along its impact normal, but the step applies it from the
  start positions, so each row's torque is the start gap's tangential part × λ. Larger, fewer
  impulses move more.
- It also makes the paint above worse.
- Without it, the rigid zones' removal stays (graded rule 8). On the elastic bars at e = 0, velocity
  contact without it falls into 355 zones and keeps 0.29 of the energy (0.905 with it).
  Reverting each of tonight's other changes one at a time, and all together, leaves 298–458
  zones, so they don't cause it.

**Energy with the exact solve off** (the default; 2 runs of 0.6 s), and with a gap-keeping
over-push accepted as resolved (`DAT_OVERPUSH=1`, a measurement, not adopted: mixed):

| Contact's work | Exact solve off | Over-push accepted |
|---|---|---|
| Ball 100 | −7.3 / −7.4 kJ of 43 kJ (17 %); zones 142 | −4.7 / −3.7 kJ; zones 6 / 13 |
| Side 60 | −9.9 / −10.5 kJ of 201 kJ (5 %); zones 2,960 / 2,390 | −7.5 / −11.3 kJ; zones 30 / 191 |
| Head 100 | −134 / −161 kJ of 1,118 kJ (12–14 %); zones 5,209 / 9,046 | −179 / −121 kJ; zones 3,374 / 3,126 |

Side 60's rigid zones did positive work on the conserved energy (+1.7 and +3.4 kJ). A rigid zone
never raises the kinetic energy, but it can raise the integrator's conserved energy, and no
zone can avoid it. Its velocity v⁺ = P·v_f is fixed by keeping the zone's linear and angular
momentum, so there is no freedom to choose. Its work m·(v⁺ − v_f)·(v⁻ + v⁺)/2 with
v⁻ = v_f − hF/m is −½|(I − P)v_f|²_M + ½·(I − P)v_f·hF. The second term is positive where the
beams push a region's deformation outward (a recoil), so a zone frozen there creates energy
(rule 1). The zones have to go, not be tuned. That needs the exact solve, made torque-free:
each row's torque is g_t × λn, g_t the tangential part of its start gap with its impact
weights. Head 100's
rows (e = 0) remove 88–120 kJ by themselves.

**Torque-free rows** (`DAT_LEVER=1`, a measurement). The step applies an impulse from the start
positions (x1 = x0 + h·v), so a row's torque is its start gap g₀ = Σ wⱼ·x0ⱼ crossed with λn. Along
g₀ the torque is zero: frictionless contact's line of action through both points, at the
positions the step uses. It is taken where g₀ is within 60° of the impact's normal.
- Angular momentum per step in head 100 (0.3 s): 3.6 → 0.36 N·m·s with the exact solve, and
  0.6 → 0.14 without it.
- Energy (contact's work, 2 runs of 0.6 s):

| | Ball 100 | Side 60 | Head 100 |
|---|---|---|---|
| Default (exact off) | 17 % | 5 % | 12–14 % |
| Torque-free rows | 20 % | 6 % | 15–18 % |
| Exact solve + torque-free rows | 7–8 %, zones 0–4 | 6 %, zones 130–386 | 19–20 %, zones 1,116–2,361 (zones −151 / −172 kJ) |

Neither is adopted: no combination meets rule 8 in all three, and each makes some crash worse.
The zones that remain under the exact solve now remove more each; they are where the next work
is.

**The acceptance suite, 8 runs each, velocity contact** (`DAT_RESPONSE=vel`, the exact solve off;
M3, `out/suite8/one.sh`, 2026-11-27): all 11 pass. In every run: 0 crossings, every impact
resolved, linear momentum to rounding, angular momentum under 2 N·m·s a step, no fixed-point
overflow.

| Crash | Paint, worst during / from 0.3 s | Wall time (8 runs, parallel) |
|---|---|---|
| Side 30, side 60, side 90 | 0 / 0 mm | 63 / 34 / 50 s |
| Head 60, head 100, head 130, offset 100 | 0 / 0 mm | 74 / 37 / 33 / 55 s |
| Head 30 pressed | 0 / 0 mm | 145 s |
| Side 60 pressed | 14 / 14 mm | 119 s |
| Head 60 pressed | 5 / 5 mm | 85 s |
| Ball 100 | 44 / 0 mm | 13 s |

**Newton's law on newly closing rows** (`DAT_E=1 DAT_RESTITUTE=newton`, a measurement, with the
impact law's guard): J v⁺ ≥ −e·u⁻ only on rows that are true impacts (closing before the kick, no
row the step before). It is the best variant measured, not adopted:

| Contact's work (2 runs of 0.6 s) | Default (e = 0) | Newton, newly closing |
|---|---|---|
| Ball 100 | 17 % | 7–9 % (zones 71–83) |
| Side 60 | 5 % | 4–6 % (zones 1,111–1,396) |
| Head 100 | 12–14 % | 7–10 % (rows −219 / −167 kJ, zones +111 / +91 kJ; 4,662–8,030 zones) |

All of them at once (`DAT_CCD=32 DAT_EXACT=1 DAT_LEVER=1 DAT_E=1 DAT_RESTITUTE=newton`): ball 6 %,
side 60 4–5 %, head 100 15–17 % (zones 1,309–1,515 taking 99–145 kJ). Worse in head 100.

The guard lowered e in 58–231 passes. In head 100 the zones' positive work on the conserved
energy (+91 to +111 kJ) is the plainest sign that the zones break rule 1 and must be replaced.

**Where the remaining zones form** (head 100, 0.3 s; `CcdStats::zones_relist`): all in the step's
first solve, none in the response's re-listed solves.
- Exact solve off: from APGD solves left unsettled.
- Exact solve and torque-free rows on: from rounds that keep meeting new pairs. 8 rounds leave
  2,361 zones, 16 rounds 2,603, 32 rounds 1,470, 128 rounds 1,119. With Newton's law on newly
  closing rows, 128 rounds leave 862, and angular momentum per step reaches 5.8 N·m·s.
  The rounds are not monotone: a re-linearised row replaces its last form, so a pair can cycle,
  and finitely many pairs don't guarantee an end. Replacing the zones needs a response whose
  iterations provably end, not more of them.
- Monotone rounds (`DAT_KEEPROWS=1`, a measurement): a re-linearised pair keeps its earlier row
  beside the new one, so the constraint set only grows.
  - Default otherwise: 4,914 zones.
  - With the exact solve and torque-free rows: 1,131 zones at 8 rounds (half of 2,361). At 32
    rounds 1,243, with the exact solve unsettled 1,425 times as its clusters grow.
  - 0 crossings, paint 11–14 mm, angular momentum per step under 1 N·m·s.
- Speculative rows (`DAT_SPEC=1`, a measurement): every pair within its gap plus both sides' paths
  is a row from the first round, linearised at the start with b = (s0 − ξ)/h ≥ 0. No help:
  4,475 zones alone, 6,812 with the exact solve and torque-free rows.

**The next direction, proposed (not started; a change of method for the owner to approve).**
Every variant measured moved the zone count, and none removed the fallback, because the rounds
have no guarantee of ending. The established method that has one is barrier contact with a
CCD-filtered line search (IPC: Li et al. 2020; GPU versions such as GIPC, 2024):
- its iterations end by construction and never cross;
- the barrier is a conservative potential acting only within a small distance d̂ (about 1 mm),
  so it removes no energy by design (rule 4: elastic, not a cushion that absorbs);
- it is an implicit solve, while the vehicles step explicitly at 2 kHz.

A first CPU prototype (`ccd_barrier.rs`, `DAT_BARRIER=1`, a measurement) minimises the incremental
potential over the step's contact nodes with preconditioned gradient steps, each limited by CCD
and backtracked. Head 100, 0.1 s: not yet a candidate.
- 185 steps hit the 200-iteration cap, and the momentum is kept only at convergence: linear
  momentum moved by 5.8 N·s in a step.
- 3,931 zones: the final positions are clear of every gap, but the CCD then tests the straight
  path from the start to them, which can cross. The barrier's own path is not that line.
- 35 s of CPU for 0.1 s.

What it needs: a Newton solve (the barrier's Hessian per pair) to converge in a few
iterations; a CCD test that matches the straight path (end positions whose straight path from
x₀ is clear, or the barrier's line search taken from x₀ along x − x₀ alone); and velocities
taken only from a converged solve.

Two ways in: an incremental potential per step on the contact set alone (the beams' forces held
from the explicit step), or the barrier as a stiff conservative force integrated by the explicit
step with a CCD step bound, which is closer to BeamNG's penalty contact but provably non-crossing.

**Open.**
- The rigid zones remain: head 60 7,902, side 90 5,192, side 60 2,969 in 1 s.
- Head 100's contact work is −12 to −14 % at e = 0, past rule 8's 5 %.
- The exact solve waits on the generator's paint at the door top and on a torque-free form of
  the rows.

## The GPU port (design, 2026-11-26)

**Where it runs.** A GPU frame normally runs each vehicle, or each contact group, for all of the
frame's steps inside one workgroup. Continuous contact needs every node's step finished before
any pair is solved: a cross-vehicle sync point each step. Only the split group path has a
dispatch boundary per step (`schedule::group_split_ops`, `explicit_plan/schedule.rs:846`). So
continuous mode runs on that path, and each step's `Member(s)` splits in two at the CPU's
`dat_start` and `dat_truncate` point:

- **nodes:** the drive and the node update with the centred correction, i.e.
  `contactMemberStepHead`'s node loop;
- **rest:** everything after it.

A new step-word bit selects the half, as `SPLIT_LEVELS` does.

**The passes between the halves**, each a dispatch in a new module `"continuous"` in
`split_modules()`. The positions at the step's start are kept in a snapshot buffer.

1. **`cc_list`.** One thread per surface node, and one per edge. They search a grid built
   from the step's start positions (the edge pass's hash-grid pattern) and write
   point-triangle and edge-edge pairs within r_q of the paint into fixed-capacity per-item
   lists. The lists are kept in ascending candidate order. Overflow sets a flag that fails
   the frame with an error; nothing is ever dropped. This covers pairs between vehicles and
   within one, and excludes pairs sharing a node (only those).
2. **`cc_spec_count`, then `cc_spec_solve`, k rounds.** These are the speculative inelastic
   contact.
   - The count pass gives each active pair's nodes an atomic count.
   - The solve pass computes each pair's impulse with the split masses and adds it per node
     into fixed-point accumulators (`gAcc`, like `shiftAdd`). Integer sums are the same in any
     order, so the result is bit exact.
   - A per-node apply pass then follows.
3. **`cc_ccd`, then the impulse rounds.** One thread per listed pair runs ACCD, with a fixed
   32-iteration cap and strict float helpers. Impacts are accumulated as in pass 2, split
   into a velocity part and a position-only part. Up to 8 rounds run; a dirty flag skips
   rounds with no impact.
4. **`cc_zone`.** Label propagation gives each still-colliding pair's nodes the smallest label
   in their component, in about log n passes. Each zone's mass, centre, momentum, angular
   momentum and inertia are fixed-point sums per label. A per-node pass then sets the rigid
   motion. It runs only when the previous rounds left an impact.
5. **`cc_detect`.** It counts would-be crossings and unresolved impacts into counters the host
   reads back once a frame. This is the health readout and the test hook.

**The CPU reference changes first**, so that GPU == CPU bit for bit:

- the same candidate order, the per-item lists;
- the same fixed-point per-node sums in place of float accumulation;
- zones sorted by label, their sums in fixed point;
- the same strict float operation order.

Then the CPU tests (`continuous_contact.rs`) are rerun to show the change kept the physics.

**Cost expectations, to measure:**

- About 6 to 10 dispatches a step, 17 steps a 120 Hz frame.
- The pair passes are proportional to the listed pairs. At head 100 that is about 200 a step
  between the cars, plus the cars' own pairs near contact.
- A frame with no contact runs `cc_list` and early-outs.

**Order of work:**

1. The deterministic CPU reference.
2. The split seam, with an empty pass between the halves. It must be bit exact against today's
   split path.
3. `cc_list`, checked against the CPU lists.
4. The speculative pass.
5. CCD and the impulses.
6. The zones.
7. The detector.
8. The bit-exact test on the generated sedan head-on.
9. The scenario-matrix A/B on CUDA, then Chrome's WGSL check.
10. The cost table.
11. The lab control.

## GPU cost, measured (2026-11-26)

Two generated sedans head on at 60 km/h each, M3 Max (Metal, native), fast arithmetic. One
frame is 34 steps. Times are ms a frame, including the readback
(`continuous_cost`, ignored test).

| | ms a frame |
|---|---|
| Split path, continuous off | ~29 |
| Continuous on, first GPU port | ~700 |
| Continuous on, now | ~92 |

Where the ~63 ms of continuous contact goes. Each block was run twice in one step: prep
resets the step's state, so the second run gives the same results, and the time it adds is
that block's cost.

| block | ms a frame | per step |
|---|---|---|
| prep | ~1.5 | ~45 µs, about one dispatch's cost |
| grid (count, prefix, fill) | ~7 | ~200 µs, mostly three dispatches |
| listing | ~13.5 | ~400 µs |
| response (one workgroup a group) | ~38 | ~1.1 ms |
| the member step cut in two | ~3 | |

**What did not help** (each measured, each lossless, none kept):

- 128 listing workgroups a vehicle instead of 32.
- 8 cell classes a query instead of 64.
- 16,384 grid buckets instead of 4,096.
- A compacted list of the pairs to test again, alone or with a flattened ACCD loop pulling
  pairs from a shared cursor.
- Zones walked as runs of a list sorted by zone.
- The round flag in workgroup memory.

**What the response's time is.** About 130 barrier-separated phases a step. Each is a short
chain of dependent device loads, run by one workgroup:

- An extra empty phase costs ~4 µs.
- An extra full ACCD pass over the CCD list costs ~60 µs.
- Capping ACCD at 4 iterations (lossy, measured only) saves ~13 ms a frame.
- Barriers alone cost nothing measurable.

A group of two cars uses one of the GPU's cores. Many groups run side by side at the same
cost, so a derby's throughput scales; one crash's latency does not.

**What is left:**

- **Lossless.**
  - A Verlet candidate list: pairs within the reach plus a skin, the exact tests rerun each
    step, rebuilt when the nodes' motion relative to their vehicle's first node passes half
    the skin. This takes most of the grid and listing, about 15 of their 20 ms.
  - One-dispatch grid builds.
- **Lossy (owner's decision).** Fewer rounds were measured only, never adopted. Times are ms
  a frame over frames 2 to 8, averaged; the default is 8 speculative and 8 CCD rounds.

  | speculative / CCD rounds | ms a frame |
  |---|---|
  | 8 / 8 (default) | ~92 |
  | 8 / 2 | ~80 |
  | 4 / 4 | ~79 |
  | 2 / 8 | ~88 |
  | 2 / 2 | ~75 |

  Fewer rounds leave more pairs to the impact zones, which pick up the work. No round count
  comes near the 16.7 ms of a 60 Hz frame. The split path alone is ~29 ms here, and the
  matrix's head-on row without continuous contact is 15.4 ms. A coloured Gauss-Seidel
  response would have the same limit: its colours are phases too.

**On CUDA (Modal, `engine-cuda` `tests/continuous.rs`, 2026-11-26).** The same head-on, ms a
frame with the readback, frames 2 to 8:

| GPU | continuous off | continuous on | its cost |
|---|---|---|---|
| T4 | ~100 | ~130 | ~+30 |
| L40S | ~39 | ~77 | ~+38 |
| M3 Max (Metal, above) | ~29 | ~92 | ~+63 |

CUDA lowers the pass's cost, but no GPU comes near 16.7 ms. On each of them, the split path
without continuous contact is already past it in this test.

**The pass every few steps, measured (CPU, reverted).** One pass every N steps, over the
nodes' paths since the last pass (`h` = N steps in its velocities):

| scene, 8 runs | every 2 steps | every 4 steps |
|---|---|---|
| ball 100 km/h | pass (27 mm, 0 crossings) | pass (23 mm) |
| side 60 km/h | pass (0 mm) | **fail**: 4 crossings |
| head 100 km/h | **fail**: 10 crossings, 3.4 m/s jitter | not reached (5-minute limit) |
| offset 100 km/h | **fail**: 6 crossings, 88 mm, 6.8 m/s jitter | not reached |

Between passes the explicit steps move nodes through each other, and the pass then sees
paths that already crossed. This design breaks the guarantee and is ruled out.

**Conclusion.** At one 2 kHz step a pass, one workgroup a group, a crash between two cars
cannot be real time on this design: the response alone is ~130 dependent phases a step.
Real time needs a change of design, which is the owner's decision. Candidates:
- ~~the response every few steps~~ (measured above: crossings at every 2 steps);
- the response spread over many workgroups with far fewer phases (a different solver);
- continuous contact only for pairs that close fast, with the explicit penalty contact for
  the rest.

**Head on at 130 km/h (expected failure), traced 2026-11-26.** The 17 mm is hood paint drawn
past the collision surface. The paint vertices are skinned to `hd_o32`, `hd_o33` and `hd_i32`.
`hd_i32` is an inner hood node on no collision triangle. When the hood folds, the inner layer
pulls the paint off the outer sheet. Contact holds the surface (no crossings in any run); the
paint is not where the surface is.

Tried and reverted: binding each paint mesh only to its part's collision-surface nodes
(`gen3::skins_with`). It made the result worse: 53 mm worst, 26 mm lasting. The outer sheet
alone fits fenders poorly, and one hood mesh still bound `hd_i32`. The fix belongs in the
skin's fit: weights for paint vertices that keep them on the surface sheet. It is not a
change of contact.
