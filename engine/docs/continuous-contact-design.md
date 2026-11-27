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
