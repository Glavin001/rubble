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

**A car against itself.** A car's own pairs that share no node take part. Pairs under 2 mm at
the first step are seam neighbours and are left out (self contact's spawn rule). The skin
authoring keeps them at 2 mm or more: seam bridges, hole caps and belly faces are rejected
closer (`rig_fill::SKIN_CLEARANCE`). The sedan's own pairs under 2 mm at rest went from 14 to
2: the rear doors' edge node is 1.2 mm from the shell at the shut line, to fix in the door
generator.

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
- Seam neighbours under 2 mm at spawn are left out.
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
   within one, excludes pairs sharing a node, and applies the spawn exclusion as a per-vehicle
   bitset built once on the host.
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

**Conclusion.** At one 2 kHz step a pass, one workgroup a group, a crash between two cars
cannot be real time on this design: the response alone is ~130 dependent phases a step.
Real time needs a change of design, which is the owner's decision. Candidates:
- the response every few steps, with the paths over those steps, in place of every step;
- the response spread over many workgroups with far fewer phases (a different solver);
- continuous contact only for pairs that close fast, with the explicit penalty contact for
  the rest.
