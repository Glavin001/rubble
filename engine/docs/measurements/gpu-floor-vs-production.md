# The gpu-floor kernel against production's explicit GPU path

A review written 2026-11-20, after the experiment in `engine/experiments/gpu-floor` stopped
(PR #137). It compares that experiment with the engine's own GPU code for explicit
node-beam soft bodies, says what the experiment taught, and says what of it can go into
production and how.

- **The experiment**: hand-written Metal kernels for one car, the lab's generated sedan
  (548 nodes, 4,251 beam records, 528 collision triangles), checked bit for bit against the
  CPU reference (`engine-softbody`). Nothing depends on it. Its own write-up, with every
  measurement, is `vehicle-perf-first-principles.md` §8.39.
- **Production**: the Slang kernels in `engine/slang/kernels/explicit_beams*.slang`, planned
  by `crates/engine-exec/src/explicit_plan.rs` and run by
  `crates/engine-webgpu/src/explicit_beams_gpu.rs`, read at `main` 96904a44. Line numbers
  below are at that commit.

Terms used throughout:

- A **frame** is 34 substeps of 0.5 ms (1/60 s of simulated time). All times are ms a frame
  on a native M3 Max unless marked.
- A **workgroup** is the set of threads a compute kernel runs together (Metal's
  threadgroup); a **lane** is one of its threads; a **warp** is 32 lanes that execute the
  same instruction at once. On this GPU one workgroup runs on one of 40 cores.
- **Strict** means the GPU's floats are the CPU reference's bits. **Fast** means the
  compiler may fuse and approximate; it is judged by closeness, not equality.
- A **proxy** is a rigid body's shape, pose and mass copied into the soft body's frame for
  the 34 substeps.

## 1. Summary

**The gap.** On the scenes both sides have, the experiment's strict kernel is 1.7 to 3.9
times faster than production, and it is bit for bit with the same CPU reference:

| scene | production (scenario matrix) | gpu-floor, strict | gpu-floor, fast | production / strict |
|---|---|---|---|---|
| one car, nothing happening | 3.22 (drive) | 1.22 | 1.06 | 2.6 |
| one car into a wall, mid-crush | 4.11 | 2.22 | 1.74 | 1.9 |
| two cars head-on, mid-crush | 15.41 | 3.99 | 3.14 | 3.9 |
| a car into a parked car's side | 9.30 | 4.00 | 3.21 | 2.3 |
| a car into rigid bricks (node contact) | 3.40 (60 bricks) | 2.00 (24 bricks) | 1.80 | 1.7 |
| rigid rocks onto a parked car | 3.81 | 1.97 | 1.62 | 1.9 |

The rows are not the same frames (§2 says what differs), so read the ratios as sizes, not
as a benchmark result.

**Why.** Five differences carry almost all of it:

1. **Where the state lives.** Production re-reads every beam and node from device memory
   every substep. The experiment holds a beam's hot words and a lane's node in registers
   and passes one scalar a beam end through workgroup memory.
2. **Fixed cost.** Production's lab schedule is about eight dependent dispatches a substep
   (243 a frame in §8.31), two of them single-workgroup kernels with internal barriers. By
   §8.31's parts its passes cost 0.7 to 1.1 ms a frame before any element is updated. The
   experiment's empty frame is 0.10 ms.
3. **Contact between two cars.** Production runs about ten dispatches a substep and passes
   everything through storage buffers. The experiment keeps each car on its own core and
   has the two workgroups meet inside one dispatch.
4. **Rare work.** Both sides learned that a phase where a few lanes run a long chain costs
   that chain. The experiment applied it everywhere; production applied it to one-car hits
   only.
5. **Code that does not run.** A kernel that holds every feature is slower on the frames
   that use none of them. The experiment picks a contact-free kernel a body a frame;
   production picks module variants once an upload.

**What to port, in order** (§6 has each in detail):

1. Small changes to production's kernels as they are: flagged contact words, the
   scan-with-a-skip audit, bounds re-read at a phase's start, and the surface pass's
   staging. Days each, bit for bit, gated by the scenario matrix.
2. The kernel choice by what is in reach, first in its guard-free form, which the
   coupling's own window already makes exact.
3. A new native frame kernel in Slang with the experiment's layout, for a body alone or
   with rigid proxies. This is the large item and where most of the 2 to 2.6 times is.
4. Two bodies in lockstep across workgroups, native first. Piles of three or more are not
   solved by the experiment.

**What does not carry over as it is.** The experiment is Metal only, one car model a
compile, 1,024 lanes, two bodies a contact at most, and without wheels. Chrome and CUDA
each need a measurement before the layout is assumed there (§7).

## 2. What is being compared, and how far the figures can be trusted

| | production | gpu-floor |
|---|---|---|
| source | Slang, compiled to WGSL, then by naga (native) or Tint (Chrome) to Metal | Metal Shading Language, written by hand |
| runs on | native wgpu, Chrome, `engine-metal`, `engine-cuda` (the last two lack some modules, known-bugs P1) | Metal only |
| vehicles | any fleet, any model, counts read at run time (`explicit_beams.slang:28-36`) | one model a compile, every count a constant |
| beam kinds | every `.truck` and JBeam kind, aero, slide nodes, triggers | the sedan's kinds |
| wheels and drivetrain | raycast wheels, tyres, drive | none |
| world shapes | ground, boxes, oriented boxes, cylinders, capsules, spheres | ground plane, axis-aligned boxes; with surface contact, boxes and spheres |
| rigid proxies | box, sphere, capsule, hull; static bodies | box, sphere, dynamic only |
| other soft bodies | contact groups of up to 64 | pairs |
| sleep, debris, pieces, grab | yes | no |
| runaway protection | every loop capped and budgeted (`explicit_loop_cap.slang`) | bounded loops and status bits, no shared budget |
| checked against | the CPU reference, by suites and same-state A/B | the CPU reference, every state word every frame |

Three cautions on the figures in §1:

- **Production's rows include work the experiment leaves out.** The matrix's car has four
  raycast wheels and the World's coupling switched on. §8.31 measured the wheels at 0.85 ms
  of a 5.7 ms frame before they were moved to their own core; there is no current figure.
  So the like-for-like gap for one car is smaller than 2.6, probably near 2. It is not
  known more exactly than that.
- **Production's figures are the matrix's baseline** (`scenario-matrix.md`, main at
  366bc068), not re-run for this review.
- **Fast is not what production runs.** Production's kernels give the reference's bits, so
  the strict column is the comparison. The fast column is what a fast mode would add (10 to
  20 %), which matters now that bit-identical is to become a test mode.

## 3. How the two are built, side by side

### 3.1 The schedule

Production plans one of five schedules a frame (`explicit_plan.rs:3240-3316`). The engine's
default and the vehicle lab's differ, and the scenario matrix runs the lab's
(`vehicle_lab/mod.rs:179-183`: `wide`, `wide_list`, `wide_fused`, `group_split` on).

| case | production, engine default | production, lab default (what the matrix times) | gpu-floor |
|---|---|---|---|
| one body alone | one dispatch a frame of `explicit_beams_frame`, a workgroup of 256 or 512 lanes a vehicle, all substeps inside | the wide passes: a thread an element over the whole fleet, dispatched every substep: nodes, two centred-damping passes, beams, list, walk, pairs, hits (`explicit_beams_gpu.rs:3497-3505`, `explicit_plan.rs:3349-3369`) | one dispatch a frame, a workgroup of 1,024 lanes a body, all substeps inside |
| bodies apart | the same, a workgroup each | the same passes, more threads | the same, a workgroup each: 40 bodies cost what one does |
| bodies in contact | the group's members in one workgroup (`explicit_beams_contact`) | once any group exists every vehicle is stepped a dispatch a substep: a member step, eight fleet-wide passes, the group's levels (`explicit_beams_gpu.rs:3242-3245`, `4606-4689`) | two workgroups in one dispatch, meeting at a shared counter every substep |
| a body and rigid bodies | proxies gathered on the GPU once a World step (`vehicle_coupling.slang:15-20`), integrated inside the soft kernel every substep | the same, in the wide passes' aux pass | the reference's proxy set in the body's workgroup, the same once-a-frame exchange |

Two things follow.

- **The experiment's schedule is the engine default's, not a new one.** One dispatch a
  frame with a workgroup a body is what `explicit_beams_frame` already does. The lab moved
  to the wide passes because, for the kernel as written, they were faster (§8.3: 5.1 to 5.4
  against 5.8 to 6.1 in free fall). The experiment shows the workgroup-a-body schedule at
  1.06 to 1.25. So the gain is in what is inside the kernel, not in the schedule.
- **A dispatch a substep has a price the experiment measured both ways.** In the
  experiment's first layout, a dispatch a substep cost +72 % (34 loads of the state). In
  production a dependent dispatch is about 2 µs, 0.4 to 0.5 ms a frame for the wide
  schedule's boundaries alone (§8.31).

### 3.2 One body's own substep

| | production | gpu-floor |
|---|---|---|
| a beam, every substep | 2 sorted static records, 2 state records and 4 node records loaded from device memory (`hotLoad`, `explicit_beams.slang:2957-2968`); its two end forces written to device memory (`:2980-2981`) | k, c, rest length, last rate and both ends in registers, four beams a lane; one scalar written to workgroup memory |
| a node's beam sum | a loop over its incidence entries in device memory (`beamSum`, `:1081-1102`) | a streamed list of (other node, scalar slot), the push rebuilt as (x_self − x_other)·g: the reference's bits at both ends |
| a node's own state | loaded from and stored to `gNodes` every substep | in the lane's registers for the frame; position and velocity published to workgroup memory |
| contact words (self-collision sums, anti-clip side memory, position steps, two-body counts) | read and cleared for every node every substep on the wide path. For a car with anti-clip and dissipative contact, as the sedan: 8 atomic loads and 11 atomic stores a node (`:1443-1456`, `:872-888`, `:930-938`; the wide node pass always passes `hits` true, `:7643-7644`) | read only by the nodes a contact wrote, four bits a node saying which |
| self-collision candidates | a kept pair list, margin 0.1 m, rebuilt through a grid counting sort (`explicit_plan.rs:502-507`, `explicit_beams.slang:4835-4842`); the sedan's late mountain reuses it on 63 to 86 % of steps; tested a pair a thread | an exact pair list in rows a triangle, a wide list rebuilt rarely and a working list filtered from it; a cheap filter in the row, the full rule a lane a candidate |
| hits | one car: from workgroup memory for 1 to 48 hits (`PERF_HITS_LOCAL`, §8.33). Cars in contact: from device memory (`PERF_SELF_HITS_LOCAL` built, off) | from workgroup memory always, a slot a hit node, in the beam scalars' array between two beam passes |
| workgroup memory | grid counts and offsets, the tested list, the pair list (`:393-396`, `:3899`, `:4116`); 29.9 to 31.5 KB of 32 with the rest, so no room for node state (handoff §3) | positions, velocities, beam scalars: 29.1 KB for the one-body kernel; lists live in device memory and are streamed |
| barriers a substep, frame kernel | 7 `sync()` on a reuse step, 12 on a build step, each a device barrier and a workgroup barrier (`:422-423`) | 7 |
| counts | uniforms and buffer words | constants (as uniforms: +6 %) |

Per phase, one car at rest, timed apart (production: §8.33's table at t 20 with wheels and
coupling in it; gpu-floor: §8.39, fast):

| phase | production | gpu-floor |
|---|---|---|
| nodes (production's includes the beam sum) | 0.85, of which 0.20 with no node updated | node pass 0.08 + force gather 0.19 |
| beams | 0.54, of which 0.35 with no beam updated | 0.24 |
| centred damping | 0.50 | scalars 0.08 + gather 0.17 |
| pair list, walk, pairs | list 1.51 (its own phases 1.28) + walk and pairs 0.51 | rows 0.27 + rebuilds 0.03 + candidates 0.09 |
| hits | 1.22, of which 0.41 with nothing to apply (0.08 of that the tyres) | 0.10 |
| a dispatch with everything left out | about 0.45 of boundaries | 0.10 |

Production's figures are each pass timed in a pass of its own, which adds about 3.4 µs a
dispatch (§8.31), some 0.1 ms a row; the experiment's are differences of whole frames.

Production's wide beam pass with real work (0.18) is already cheaper than the experiment's
beams (0.24): it uses many cores for one car. What production pays is the part of each
pass that is there with nothing to do, and the two single-workgroup kernels (list, hits).
The matrix's truss row says the same from outside: a small structure with no wheels costs
1.56 ms.

### 3.3 Two soft bodies in contact

| | production (lab: split path) | gpu-floor |
|---|---|---|
| who searches | fleet-wide passes, a thread a triangle, a part, a pair (`explicit_beams_fleet.slang:963-986`) | each body its own triangles, in its own workgroup |
| who responds | the group's workgroup applies the levels in a loop with barriers (`explicit_beams.slang:12582-12652`) | the owners in the reference's order, each on its own core, the other waiting |
| how state is shared | storage buffers between dispatches; nothing in workgroup memory across them (`explicit_beams_split.slang:11-13`) | atomic stores to device memory inside one dispatch |
| dispatches a substep, two bodies | about 10 | 0 (the frame is one dispatch) |
| candidates | found every substep (`PERF_FLEET_KEEP` is off: it ended elsewhere in Chrome) | found every substep |
| group size | up to 64 | 2 |
| measured, head-on | 15.41 | 3.99 strict |

### 3.4 A soft body and rigid bodies

The two agree on the model: proxies once a World step, free bodies inside the frame,
impulses back at the next rigid step, fixed-point sums a shape. They differ in detail.

| | production | gpu-floor |
|---|---|---|
| which shapes a node tests | a kept list of up to 16 shapes a node, rebuilt when the node leaves its skin (`explicit_beams_rigid.slang:753-773`, `:950-953`) | the active shapes, bounds first |
| a body's integration | a lane a body, over its shapes' range from the plan, skipping other bodies' shapes (`:621-631`) | a lane a body, over a list of its own shapes |
| measured, bricks / rocks | 3.40 / 3.81 | 2.00 / 1.97 strict |

Production's per-node list is the finer filter and the experiment has nothing like it. The
experiment's finding here was the integration loop (§5, lesson 3), which production has
already narrowed by the plan's range. Whether that range is always set on the lab's path
is worth one check (`gsCpl[0] & 4` falls back to every shape).

### 3.5 Surface contact

| | production | gpu-floor |
|---|---|---|
| geometry | `slang/physics/feature_query.slang` | the same file, converted to Metal as text |
| finding pairs | each element scans every shape, appending a pair with an atomic inside the loop (`explicit_beams_surface.slang:1002-1015`) | near shapes as bits of a mask, then listed from the mask |
| the closest-feature query | whole, in one lane a pair (`surfPairFind`, `:1100-1112`); a triangle's lane runs it inline against static slots (`:560-582`) | a triangle or beam against a box is staged: a lane a (pair, box edge), the kept candidates recorded, a lane a pair to finish |
| when nothing is near | no proximity test; with the coupling on, nine surface pipelines are dispatched every substep for every vehicle and return early (`explicit_plan.rs:3277-3285`) | one barrier a substep; or no surface code at all in the contact-free kernel |
| shapes | static slots of five kinds, proxies of four kinds, casts for fast proxies | boxes and spheres, no casts |

The experiment's first version was production's shape: a lane a pair, the whole query. Its
measured path from there (bricks, strict): 6.76 → 5.68 (box tests staged) → 4.79 (a body's
shapes from its own list) → 4.47 (pairs from a mask) → 4.26.

### 3.6 Arithmetic

Production compiles with Metal's fast math on (`engine-metal/src/explicit.rs:10-11`) and
gets the reference's bits by wrapping operations: an XOR with a uniform that is zero after
each product or sum, and a division and a square root each corrected with an `fma`
(`slang/physics/strict.slang`). The experiment turns fast math off and adds
`#pragma clang fp contract(off)`; plain `a*b+c`, `/` and `sqrt` then match the CPU over
4 million random inputs. Both are exact. The experiment's needs the kernel compiled by
its own host, which wgpu and Chrome do not allow.

### 3.7 What production already has that the experiment confirmed

These were arrived at independently and need no port:

- a kept pair list for self-collision (`wide_list`);
- a beam's loads issued together from sorted records (`jbeamHot`);
- hits from workgroup memory for one car (`PERF_HITS_LOCAL`);
- ordered force sums through incidence slots, not float atomics;
- empty work skipped by indirect dispatch (sleeping vehicles, empty contact passes);
- module variants without the coupling or surface contact (`_nocpl`, `EB_NO_SURFACE`),
  chosen an upload;
- the proxies' gather on the GPU with a swept broad phase, which is exactly the input the
  experiment's kernel choice needs.

## 4. Where the experiment contradicts what production's docs say

| stated today | where | what the experiment measured |
|---|---|---|
| "Threads are cheap and loads are dear", so every car belongs in the same few dispatches a substep | perf doc §8.1 | A core charges instructions (0.35 ns a warp-instruction), a memory access about ten of them, and registers nothing up to about 48 a lane. One core a body with the state in registers is 1.06 to 1.22 for the car; the wide schedule is 3.2. |
| Workgroup memory is full; node state cannot move there | handoff §3 | It can if the lists move out. The experiment spends 29.1 KB on positions, velocities and beam scalars and streams its lists from device memory. |
| 1,024 lanes is slower: wider barriers, a longer serial grid scan | `explicit_plan.rs:3676-3680` | An empty barrier phase at 1,024 lanes is 0.073 µs. The serial grid scan was that kernel's, and the pair list's rows replace it. |
| There are no spare barriers to remove | handoff §3 | Agreed, and it does not matter: seven phases are 0.5 µs a substep. Chains on few lanes are what cost. |
| The floor for one car is 0.3 to 0.5 ms | perf doc §1.3 | Not reached: 1.06 fast. The estimate priced memory at cache bandwidth; the core prices it per access. |
| Splitting a contact group across workgroups would be lossless and is the next speed-up | handoff §3, §8 | Done for two bodies, bit for bit: 3.99 against 15.41. |

## 5. Lessons

Each with its evidence, and what it means for production.

**1. Price a kernel in warp-instructions and accesses, not in bytes.** On this GPU a phase
is bound by its core's throughput: 0.35 ns for each instruction of each busy warp, about
ten instructions for a workgroup-memory or streamed device load, 20 to 80 for a scattered
device load, and a core is full at about eight busy warps. An idle warp is free; a barrier
is free. *For production:* the wide passes spread one car over many cores, which hides
this for one car and stops helping once the fleet fills the GPU.

**2. A phase in which a few lanes run a long chain costs that chain.** One warp alone runs
at 1.3 ns an instruction and 37 ns a scattered atomic, about four times what a
warp-instruction costs when the core is full. Hits were the worst case: 170 scattered words a hit. *For production:* its
single-workgroup kernels (list 1.5 ms, hits 1.2 ms, the coupling's bodies, the levels) are
this. §8.33 reached the same conclusion for one car's hits and stopped there.

**3. A lane's item must be given to it, not found by a scan with a skip.** A warp runs a
loop body once for every iteration any of its lanes takes. "Scan the shapes, skip the ones
that are not mine" ran the work once a lane, in series: 24 proxy bodies' integration was
1.0 ms a frame and became 0.09 with a list a body. An append with an atomic inside a scan
has the same shape. *For production:* §6.1 lists the loops to check. The lesson has a
limit, also measured: where every lane's turn is already the same iteration, a mask
bought 0.1 ms where 0.8 was expected.

**4. Rare work belongs on compact lists, with the cheap test in the wide phase.** A warp
pays for its slowest lane, so the full contact rule inline in a row makes every row pay
it. A filter of the rule's own cheap tests cut candidates at rest from 722 a substep to
0 to 2 with the same bits.

**5. Code that does not run still costs, through the values it keeps alive.** The compiler
keeps a value in one place for its whole life. A loop bound read at the kernel's start
loses its register as the kernel grows and is then loaded on every turn of a hot loop.
Reading bounds and bases again at each phase's start gained 2 to 7 %, most where the
kernel is largest. What it cannot fix (the lane's own dynamic state) is why a kernel with
no contact code is 0.16 to 0.35 ms faster on a frame with no contact. *For production:*
its frame kernel is far larger (one module is 1.44 M inlined lines) and reads its
vehicle's offsets and counts once at entry (`explicit_beams.slang:6868-6887`).

**6. Choosing a kernel a body a frame can be exact.** A guard of one compare a node a
substep ends the contact-free kernel at a substep boundary, and the whole kernel continues
from there in the same command buffer. Nothing is re-simulated, so it is within the
forward-only rule. A wrong plan costs time and never accuracy.

**7. What two workgroups share inside a dispatch must be atomic on both sides.** A plain
store by one workgroup was never seen by a workgroup on another core (4,095,999 of
4,096,000 reads stale). The first two-body kernel was right in the emulator and wrong on
the GPU for this. *For production:* nothing shares this way today; it becomes the first
rule of §6.4.

**8. Several things that look like wins are not.** Each was built and measured on whole
frames: position and velocity side by side (0 %), a node's lane owning three of its beams
(0 %), unordered force sums through atomics (+18 % slower), hit sums carried in registers
(+7 %), the contact rule run where a point passes its box (+5 %), what a body publishes
laid out by lane (slower).

**9. Method: only a whole-frame A/B in one batch decides.**

- Leaving a phase out of a frame mis-measures every phase later substeps depend on (one
  came out at −6 ms).
- A profile that runs a substep a dispatch overstates anything that waits across
  workgroups (2.1 ms against 1.28).
- Three changes the profile favoured made the frame slower.
- The host reading a body's shared buffer just before a dispatch adds 0.02 to 0.03 ms to
  that dispatch's GPU time with any kernel.
- Equivalent builds of a large kernel differ by 0.03 to 0.04 ms, so smaller gains cannot
  be judged on it.

Production's scenario matrix already embodies the first rule. The host-read trap applies
to any A/B on Metal's shared buffers (`engine-metal`).

**10. An emulator found every logic bug before the GPU did.** The Metal source compiled as
C++, a lane as a coroutine, a barrier as the point where every lane has finished its
phase, lanes and workgroups run in either order, every state word compared with the
reference. It found three bugs before any GPU time, one a real cross-core race, and it
cannot wedge the GPU. A full check of the kernel choice is 24 runs in a minute.

## 6. What to port

Ordered by gain for effort. "Gate" is the ladder in `AGENTS.md`: `fast-check.sh` after an
edit, the scenario matrix before keeping a change. Expected gains are the experiment's on
its own kernel unless a production figure is given; production's will differ and each item
starts with a measurement.

### 6.1 Changes to production's kernels as they are

| # | change | production today | measured in the experiment | first step | risk |
|---|---|---|---|---|---|
| a | Contact words read and cleared only for nodes a contact wrote (a few bits a node, set by whoever adds) | every node, every substep on the wide path: 8 atomic loads, 11 atomic stores (§3.2) | head-on −8 %, wall crash −11 % strict | a time-only build that skips the rolls: it bounds the gain. On the wide path the node pass runs across cores, so expect less than the experiment's: at most 0.2 ms of the 0.85 | low; the bits are one more word a node |
| b | Scan-with-a-skip audit | `surfPairList` appends inside its loop over proxies; a triangle's lane runs the whole query inline against static slots; `triangleGridWalk` and `trianglePairs` append inside bucket loops (`explicit_beams.slang:4025`, `:4148-4172`); the coupling's integration skips inside its range | 1.0 → 0.09 ms (proxies), search 0.57 → 0.17 (surface, bricks) | time each loop apart; change only those where lanes take different iterations | low, each is local |
| c | Loop bounds and bases read again at a phase's start | read once at entry of the frame, contact and member-step kernels | 2 to 7 % | the member-step kernel (5.8 ms at eight cars), one phase, then the matrix | low; needs a load the compiler cannot fold |
| d | The surface pass's box queries staged, pairs listed from a mask | a lane a pair, the whole query | bricks 6.76 → 4.26 strict over four steps (one of them the proxies' list of (b)) | put the staged form in `feature_query.slang` beside the whole query, with a test that both give the same contact on random cases (the experiment's `fqcheck` did this for 300,000) | medium: more barriers, which must be uniform for Chrome |
| e | `PERF_SELF_HITS_LOCAL` on (already built) | off: with its sibling switch −1.7 to −2 % at eight cars natively, inside the spread in Chrome (§8.36) | the experiment's hits from workgroup memory: −20 % of the head-on frame | re-measure after (a) and (c); it shares their phases | none, it exists |
| f | A wide pair list rebuilt rarely, the working list filtered from it | one kept list, rebuilt through the grid sort (5 more `sync()` on a build step) on 14 to 37 % of the late mountain's steps | head-on −9 %, wall-crash run −7 % | count rebuilds a frame on the wall and head-on fixtures, and time a build step against a reuse step | medium |
| g | Reductions done by the lanes that hold the values (atomic min and max) | to check: `vv_bounds`, the surface pass's body box | 0.02 to 0.08 ms each | read those two | low, small |

### 6.2 The kernel by what is in reach

Production chooses its module once an upload. The experiment shows a choice a body a frame
is exact and worth 0.16 ms (node contact) to 0.35 ms (surface contact) on every frame with
nothing in reach.

- **Step 1, no guard needed.** The coupling already defines what a vehicle can touch in a
  window: its proxy set. A vehicle whose proxy set is empty cannot touch a rigid body in
  that frame, by the coupling's own rule. So `cpl_plan` can write an indirect count that
  sends such a vehicle to the `_nocpl` module. Surface contact against the world's static
  shapes is separate: skipping it needs the vehicle's swept box clear of every static
  slot, or the guard of step 2. Where both hold, the nine surface dispatches a substep go
  too.
- **Step 2, the guard.** For the frames where proxies are gathered but not yet reached
  (an approach), port the guard: one compare a node a substep against a plane from the
  proxies' swept box, the dispatch ending at that substep, the full kernel continuing by
  indirect count. In the experiment 32 of 60 frames of a drive at a post were
  contact-free.
- **Measure first.** Run the matrix's drive row with the `_nocpl` twin against the coupled
  module with an empty proxy set. That difference is what step 1 gives production; the
  experiment's figure is its own kernel's.
- **Not covered by the experiment:** two soft bodies near each other, and a body among
  scattered proxies (one box holds everything it could touch; production's per-node lists
  are the finer tool).

### 6.3 A native frame kernel with the experiment's layout

This is where most of the gap for one body is, and it is a new kernel, not an edit.

**What it is.** A workgroup of 1,024 lanes a body, all substeps in one dispatch:

- four beam slots a lane with the hot words in registers;
- one scalar a beam end through workgroup memory;
- each node gathering in the reference's order from a streamed list;
- the node's state in registers;
- pair rows a triangle;
- contact work on compact lists;
- hits from workgroup memory;
- the proxies in the workgroup.

**What it needs from production.**

- *Workgroup memory re-budgeted.* The grid and list arrays leave it for device memory;
  positions, velocities and beam scalars take their place. The experiment's one-body
  kernel reports 29,120 bytes, under the browsers' 32,768.
- *Counts.* Constants are worth 6 %. A kernel with its arrays sized for the largest
  supported body and its counts as uniforms keeps "any fleet, one module" at that price. A
  module a model is the alternative, at a compile of seconds each.
- *A size limit, stated.* The layout holds at most 4,096 scalar beam slots and 4,096
  vector slots, and fewer once a larger body's positions and velocities share the 32 KB.
  A larger body must fail with a clear error or take the existing path. Splitting a body
  over two workgroups was not built.
- *Everything the experiment left out*, each on its own lanes or its own core as §8.31
  did for the wheels: raycast wheels and tyres, the remaining shape and proxy kinds, sleep
  tracking, debris, the loop caps and budget.
- *Uniform barriers for Chrome.* The experiment's `break` out of the substep loop and its
  early returns sit above barriers; in WGSL those conditions must be broadcast through
  workgroup memory (`gsUniform*`).

**How to land it.** As a new module behind a switch, native first, for a body alone and a
body with proxies. The existing kernels stay the reference on the GPU: `fast-check.sh`
with the switch compares every state word from the same state, then the matrix's drive,
tumble, wall, bricks, rocks and truss rows. Bit for bit against the existing kernels is
the bar, because both already equal the CPU reference.

**What to expect.** The experiment's 1.22 strict is for the car without wheels, with the
experiment's strict arithmetic. Production's XOR helpers and the wheels will add to it.
The owner's target for one soft body is 1 ms.

### 6.4 Two bodies in lockstep across workgroups

**What it is.** Each body keeps its own workgroup and core. Lane 0 of each adds one to a
device counter after its own barrier and waits, for a bounded number of loads, until every
group has. Each body searches its own triangles against what its partner published, and
the owners take their turns in the reference's order.

**Rules that came out of it.**

- Every shared word is stored atomically (lesson 7).
- What a body publishes lives in two slots by the substep's parity, because a body can be
  one barrier ahead of its partner.
- The part of every owner's turn that reads only the pass's start is done by all owners
  at once.

**Limits to plan around.**

- *Two bodies.* The owners' turns generalise to a group; the pairing and the meeting do
  not yet. The matrix's pile rows (8 and 32 cars) are not addressed by anything in the
  experiment.
- *Concurrency is assumed, not guaranteed.* The meeting needs both workgroups to be
  running at once. On the M3 Max no meeting was missed for 2 to 40 groups. No API promises
  it. A missed meeting sets a status bit, and that frame must fail loudly and fall back to
  the split path.
- *Chrome.* WGSL has the atomics, but nothing says a dispatch's workgroups run together.
  It needs a probe on the page (the experiment's `xsync` is the model) and the existing
  path as the fallback.

**How to land it.** After §6.3, as the contact path for a group of two on native.
Production's `pile2`, `headon`, `side` and `barrier` rows are the gate.

### 6.5 Tools and method

| what | port | why |
|---|---|---|
| The closeness report for builds that are not bit for bit: the one-frame defect and the free run's deviation, each against the reference's own spread from starts moved by a micron | yes, soon | bit-identical is to become a test mode with a fast default; a fast default needs an acceptance measure. Its kinetic-energy term misleads for a car at rest and should be changed first |
| An emulator for the kernels | a spike | it found a cross-core race with no GPU time and cannot wedge the GPU. The experiment's shim compiles Metal source as C++; production's kernels would go through naga's Metal output or Slang's C++ target. Whether that is a day or a month is not known |
| Strict by compile options instead of XOR helpers | `engine-metal` only, measure | that host compiles its own Metal, so its strict mode could turn fast math and contraction off and make the helpers plain operations. In the experiment strict division and square root cost 4 to 6 times fast ones either way. Not possible through wgpu or Chrome |
| Timed frames given their inputs beforehand | yes | the host-read trap (lesson 9) |
| The microbenchmark of the cost model | run it on each new backend | §7 |

### 6.6 Measured and not worth taking

Position and velocity side by side. A node's lane owning its beams. Unordered atomic force
sums. Hit sums in registers. A row's first candidate run inline. The contact rule where a
point passes its box. A phase fewer in the hit passes. What a body publishes laid out by
lane. The candidates' rule in two stages. Each is in §8.39 with its figure.

## 7. What the experiment does not show

- **Chrome and WGSL.** Not built. Known obstacles: 1,024 lanes needs
  `maxComputeInvocationsPerWorkgroup` raised from its default of 256; barriers need
  uniform control flow; fast math cannot be turned off; the compiler is Tint, not naga, so
  lesson 5's register behaviour may differ.
- **CUDA.** The cost model is the M3 Max's: one workgroup a core, a core full at eight
  warps. An NVIDIA block runs on a streaming multiprocessor with other throughput and
  other limits, and whether blocks of one launch run together is the scheduler's choice.
  Run the microbenchmark and `xsync` there before porting the layout. This matters
  because CUDA on Modal is a stated target.
- **Fleets.** The experiment measured copies of one body: 40 cost about one (1.21 at
  rest), 200 at rest 6.2 ms. Production has no current figure for many cars apart on the
  lab's schedule; the last in the perf doc (§1.5, 9.3 / 10.1 / 11.8 for 1 / 8 / 32)
  predates §7 and §8's work. A matrix row of 32 cars apart would settle which schedule
  wins at fleet scale.
- **Piles.** Three or more bodies in contact.
- **Wheels and the drivetrain**, and a body larger than one workgroup.
- **The unexplained factor.** A substep at rest prices at about 20 µs by the cost model
  and measures 35 to 38. Two guesses at the cause were tested and ruled out. A kernel
  built to this layout inherits that gap.
- **One fidelity question, the engine's and not the kernel's.** With surface contact the
  car rebounds from a fixed pole at 2.6 to 5.1 m/s. The kernel reproduces the reference
  bit for bit; there is no reference run that says what the rebound should be.

## 8. Suggested order

1. §6.1 (a), (b), (c) on the existing kernels, each through the matrix. A week. This also
   tells how much of the experiment's lessons survive Slang and naga.
2. §6.2 step 1, the guard-free kernel choice, with its measurement first.
3. A matrix row of 32 cars apart, and the microbenchmark on CUDA. Both decide the shape of
   step 4 before it is built.
4. §6.3, the new frame kernel, native, a body alone and with proxies, wheels included.
5. §6.5's closeness report, when the fast default is taken up.
6. §6.4, pairs in lockstep on native; then the design for groups of three or more, which
   is new work.
7. §6.1 (d), the surface pass, when surface contact is to be the default configuration.

Where to look in the experiment: `engine/experiments/gpu-floor/README.md` (how to run and
check it), `kernels/floor2.metal` (the kernel; its header is the layout), `kernels/pair.metal`
(two bodies), `src/floor.rs` (`reach_plan`, the layout's tables), `reach-emu.sh` and
`surf-emu.sh` (the emulator suites).
