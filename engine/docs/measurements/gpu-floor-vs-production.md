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

## Correction, 2026-11-21: work this review did not see

This review read `main` and the experiment's branch. It did not read four pull requests
that were open when it was written (all 2026-11-19, none merged), which had already
measured parts of the port on production. Two of its recommendations are contradicted by
them. Read this before §1, §6 and §8; the sections below are left as written, with a
pointer here where they are wrong.

| PR, branch, perf doc section | what it measured | what it changes here |
|---|---|---|
| #138 `perf/floor-port`, §8.42 | production by pass against the experiment's phases; one dispatch a frame; the contact words' ceiling (1 to 3 %); and a decision: two workgroups in lockstep is not safe in WebGPU (no forward progress promised across workgroups, no device-scope fence, no buffer bound as both atomic and plain) | §6.4 is a native-only path of its own (Metal or CUDA), a separate decision, not a port into the shared kernels. §6.1 (a)'s first measurement had been done |
| #139 `perf/solo-car-kernel`, §8.43 | the experiment's layout built as a Slang kernel for the core frame (nodes, centred damping, beams; 1,024 lanes, positions and velocities in workgroup memory, a scalar a beam end, hot words in registers), bit for bit with the frame kernel and the CPU, natively and in Chrome. **Slower: 2.37 ms against 1.33 natively, 2.04 against 1.58 in Chrome. No-go, off by default.** Today's frame kernel at 1,024 lanes: 1.356 against 1.362 at 512 | §6.3, "the large item and where most of the 2 to 2.6 times is", was already built for the core and lost through Slang, WGSL and naga. §1's cause 1 (where the state lives) does not hold for production's core as a cause of the gap: its core frame is 1.33 ms, near the experiment's whole frame at rest |
| #140 `docs/vehicle-floor-status`, §8.44 | production per phase and per car count against the experiment's phases | the per-phase comparison §3.2 makes from counts exists there from timings |
| #141 `perf/hits-list-floor`, §8.45 | the experiment's kernel run again on states with hits, strict: its hit passes cost 0.74 to 0.78 ms a frame with 4 to 20 hits a step (0.0005 at rest), its frame 2.22 to 2.75. Production's hits and self-collision list against that, like for like; a slot map for the hits (on by default on that branch) | §1's table sets production's rows against the experiment at rest or in its own scenes; like for like the one-car gap in the hits and the list is smaller than §3.2 says, and what is left has a measured reason there |

Since then, on `claude/port-gpu-floor-lessons-1289bd` (not merged):

- §8.40: this review's §6.1 (a), (b), (c) measured on production, none pays (§6.1's
  outcome below).
- §8.46: wgpu's injected runtime checks (bounds on every access, a forced bound on every
  loop, the integer-division guard) cost 5 to 12 % of every native scenario-matrix row,
  bit for bit without them on 20 fixtures, 12 rows and 11 suites. A measurement switch,
  off by default; Chrome cannot turn its own checks off. The same Slang straight to Metal
  (`engine-metal`) differs from wgpu without the checks by about the frame kernel's
  workgroup zeroing. This is a cause of the gap that §1 does not list: the experiment's
  Metal has no such checks and gets strict arithmetic from a compiler pragma, where
  production's WGSL pays an XOR an operation.

What this leaves of §6 and §8:

- **Still open, measured by none of the above:** §6.2 (the kernel by what is in reach:
  no code for the coupling on frames with nothing in reach), §6.1 (d) and the surface
  half of (b) (no saved state has surface contact on), §6.5's tools.
- **Open as a question, not a recommendation:** whether the experiment's layout pays on
  a native path with no injected checks and strict arithmetic by compile option
  (`engine-metal`, CUDA). #139 gives the hypothesis (the core is arithmetic-bound under
  WGSL's strict path, so a layout that trades memory for arithmetic loses); nobody has
  measured the layout there. That is the only form in which §6.3 still stands.
- **Closed for the shared kernels:** §6.3 through Slang to WGSL (#139), §6.4 in WebGPU
  (#138), §6.1 (a), (b) walk and proxies, (c) (§8.40).

## 0. Read this first

### Who this is for and what is asked

This file is written to be reviewed by someone who has seen neither the experiment nor the
work that led to it. Everything needed to follow the argument is in the file: the
background below, the experiment in Appendix A, production's path in Appendix B. The
paths and line numbers are for checking a claim, not for understanding it.

A reviewer is asked four things:

1. Are the statements about production right? Each has a file and line; Appendix C says
   which I read myself and which rest on a second reader's report.
2. Are the comparisons fair? §2 lists what differs between the two sides.
3. Does each port item in §6 hold up: does the lesson survive Slang, WGSL and two shader
   compilers, and is the expected gain honest?
4. Is the order in §8 right, given the constraints below?

The claims I think most likely to be wrong, and so most worth a reviewer's time:

- that the like-for-like gap for one car is "near 2" (§2): it rests on an old figure for
  the wheels' cost;
- that flagged contact words are worth at most 0.2 ms in production (§6.1 a): an estimate
  from counts, not a measurement (since measured: 0.01 to 0.17 ms, §6.1's outcome);
- that a vehicle with an empty proxy set can skip the coupling's code with no guard
  (§6.2 step 1): an argument from the coupling's design, not tested;
- that re-reading loop bounds helps through naga and Tint as it did in hand-written Metal
  (lesson 5, §6.1 c) (since measured: it was wrong for production, the reload survives
  both compilers and changes nothing, §6.1's outcome);
- that two workgroups of one dispatch always run at the same time (§6.4): measured on one
  GPU, promised by no API.

### Background: the system

- **Rubble** is a physics engine with one GPU `World` that runs two solvers together: a
  rigid-body solver stepping at 60 Hz, built for 100,000 bodies and more, and an
  **explicit node-beam solver** for soft bodies (deformable vehicles and structures)
  stepping at 2 kHz, so 33 or 34 substeps inside each World step. This file is about the
  second.
- **A soft body** is nodes (point masses with position and velocity), beams (spring and
  damper between two nodes, which can yield and break), collision triangles over three
  nodes, and contact points (nodes that can touch triangles). A substep computes every
  beam's force, sums forces per node, handles contact, and integrates.
- **Contact has three kinds**, and they differ in mechanism:
  - *self-collision*: a body's nodes against its own triangles;
  - *vehicle contact*: one soft body's nodes against another's triangles. The body whose
    triangles are hit is the **owner** of those hits, and the specification applies owners
    one after another in a fixed order;
  - *the coupling*: a soft body against the World's rigid bodies. Once a World step the
    rigid bodies near a vehicle are copied into its frame as **proxies**; during the 34
    substeps they move as free bodies and take the vehicle's reactions; the summed
    impulses go back to the rigid solver for its next step. By default only a node inside
    a shape is a contact (*node contact*). The opt-in *surface contact* also tests
    triangles and beams against shapes, so thin shapes cannot slip between nodes.
- **The CPU reference** (`crates/engine-softbody`) is the specification in code. A GPU
  kernel is **bit for bit** when every word of state it leaves (positions, velocities,
  forces, beam state, broken flags, contact sums) equals the reference's after the same
  frame. Contact sums are integers in fixed point so that their order does not matter.
- **The settings in play** are the lab sedan's: centred damping (a correction pass after
  the node update, two extra phases a substep), anti-clip (each node remembers which side
  of a triangle it was on), the two-body rule (a hit splits masses between the bodies),
  and dissipative contact (contact may not add energy; it also moves positions by small
  fixed-point steps).

### Constraints any port must meet

These are the project's rules (`AGENTS.md`) and its owner's decisions. They bound §6.

- **Lossless only.** A performance change may not alter results: no dropped contacts, no
  looser tolerances, no simplified shapes. A limit must be an explicit setting that fails
  with a clear error.
- **Forward-only.** No step is ever undone and solved again.
- **A fixed checking ladder.** After every kernel edit `fast-check.sh` replays saved
  states both ways and compares every state word (seconds). Before a change is kept, the
  **scenario matrix** replays a dozen saved use cases with the change on and off and
  passes only if every row is bit for bit and none is slower.
- **Native and Chrome.** Production's kernels must compile and run under WebGPU in
  Chrome, which is stricter than native (uniform control flow around barriers, 32,768
  bytes of workgroup memory, at most 256 threads a workgroup unless more is requested).
- **Targets.** One soft body 1 ms a frame; eight in contact 3 ms; a hundred-car derby as
  the goal behind them. CUDA is a wanted backend.
- **Bit-identical is to become a test mode** (decided 2026-11-19): strict arithmetic
  opt-in, a faster default. So a way to accept a non-identical build is needed (§6.5).

### References and terms

"§8.31" and the like mean sections of `engine/docs/measurements/vehicle-perf-first-principles.md`
(the perf doc). "Handoff" is `engine/docs/handoff/soft-rigid-world.md`. Where a figure is
taken from either, the figure is restated here.

- A **frame** is 34 substeps of 0.5 ms (1/60 s of simulated time). All times are ms a frame
  on a native M3 Max unless marked.
- A **workgroup** is the set of threads a compute kernel runs together (Metal's
  threadgroup); a **lane** is one of its threads; a **warp** is 32 lanes that execute the
  same instruction at once. On this GPU one workgroup runs on one of 40 cores.
- A **barrier** is the point in a kernel where every lane of a workgroup waits for the
  others; a **phase** is the code between two barriers. A **dispatch** is one launch of a
  kernel over some number of workgroups. A dispatch is **dependent** when it must wait for
  the one before it.
- **Workgroup memory** is shared by a workgroup's lanes and gone when the dispatch ends;
  **device memory** is the GPU buffers, which persist. A **register** is a value a lane
  holds itself.
- **Strict** means the GPU's floats are the CPU reference's bits. **Fast** means the
  compiler may fuse and approximate; it is judged by closeness, not equality.
- A **proxy** is a rigid body's shape, pose and mass copied into the soft body's frame for
  the 34 substeps. The **window** is that frame.
- A **hit** is one node found in contact with one triangle in a substep. A **contact
  group** is a set of soft bodies whose boxes overlap; its **levels** are its owners'
  turns in order.
- A **pair list** is the kept set of (triangle, node) pairs close enough to touch, reused
  over several substeps until nodes have moved more than its margin (the **skin**).
- The **wide passes** are production's kernels that run a thread an element over the
  whole fleet, as opposed to a workgroup a vehicle.

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
   staging. Days each, bit for bit, gated by the scenario matrix. **Tried since
   (2026-11-20): the first three were measured on production and none pays, 0 to 3.5 %
   of a frame each (§6.1's outcome). The order now starts at item 2.**
2. The kernel choice by what is in reach, first in its guard-free form, which the
   coupling's own window already makes exact.
   (Items 3 and 4 are contradicted by PRs #139 and #138: see the correction at the top.)
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
  366bc068), not re-run for this review. A later run on main at 96904a44 (the port
  session's baseline, two runs, §8.40 on `claude/port-gpu-floor-lessons-1289bd`) gave
  drive 3.80 and 3.29, wall 4.75 and 4.68, head-on 14.00 and 13.82, side 9.30 and 9.85,
  bricks 4.03 and 4.05, rocks 4.39 and 4.33. A one-car row sits at 3.2 to 3.5 ms in one
  process and 3.9 to 4.1 in the next with the same binary, so §1's one-car ratios are
  known to about 20 %; the head-on's is nearer 3.5 than 3.9.
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

**Outcome of (a), (b) and (c), 2026-11-20.** A separate session measured each on
production's kernels (main at 96904a44, the scenario matrix) before building it. None
reaches 5 % of a frame on its best row and none is kept; main's kernels are unchanged.
The write-up is §8.40 of the perf doc on branch `claude/port-gpu-floor-lessons-1289bd`
(docs only, not merged; its two measurement builds are commits c2b8b478 and 05b15b41,
each reverted). This review's estimates for these three rows were too high:

| # | measured on production | why it differs from the experiment |
|---|---|---|
| a | the node phase's whole read-and-clear of the contact words is 0.01 to 0.17 ms a frame, 0 to 2 % of a row (each word read and written once more, bit for bit) | the same words cost the experiment 1 to 2 ms a frame; production's are 5 to 10 ns a warp access. Worth building when one car's frame is near 1 ms, where the same 0.06 ms is 6 % |
| b | the proxies' integration forced over every shape: −0.2 to +0.5 %; the pair list's whole walk once more: at most +3.5 % (piles), so moving its appends gains a part of that | the plan's per-body range is set on the lab's path, and the body's loop reads workgroup memory. `triangleGridWalk`'s append, `pairReuseList` and `trianglePairs` run in no saved state; surface contact is on in none, so its loops were not timed |
| c | built (its own functions behind a perf bit, a load no compiler can fold, bit for bit on 20 fixtures and 12 rows, Chrome compiles every module): member step −0.7 to +0.6 %, frame kernel −0.8 to +0.3 % | a lane takes one or two nodes and eight beams a phase, each hundreds of instructions; one reload is too small a share. (The session's guess, not measured further.) |

Two things follow for the rest of this section. First, the small lessons do not transfer
to production's kernels one at a time: its time is in the schedule and the layout (the
member step is a third to a half of every row with cars in contact, the split levels a
third of the head-on and of pile32), which are §6.2 to §6.4. Second, (d) and the surface
half of (b) still have no production measurement, because no saved state has surface
contact on; a matrix row with it on comes before either.

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

**Already built and measured, and it lost** (PR #139, §8.43; the correction at the top):
the core frame with this layout in Slang is 2.37 ms against today's 1.33 natively and
2.04 against 1.58 in Chrome, bit for bit. What follows is the review as written; it
stands only as a question for a native path without wgpu's checks and WGSL's strict
arithmetic.

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

**Decided against for the shared kernels** (PR #138, §8.42; the correction at the top):
WebGPU promises neither that two workgroups of one dispatch are resident at once nor a
fence across them, and a software adapter (CI's SwiftShader) need not run both. It
stands only as a native path of its own, which is a separate decision.

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

1. §6.1 (a), (b), (c) on the existing kernels, each through the matrix. **Done
   2026-11-20: none pays (§6.1's outcome).** (c) survives Slang, naga and Tint as a real
   load and still changes nothing, so the experiment's small lessons depend on its
   layout.
2. §6.2 step 1, the guard-free kernel choice, with its measurement first. Still open.
3. A matrix row of 32 cars apart, and the microbenchmark on CUDA. Both decide the shape of
   step 4 before it is built.
4. §6.3, the new frame kernel, native, a body alone and with proxies, wheels included.
   **Superseded 2026-11-21:** PR #139 built its core through Slang and it was slower
   (the correction at the top). What is left is one measurement, not a build: #139's
   kernel and today's frame kernel both on `engine-metal` (no injected checks), and
   with strict arithmetic by compile option, to see whether the layout pays anywhere
   outside hand-written Metal.
5. §6.5's closeness report, when the fast default is taken up.
6. §6.4, pairs in lockstep on native; then the design for groups of three or more, which
   is new work. **Not for the shared kernels** (PR #138); a native path of its own needs
   the owner's decision first.
7. §6.1 (d), the surface pass, when surface contact is to be the default configuration.
   A scenario-matrix row and a fast-check fixture with surface contact on come first:
   there is none today.

## Appendix A. The experiment, in full

Enough to judge §3 to §6 without opening its code.

### A.1 Scope

`engine/experiments/gpu-floor` is a workspace crate nothing depends on: about 5,700 lines
of Metal in `kernels/` and 4,400 of Rust in `src/`. It steps the lab's generated sedan as
the CPU reference does, with these of the reference's features: centred damping, anti-clip
with the swept test, the two-body rule with its residual pass and work guard, fixed-point
contact sums, dissipative contact with exact friction, beam yield and break, break groups,
breakable triangles. It was built in four parts, each bit for bit before the next:

1. one body and a fixed world (a ground plane, static boxes);
2. a body against another soft body (two bodies);
3. a body against rigid proxies, two-way, by node contact;
4. surface contact (triangles, beams and nodes against boxes and spheres), then the
   kernel chosen a frame by what is in reach.

### A.2 The layout

One workgroup of 1,024 lanes a body. One dispatch runs all 34 substeps. Every count is a
compile-time constant, so the kernel is compiled for one car model.

- **Beams.** Each lane holds up to four beams. A beam's hot words (its two ends, stiffness,
  damping, rest length, last extension rate) stay in the lane's registers for the frame.
  Everything else about a beam is read only on a substep where it yields, breaks, or is one
  of the car's 71 special elements.
- **Forces.** A beam writes one scalar (force over length) to workgroup memory. A node
  reads its beams' scalars from a streamed list in the reference's order and rebuilds each
  push as (x_self − x_other)·scalar. IEEE subtraction and multiplication are
  sign-symmetric, so both ends get the reference's bits.
- **Nodes.** Each of the first 548 lanes owns a node and keeps its position, velocity and
  force in registers. Positions and velocities are also published to workgroup memory for
  the beams and the contact tests.
- **Self-collision.** A lane a triangle walks its row of an exact pair list (every pair
  the reference's box test could pass). The list has two levels: a wide one (skin 0.30 m)
  built rarely, and a working one (skin 0.05 m) filtered from it. Each pair gets the
  contact rule's cheap tests in the row; pairs that pass go on a compact list and get the
  full rule, a lane a pair.
- **Hits.** Each node touched by a hit gets a slot in workgroup memory (in the beam
  scalars' array, which is free between two beam passes). The contact rule's passes read
  and sum there, not in device memory.
- **Device memory** holds what is rare or persistent: contact sums, anti-clip memory,
  break groups, the pair lists, and an image of the hot state that starts the next
  dispatch.

A substep with no contact is seven phases. Workgroup memory is 29,120 bytes for the
one-body kernel.

### A.3 Two soft bodies

Each body keeps its own workgroup, so its own core. Metal has no barrier across
workgroups, so each group's lane 0 adds one to a counter in device memory after its own
barrier and then loads the counter, a bounded number of times, until every group has
added. A wait that runs out sets a status bit; it never did.

Each substep: each body publishes its box and top speed; the bodies meet; if the boxes are
apart that is all. Otherwise each publishes its nodes with atomic stores, they meet again,
and each body tests its own triangles against the partner's contact points (grouped in
clusters whose boxes the partner publishes). Hits are then applied owner by owner in the
reference's order. The part of each owner's turn that reads only the state at the pass's
start is done by both bodies at once; only the residual pass and the work guard wait for
the other owner.

### A.4 Rigid proxies and surface contact

The reference's proxy set lives in the body's workgroup for the frame. Each substep the
shapes are posed, a node tests the shapes whose bounds it is inside, a node in contact
takes the two-body rule against that body, reactions are summed per shape in fixed point,
and a lane a body integrates them.

Surface contact adds a pass of seven stages after the self hits: the body's box against
the shapes' boxes; a search for (element, shape) pairs; a cull and the direct tests
(points, and anything against a sphere); then for a triangle or beam against a box, three
staged steps (a lane a box edge computes candidate gaps, the candidates at the least gap
are walked again and recorded, a lane a pair finishes the query); then a lane a hit applies
the response. The geometry is production's own `feature_query.slang`, converted to Metal
as text, and checked against the Rust twin on 300,000 random cases with no difference.

### A.5 The kernel chosen by what is in reach

Three pipelines are compiled from the one source:

- *the whole kernel*;
- *the contact-free kernel*: no code for proxy shapes or surface contact. It keeps the
  ground, the world's boxes by node, and self-collision;
- *the kernel that takes a frame up*: the whole kernel, starting at a given substep.

Before a frame the host puts everything the body could touch in that frame in one box
(each shape as posed, widened by its turning, swept along its free flight) and gives the
kernel one plane: the face of that box the body is farthest outside. The contact-free
kernel is chosen when the body is clear of the plane by a margin (four times what its
fastest node moves a substep, 2 cm at least) plus what it would travel in the frame.

The **guard** makes the choice safe whatever the host computed. At each substep's end
every node compares its distance to the plane with the margin. If any node is within it,
the dispatch ends at that substep and writes how many substeps it did. The next dispatch
in the same command buffer is the take-up kernel, launched by indirect count, which is
zero unless a guard asked. Why this is exact:

- node contact at a substep tests positions from the substep's start, which the guard
  checked at the previous substep's end;
- surface contact's first test is the body's box after the node update, and the
  contact-free kernel checks there that every node is still on the clear side. If that
  second check fails the frame is flagged as not proven. No run set the flag;
- the proxies take no reaction while nothing touches them, so the contact-free kernel
  moves them by free flight, with the reference's arithmetic.

Nothing is computed twice and nothing is undone.

### A.6 How it was checked

- **An emulator.** The Metal source is compiled as C++ behind a small header. A lane is a
  coroutine, a barrier is the point where every lane has finished its phase. It runs
  against the CPU reference and compares every state word after each frame or substep,
  with lanes and workgroups in either order. No GPU is involved.
- **Strict on the GPU.** Fast math off plus `#pragma clang fp contract(off)`. With fast
  math off alone Metal still fuses a·b + c, and 9 % of results differ from the CPU's; with
  the pragma none of 4 million did. Every scene of every part matched the reference in
  every state word every frame: drops, a 16 m/s wall crash over 120 frames, head-on, side
  and rear impacts, 24 bricks, four falling rocks, a loose post, a fixed pole.
- **Closeness, for the fast build.** Two measures, each against the reference's own spread
  when its start is moved by a micron: the error after one frame from the reference's
  state, and the drift of a free run. The fast build is inside that spread on position;
  its kinetic-energy measure misleads for a car at rest.
- **Timing.** Every figure is GPU time of a whole frame from a fixed saved state, medians,
  two builds side by side in one batch under the machine's GPU lock. A state repeats to
  1 %; equivalent builds differ by 0.03 to 0.04 ms.

### A.7 The cost model it measured

Microsecond figures are for one phase of one workgroup of 1,024 lanes.

| what | measured |
|---|---|
| an empty phase | 0.073 µs |
| 128 dependent multiply-adds with 1 / 8 / 32 warps busy | 0.24 / 0.42 / 1.51 µs |
| the same work in 16 of 32 warps | half |
| a workgroup-memory load | 2.5 to 4.9 ns a warp |
| device loads, streamed / scattered | 3.9 / 23 to 29 ns a warp-load |
| a device atomic add, own word / scattered | 2.6 / 38 ns |
| values a lane keeps and updates each phase: 16 / 48 / 80 / 112 | 0.19 / 0.32 / 1.1 / 5.0 ns an update |
| square root or division, fast / strict | 0.7 / 2.7 to 4.1 ns a warp |
| a meeting of workgroups on different cores | +0.1 to 0.7 µs, none missed for 2 to 40 groups |
| a plain store read by another workgroup in the same dispatch | stale on every read; atomic store: right on every read |

### A.8 Results beyond §1's table

- **The kernel choice** (fast | strict): parked by a post with surface contact
  1.41 → 1.07 | 1.58 → 1.22; with node contact 1.23 → 1.07 | 1.40 → 1.21; a build with no
  proxy code at all 1.05 | 1.21. A 60-frame drive at a post, mean 1.95 → 1.78 |
  2.34 → 2.17. Frames in contact unchanged.
- **Surface contact in contact** (fast | strict): a loose post 3.40 | 4.24; 24 bricks
  3.36 | 4.10; four rocks 2.40 | 2.93.
- **Copies in one dispatch** (fast): 40 bodies at rest 1.21, 41 bodies 2.31, 200 bodies
  6.23; 20 head-on pairs about 4.5, 60 pairs about 14. A step up every 40 workgroups, one
  a core.
- **The CPU reference** on one core: about 10.5 ms a frame.

## Appendix B. Production's path, in full

Enough to judge §3 without opening its code. All of it is at `main` 96904a44.

### B.1 Data

State is in device buffers shared by every vehicle: per node three records of four floats
(position and mass, velocity and friction, force); per beam static records (ends, kind,
stiffness, damping, bounds) and two state records; per node a list of **incidence
entries**, one per beam end, each holding that beam's force on that node; integer
accumulators per node for contact sums, anti-clip memory and position steps; lists and
grids for the searches. Counts and offsets per vehicle are read from a table at run time.
Arithmetic that must match the CPU goes through helpers that stop the compiler fusing or
reordering (`slang/physics/strict.slang`). Every loop is capped and draws on a per-frame
budget, so a corrupt count cannot run away (`explicit_loop_cap.slang`); a stopped loop
fails the step.

### B.2 The frame kernel (the engine's default for one small vehicle)

`explicit_beams_frame`: a workgroup of 256 or 512 lanes a vehicle, one dispatch a frame,
the substep loop inside. Lanes stride over the vehicle's nodes, beams and triangles. A
substep is a node phase (sum incidence forces, contact with ground and shapes, integrate),
two centred-damping phases, a beam phase (force, yield, break; the end forces written to
the incidence entries), the pair tests, and the hits' application, with barriers between.

### B.3 The wide passes (the lab's default, and what the matrix times)

Small kernels dispatched every substep, a thread an element over the whole fleet: nodes;
centred damping's beams and nodes; beams; then self-collision as three dispatches (a
single-workgroup *list* kernel that tracks node travel and rebuilds the kept pair list
through a grid sort when needed, a *walk*, and *pairs* with a thread a pair); then a
single-workgroup *hits* kernel that applies the contact rule. Raycast wheels and the
coupling run in a workgroup a vehicle of their own beside the list kernel. A frame is 243
dispatches (§8.31). One car at rest, each pass timed apart: hits 1.22, list 1.51, nodes
0.85, beams 0.54, walk and pairs 0.51, centred damping 0.50 (§8.33). The whole frame by
the matrix: 3.22.

### B.4 Vehicle contact

A plan, made once for at least 34 substeps, takes each vehicle's box padded by its travel
and forms **contact groups** of vehicles whose boxes overlap. Two paths then exist.

- *Group path* (engine default): each group runs in one workgroup for the whole frame, its
  members' steps and the contact pass together.
- *Split path* (lab default): once any group exists, every vehicle is stepped one dispatch
  a substep. Each substep is a member step (a workgroup a vehicle), about eight fleet-wide
  dispatches that find candidates and hits (a thread a triangle, a part of a triangle, a
  pair), and the group's workgroup applying the owners' levels in a loop with barriers.
  Nothing stays in workgroup memory between dispatches. Candidates are found every
  substep; a kept list exists but is off because the page in Chrome ended in another state
  with it.

Measured by the matrix: two cars head-on 15.41, side impact 9.30, eight cars piling up
14.43, thirty-two 56.39.

### B.5 The coupling and surface contact

GPU passes in the World gather each vehicle's proxies once a World step by a swept test
(a collider's bounding sphere swept over the window against the vehicle's swept box).
Inside the frame each substep integrates the proxy bodies from the last substep's sums,
counts nodes against shapes for the mass split, and applies contact in the node phase.
Each node keeps a list of up to 16 nearby shapes, rebuilt when it leaves its skin.
Reactions are fixed-point sums a shape. Proxies are boxes, spheres, capsules and hulls.

Surface contact is opt-in, off in the lab and the matrix. Each element scans the shapes'
bounds and lists pairs; one lane a pair runs the whole closest-feature query; hits are
applied. When the vehicle is also coupled to the World the pass runs as nine separate
pipelines between segments of the frame kernel, dispatched every substep.

### B.6 Variants and switches

Modules are compiled per fleet, not per model: 256 or 512 lanes; with or without aero and
commands; solo, small-group and large-group contact; with or without surface contact;
with or without the coupling. About two dozen lossless speed-ups are bits of one mask
(`perf_flags_default`); those on by default were each proved bit for bit on saved states
first. Vehicle sleep is on in the World and off in the lab; a sleeping vehicle is not
dispatched.

## Appendix C. Where each statement comes from, and how sure it is

| statement | source | confidence |
|---|---|---|
| the experiment's figures (§1, A.7, A.8) | measured in this work, whole frames from fixed states, under the GPU lock; logs in `~/Development/rubble-artifacts/gpu-floor/` | high for sizes; differences under 0.04 ms are noise |
| the experiment is bit for bit | every state word compared every frame, emulator and GPU | high |
| production's matrix figures (§1, B.3, B.4) | quoted from `scenario-matrix.md`, measured at main 366bc068, not re-run | as good as that doc; main has moved since |
| production's per-pass figures (§3.2, B.3) | quoted from §8.31 and §8.33, each pass timed in a pass of its own | each is high by about 0.1 ms; they are for ranking |
| the wheels' cost, 0.85 ms | quoted from §8.31, before the wheels moved to their own core | stale; it is the weak point of "near 2" |
| the planner's schedules, the wide plan, the lab's defaults, the perf mask | read by me: `explicit_plan.rs:440-472`, `:3236-3372`; `vehicle_lab/mod.rs:172-196` | high |
| what a beam and a node load every substep; the per-node contact words; `hits` always true on the wide path | read by me: `explicit_beams.slang:1081-1092`, `:1183-1215`, `:1438-1458`, `:872-890`, `:926-948`, `:2955-2985`, `:7630-7650` | high |
| production's workgroup arrays; the 1,024-lane note; the pair list's margin and reuse rate | read by me: `explicit_beams.slang:391-397`, `:3898-3900`, `:4115-4117`; `explicit_plan.rs:494-507`, `:3675-3700` | high |
| the proxies' integration loop and its range; the per-node shape list | read by me: `explicit_beams_rigid.slang:1-30`, `:612-634`, `:664-678`, `:773`, `:948-954` | high |
| the surface pass: pairs appended inside the loop, a lane a pair, nine pipelines a substep | read by me: `explicit_beams_surface.slang:555-585`, `:996-1018`, `:1096-1114`; `explicit_plan.rs:3276-3285` | high |
| the split path steps every vehicle a dispatch a substep | read by me: `explicit_beams_gpu.rs:3236-3252`, `:3495-3506` | high |
| "about ten dispatches a substep" on the split path | counted from the list of operations by a second reader (`explicit_beams_gpu.rs:4606-4689`), not by me | medium |
| vehicle contact's caps, the levels' loop, sleep, how kernels are compiled and cached, shape kinds and casts in the surface pass | a second reader's report with lines; I checked a sample of the lines, not all | medium |
| fast math is on in production's Metal builds; strict by XOR helpers | read by me: `engine-metal/src/explicit.rs:6-14`, `:100-106`; `strict.slang:1-40` | high |
| expected gains in §6 | the experiment's own kernel, unless marked | they bound nothing in production; each item names a first measurement |
| WebGPU's default of 256 threads a workgroup; that no API promises concurrent workgroups | general knowledge of the specifications, not tested here | check before relying on it |

"A second reader" is a read-only search agent given a narrow question and asked for file
and line on every answer.

## Appendix D. How to re-check

From `engine/`. GPU runs go through the machine's lock and are bounded; never start two.

```sh
# The experiment: build, then the checks that need no GPU.
cargo build --profile fast -p gpu-floor
./target/fast/floor --v2 --emu --frames 20 --wall 3 --speed 16    # one body, a wall crash
./target/fast/pair --scene headon --emu --frames 12 --reverse     # two bodies, lanes reversed
./target/fast/rigid --scene bricks --emu --frames 12              # rigid proxies
experiments/gpu-floor/surf-emu.sh                                 # surface contact
experiments/gpu-floor/reach-emu.sh                                # the kernel choice, 24 runs

# The experiment on the GPU: strict is the default; add --fast --native for the fast build.
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/floor --v2 --frames 60 --phases
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/pair --scene headon --gpu --frames 12
python3 scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 ./target/fast/rigid --scene post --surface --gpu --reach --list 7

# Production's figures.
tools/scenario-matrix/run.sh --make        # once: save the states
tools/scenario-matrix/run.sh --baseline    # each row's ms a frame
PROFILE_SPLIT=1 tools/scenario-matrix/run.sh perf=0 drive   # one row by dispatch kind
```

Each driver prints "bit for bit" or the first differing word, and the frame times. Flags
and the comparison switches are in `engine/experiments/gpu-floor/README.md`.

Where to look in the experiment's code: `kernels/floor2.metal` (the kernel; its header is
the layout), `kernels/pair.metal` (two bodies), `src/floor.rs` (the layout's tables, and
`reach_plan`), `src/gpu.rs` (the host, 115 lines).
