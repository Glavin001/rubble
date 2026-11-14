# Fleet benchmark: many generated sedans

The gate for the vehicle frame's performance work (2026-11-14). `engine/tools/explicit-2khz/fleet.sh`
runs `examples/explicit_vehicles_bench` with the generated sedan (`sedan_new`, JBeam mode
on its pressure wheels: 804 nodes, 6,429 beams, 528 collision triangles), native wgpu on
Metal (M3 Max), GPU time of each 1/60 s World step from submit to completion, 3 s.
The page's own numbers (headless Chrome, raycast wheels: 548 nodes) are in
`lab-mountain-perf.md`. Target: p99 under 16.7 ms.

A run that overlaps another GPU job reads far too high (32 cars apart: 132 ms once,
14 ms repeated on an idle GPU): check the GPU's utilisation first.

## Baseline (`fe3b9074`, before the pair list's reuse)

| Cars | Apart: median / p99 (ms) | Crash town: median / p99 (ms) |
|---|---|---|
| 1 | 11.7 / 12.9 | 14.9 / 15.6 |
| 2 | 11.9 / 13.1 | |
| 4 | 12.1 / 13.2 | 15.1 / 45.5 |
| 8 | 12.7 / 13.8 | 66.6 / 71.0 |
| 16 | 13.0 / 14.5 | 72.6 / 112.9 |
| 24 | 13.5 / 15.0 | |
| 32 | 13.9–14.2 / 15.4–16.3 | fails: more colliders than the coupling's 1,024 shapes |

"Apart": the cars couple with nothing and do not meet (`EV_NO_CONTACT=1 EV_NO_VV=1`).
The World without vehicles is 2.0–2.4 ms of each figure.

What it shows:
- **Separate cars cost one car's latency.** 1 to 32 cars: 11.7 to 14 ms. Each car is its
  own workgroup; the GPU is far from full.
- **Cars in contact are the wall.** In the crash town the cars drive into houses and each
  other: 8 cars take 67 ms, 16 take 73 ms (p99 113). Contact groups step several cars in
  one workgroup.
- 32 cars in the town run out of coupling shapes at step 70 (a capacity error, as designed).

## After the reused pair list and own-travel boxes (`e356f58a`, 2026-11-14)

Both are in `lab-mountain-perf.md` §5. This bench's cars run the reuse (on by default,
lossless); own-travel boxes are the lab's setting and are off here.

| Cars | Apart: median / p99 (ms), before → after | Crash town: median / p99 (ms), before → after |
|---|---|---|
| 1 | 11.7 / 12.9 → 8.9 / 10.0 | 14.9 / 15.6 → 12.1 / 13.7 |
| 4 | 12.1 / 13.2 → not run | 15.1 / 45.5 → 12.3 / 47.1 |
| 8 | 12.7 / 13.8 → 9.6 / 10.8 | 66.6 / 71.0 → 67.6 / 71.4 |
| 32 | 13.9–14.2 / 15.4–16.3 → 11.4 / 12.5 | fails (coupling shapes), as before |

(The GPU showed 92 % utilisation just before the "after" runs; they are no lower than a
quiet run would be.) Separate cars gained 2.6–2.9 ms each. Cars in contact did not:
contact groups keep the walk every step, and are the next work.

Headless Chrome, one car in the lab (raycast wheels), GPU ms per World step, before → after:
parked 6.9 → 4.5; mountain free fall 7.8 → 5.0, 8–16 s 11.5–12.6 → 9.6–10.1, 20 s
13.8 → 11.1 (60 steps/s until 22 s, was 19 s), 26 s on 17–20 → 18–22 (40 steps/s, unchanged:
the detached nodes in free fall invalidate the list most steps).

## Cars in contact: where a contact group's frame goes, and three lossless fixes (2026-11-14)

**Why 8 cars cost 5× 4 cars.** At 4 cars the crash town forms no contact group (the cars
strike houses); from 8 the cars meeting head on form groups of 2 (`EV_GROUPS=1`: 2 groups
of 2 at 8 cars, 5 at 16, 10 at 32). A group steps its members in one workgroup, 128
lanes each, 34 steps a frame, and the frame waits for its slowest group. Same-state
ablations of the explicit frame (`explicit_vehicles_bench EV_AB_AT=150 EV_AB_MASKS=…`:
one snapshot at World step 150, 20 frames of 34 steps under each ablation mask, time
only), 8 cars, ms a frame:

| Ablated | ms a frame | Share |
|---|---|---|
| nothing (the ablation build) | 68.4 | |
| vehicle–vehicle contact pass (512) | 29.9 | 38.5 |
| its triangle tests (131072) | 30.0 | 38.4 |
| its candidates (262144) | 30.2 | |
| its hits' application, the levels (524288) | 55.1 | 13.3 |
| the members' self-collision tests (1048576, new) | 52.5 | 15.9 |
| the proxies' node contacts (65536), the coupling passes (57344) | 68.3, 68.1 | ~0 |
| contact pass and members' self-collision | 14.0 | |

The contact pass's (T) phase tested every listed triangle of an owner against every
candidate (the partner's contact points in its box), each triangle on one lane; its levels
found each hit's chain by scanning every hit twice per hit.

**Fixes** (each behind `set_vv_grid`, on by default; bit for bit on against off in
`vv_grid_gpu`, and against the frozen reference in `explicit_beams_bitwise`):
1. The candidates in a grid (`5240985e`, failing test `f906a3a2`): each owner's
   candidates counted into its own buckets of cells (workgroup memory free in a group's
   frame), a triangle walks the cells of its box padded by the most any candidate's box
   test pads it. `vv_grid_gpu`: candidates visited 11,141,250 → 1,509,827.
2. The tests as a pair list (`00105901`): the walk lists the pairs its box test passes into
   the members' own pair regions, the lanes test them a pair each.
3. The levels' hits bucketed by contact point (`00105901`): each hit's chain from its
   point's bucket, not a scan of every hit.

Same state, 8 cars: 68.4 → 65.6 (1) → 54.9 (1 + 2) → 52.0 (1 + 2 + 3) ms a frame. The
remainder: the members' self-collision (the walk with its tests inline every step: no
reused pair list in a group) about 16 ms, the triangle tests about 10 ms, the levels'
application itself about 12 ms (the chains run serially per contact point), the rest 14 ms.

`fleet.sh town`, GPU ms median (180 steps; the lab's `sedan_new` as re-exported at 11:58,
so not comparable with the tables above; the grid off is the old path):

| Cars | grid off | grid on (default) | bonds broken |
|---|---|---|---|
| 1 | 11.9 | 12.6 | 9 |
| 4 | 30.0 | 30.7 | 68 |
| 8 | 71.4 | 53.5 | 169 |
| 16 | 75.3 | 58.9 | 224 |
| 32 | 138.9 | 103.6 | 293 |

The same bonds broke each way. **Not at the target** (8 and 16 cars within about 2× one
car): a group of two runs on one workgroup, so it can be no faster than its members' own
work at half the lanes each; reaching 2× needs the group split across workgroups (C2 in
known-bugs), not done.

**32 cars.** One run failed at World step 70 with "more colliders than the shapes room";
five later runs (with the grid on and off) ran 180 steps. The proxy shapes sampled every
step never passed 285 a car (room 1,024), and the vehicles' node boxes stayed under 9 m.
The status bit was shared by two capacities: the window's shapes and cpl_query's
candidate list (duplicates included, three times the shapes room). The latter now reports
its own message (`3f551fb5`), so the next failure will say which. Not reproduced since,
so its cause and the real need are not established.

### A contact group across workgroups (C2): what each car needs from the others

What one 2 kHz substep of a small group does, in order, inside its one workgroup
(`contactGroup` → `contactMembersStep`, `vvPassSmall`):

1. Per member: the node phase (reads its beam slots of the last beam phase and its
   contact accumulators, **including the vehicle-contact forces the previous substep's
   pass added**), its frame box keys (`vvBoxKey`, gAcc), the throttles, the beam phase,
   its self-collision (the walk, now a pair list) and `selfApply` (writes the
   accumulators).
2. Barrier (workgroup).
3. The vehicle-contact pass over the group: reads **every member's box keys of this
   substep**, every member's node positions and velocities **after this substep's node
   phase**, and each contact point's force **after this substep's beams and
   self-collision** (`vvPointForce`, `vvSnapshot`: beam sum plus accumulators); lists
   and tests; applies the hits level by level, each hit reading its contact point's force
   with the earlier hits added; adds to **both cars'** accumulators (the owner's
   triangle nodes and the partner's points), two-body counts and the residual phases.
4. Barrier; the next substep's node phase reads those accumulators.

So a member's substep s + 1 depends on every partner's substep s (twice: after the
member phase, and after the contact pass), and the contact pass depends on every
member's substep s complete. Two designs:

- **Same-substep (today's order), across workgroups:** each substep becomes two
  dispatches, the members' steps (a workgroup per car, 256 lanes) and the groups' contact
  passes (a workgroup per group), 68 dispatches a 34-step frame, every workgroup-memory
  state that lives across substeps (the pair list's reuse state, the coupling's sphere
  copies, the drive's and the throttle's per-frame words) moved to storage. The arithmetic
  and its order are unchanged (the member step does not depend on its lane count:
  per-element updates, fixed-point sums, hits applied in the specification's order), so it
  can be identical, but it is the whole group path rebuilt. Its cost, from the
  ablations: about 34 × (one car's substep, 0.27 ms, + the contact pass's share) plus
  68 dispatch boundaries; and the lone cars' frames (one dispatch, 9 ms) would run before
  or after it, not beside it (one queue). Estimated 25–35 ms for 8 cars: under today's
  47–53 ms but not within 2× one car.
- **Two-phase substep** (each car's contact detection on the last substep's partner
  state, so cars need no barrier inside a substep): changes results (the partner's
  positions, velocities and forces are a substep old). Not identical; not pursued.

None is cheap; the first is the only identical one and I have not built it.

**Members' self-collision as a pair list: tried, not identical, not shipped.** A grid
member's tested triangles' geometry once a step (triGeoFill), its walk listing the pairs
its box test passes (`triangleGridAtList`, a per-member counter) and its lanes testing
them a pair each (tryContact on the stored geometry), as the lone car's frame does. Two
generated sedans head-on, switch off against on from one state: frame 7 differed in 6
contact accumulator words by one fixed-point unit (`acc[777]` 0x6c58 against 0x6c59),
while the contact grid alone (the member lists disabled) was the same in all 40 frames.
The cause is not established. A likely one: a tryContact that reads a node's force while
other lanes add to it (the documented same-contacter race, resolved by `selfApply`'s
ordered re-evaluation in the lone frame) sees a different set of adds when the tests are
reordered across the member's lanes. Kept out; the kernel source is in the worktree's
gitignored `engine/out/late/explicit_beams.member.slang` (8 cars 47.1 ms a frame same
state, against about 53.5 without it).
