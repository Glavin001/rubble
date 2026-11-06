# Soft-body vehicles in the World broadphase: measurements (2026-11-06)

Branch `feat/softbody-world-broadphase`. M3 Max, GPU shared with other agents. Timings are
interleaved A/B in one process (broadphase and scan alternating, 8 rounds), medians.
Design: `explicit-vehicles-in-world.md` §11, `jbeam-semantics-spec.md` §38.3–38.4.

## 1. The coupling plan against map size

`engine-world/examples/vehicle_broadphase_bench.rs`: repo sedans (surface contact on) in
lanes 12 m apart on a 1 km × 1 km map; `n` rigid bodies, half static posts (a third of
them hulls), half 40 cm boxes resting on a ground slab (asleep after 240 settling steps),
spread uniformly, plus the same 40 posts and 20 boxes beside each car at every `n` (11
proxy shapes per car). No GPU fracture: the World keeps its resting table.

*Plan*: bounds, the query (or the scan), the plan and the share counts, 20 plans (5 at 256
cars) in one submission, wall ms per plan. *Step span*: a World step from submit to
completion with its wait (the explicit frame's 34 steps included).

| bodies | cars | plan, broadphase | plan, scan | step, broadphase | step, scan |
|---|---|---|---|---|---|
| 10,000 | 1 | 0.107 | 0.122 | 33.34 | 33.35 |
| 30,000 | 1 | 0.109 | 0.188 | 33.41 | 33.46 |
| 100,000 | 1 | 0.107 | 0.428 | 33.47 | 33.76 |
| 10,000 | 1 (surface off) | 0.097 | 0.110 | 2.36 | 2.38 |
| 100,000 | 1 (surface off) | 0.100 | 0.412 | 2.47 | 2.79 |
| 10,000 | 64 | 0.102 | 0.151 | 35.17 | 35.27 |
| 100,000 | 64 | 0.103 | 0.622 | 35.13 | 35.76 |
| 10,000 | 256 | 0.194 | 0.748 | 60.40 | 61.41 |
| 100,000 | 256 | 0.192 | 3.077 | 60.97 | 63.96 |

(1-car rows with `--no-sleep`: the car stays awake. With vehicle sleep the parked car
sleeps and the World skips the step: 0.00–0.01 ms, the plan 0.10 → 0.19 ms over 10k →
100k.)

* The plan from the broadphase is flat in the map's bodies (0.10–0.11 ms for one car or
  64, 0.19 ms for 256); the scan grows with them (0.12 → 0.43 ms for one car, 0.15 →
  0.62 for 64, 0.75 → 3.08 for 256).
* The step is flat in the map's bodies with the broadphase (33.3–33.5 ms for one car,
  60.4–61.0 for 256); the scan's share shows at 256 cars (+3 ms at 100k).
* What remains per window that follows the World's size: clearing the per-body share
  counts (4 bytes × body capacity, `enc.clear_buffer`), and the rigid step's own
  O(total) passes (`list_moving_colliders`, `pose_pack`: `large-maps.md` §1a).
* The surface pass dominates the step: 33 ms per World step (34 explicit steps) for one
  sedan among 11 proxy shapes, against 2.4 ms with it off. That is the pass's known cost
  (§38.1: a triangle's narrow phase on one lane, twice a step), not the broadphase's;
  it does not grow with the map.

**The town.** The crash town's houses are GPU-fractured: such a World takes the rigid
full pass and keeps no resting table (`threshold.rs`), so its coupling plan takes the
scan (`World::explicit_plan_used_broadphase` reports it). Its plan cost is unchanged by
this branch (0.6–0.9 ms at 64 cars, `explicit-vehicles-in-world.md` §8.2–8.4). Keeping the
resting table in a fracture world (the fracture re-parents colliders on the GPU) is
`large-maps.md` L1's.

## 2. Fast rigid bodies into a door (casts, §38.4)

`engine-world/examples/ccd_cannonball.rs`: a steel sphere (6 cm, 15 cm radius) or box
(20 cm) at 30–600 m/s horizontally into the side of a parked car at door height, surface
contact on; CPU reference, wgpu (the proxy written by the host each frame) and the World
(a dynamic body). *Given*: the momentum the car took, as a share of the projectile's
(over 100 %: it bounced back). *Tunnel*: it ended past the car's far side having given
less than 2 %. The ETK was not available here (`RUBBLE_ETK` unset): the repo sedan
(367 nodes) stands in for a coarse car beside sedan1 (1,026 nodes).

Before (`RUBBLE_SURFACE_CAST=0`) → after, CPU (GPU within 2 points of the CPU on every
row; deepest = the deepest triangle point inside the projectile, CPU, sampled every step):

| car | projectile | m/s | given, before | given, after | deepest mm, before → after |
|---|---|---|---|---|---|
| sedan1 | sphere 6 cm | 100 | 99.5 % | 102.5 % | 37.4 → 0.0 |
| sedan1 | sphere 6 cm | 300 | 99.9 % | 99.6 % | 12.3 → 59.1 |
| sedan1 | sphere 6 cm | 600 | 97.7 % | 99.5 % | 59.7 → 59.6 |
| sedan1 | sphere 15 cm | 300 | 75.8 % | 76.9 % | 147.7 → 148.8 |
| sedan1 | sphere 15 cm | 600 | 50.0 % | 60.5 % | 149.3 → 149.6 |
| sedan1 | box 20 cm | 600 | 90.9 % | 92.2 % | 98.5 → 98.1 |
| sedan | sphere 6 cm | 100 | 101.6 % | 75.9 % | 30.0 → 0.0 |
| sedan | sphere 6 cm | 300 | 99.7 % | 122.4 % (bounced) | 54.9 → 7.3 |
| sedan | sphere 6 cm | 600 | **46.6 % (through)** | 107.9 % (bounced) | 31.3 → 46.6 |
| sedan | sphere 15 cm | 600 | 35.4 % | 65.7 % | 139.6 → 148.7 |
| sedan | box 20 cm | 600 | 58.2 % | 82.7 % | 99.2 → 99.5 |

At 30 m/s every row is the same before and after (the casts never run). No row tunnels
after the change and none gains speed. The heavy projectiles (the 15 cm sphere, 111 kg; the box,
63 kg; the 6 cm sphere is 7 kg) still go deep at 300–600 m/s: they crush the door and leave through
the car with a third to a half of their momentum; that is penetration by a body far
heavier than the panel's nodes, not a missed contact. The 6 cm ball at 100 m/s into the
repo sedan now gives 76 % (it passes partly through after crushing the panel; before, it
bounced): the coarse panel's nodes take the cast contacts.

World runs (the shot started two World steps' travel farther out, so it meets the car in
the vehicles' windows; the ball integrates in the rigid step between windows) give the
same picture: no tunnel at any speed after the change (6 cm at 600 m/s: 103 % into the
repo sedan, 99.5 % into sedan1). The full tables, every mode, are the example's output.
