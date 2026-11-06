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

## 3. Surface contact's cost against the proxies (branch `feat/surface-contact-perf`)

Design: `jbeam-semantics-spec.md` §38.5 (pair path, edge culls) and §38.6 (friction
against a turning body). M3 Max shared with up to four other agents (load 10–160): World
step spans swing by tens of ms between runs, so the comparisons that count are the
explicit frame's own GPU timestamps, every switch timed from one saved state.

### 3.1 Where the time went (before)

`vehicle_broadphase_bench --cars 1 --n 10000 --no-sleep --probe` with `EV_ABLATE` (bits
2^20–2^30 of `explicit_beams_surface.slang`, time only), one repo sedan among 11 proxy
shapes, a frame of 17 steps:

| Frame with | ms |
|---|---|
| as shipped | 16.73 |
| the surface pass skipped with its barriers | 0.81 |
| its work skipped, barriers kept | 0.93 |
| proxies skipped (static slots and snapshot only) | 1.26 |
| proxies: culled, nothing built | 3.03 |
| proxies: culled, poses and bounds built, no narrow phase | 3.80 |
| triangles / segments / points skipped | 5.72 / 13.62 / 15.80 |
| the apply skipped | 10.75 |

The narrow phase was 77 % (12.9 ms), the per-element cull 11 %, the poses and bounds per
element 5 %. 134 pairs met by box, 9 hit.

### 3.2 After: the lossless switches, same build, same state

Frame GPU ms, the switches timed in turn from one state (`set_surface_pairs`,
`set_surface_culls`):

| Case | both on | pairs off | culls off | both off (before) |
|---|---|---|---|---|
| one sedan, 11 proxies (`vehicle_broadphase_bench --probe`, median of 15) | 4.16 | 11.85 | 6.59 | 21.94 |
| crash town, 4 cars, 29–77 proxies each (`explicit_frame_probe`, EV_SURFACE=1, median of 9) | 17.08 | 55.12 | 18.25 | 55.67 |
| crash town, 16 cars (min of 9) | 30.5 | 23.9 | 18.0 | 57.5 |
| crash town, 32 cars (min of 9) | 37.2 | 50.1 | 42.2 | 212.6 |

The 16- and 32-car rows ran beside other agents' GPU jobs (medians spread 2–5×), so their
minima are given; the other rows were steady. The same 32-car town with surface contact
off: 16.7 ms (min; median 27.7); on with the casts off, 36.7 ms.

One sedan, World step span (submit to completion, `--no-sleep`, loaded GPU; on a quieter
GPU the base measured 2.4 ms off and 33 ms on, §1):

| bodies | surface off | on, before | on, after |
|---|---|---|---|
| 10,000 | 33.96 | 68.04 | 38.98 |
| 100,000 | 38.87 | 79.98 | 39.01 |

The pass's own cost for one car falls from 1.24 to 0.20 ms of GPU time per explicit step
(frame time above the 0.81 ms frame without it, over 17 steps), flat in the map's bodies.

**Not measured: 64 and 256 cars in the town.** A 64-car crash-town probe on the merged
branch (with `feat/vehicle-sleep`) did not finish its 150 settling steps: one GPU
submission ran for over an hour, with surface contact on and also with it off, so it is
not the surface pass (32 cars run in seconds). Open, for the vehicle-sleep side.

### 3.3 Cannonball after the friction fix

`ccd_cannonball --modes cpu` (CPU reference; the GPU agreed within 2 points before the
change and runs the same arithmetic). Columns: momentum given; kinetic energy after over
the projectile's at the start (the projectile's spin and the car included); the
projectile's spin at the end. Before = the branch with the casts (c79f533c), after = with
the friction cap, both before the `feat/vehicle-sleep` merge (which changed the CPU
reference elsewhere; merged numbers follow).

| car | projectile | m/s | given before → after | KE ratio before → after | spin rad/s before → after |
|---|---|---|---|---|---|
| sedan | sphere 6 cm | 100 | 75.9 → 103.0 % | 0.261 → 0.048 | 256 → 155 |
| sedan | sphere 6 cm | 300 | 122.4 → 101.5 % | 0.222 → 0.075 | 1,629 → 295 |
| sedan | sphere 6 cm | 600 | 107.9 → 105.0 % | 0.700 → 0.019 | 12,682 → 596 |
| sedan | sphere 15 cm | 600 | 65.7 → 76.2 % | 0.329 → 0.270 | 38 → 440 |
| sedan | box 20 cm | 300 | 97.8 → 89.6 % | 0.057 → 0.087 | 61 → 395 |
| sedan | box 20 cm | 600 | 82.7 → 64.4 % | 0.055 → 0.146 | 295 → 404 |
| sedan1 | sphere 15 cm | 600 | 60.5 → 47.0 % | 0.210 → 0.316 | 103 → 126 |
| sedan1 | box 20 cm | 600 | 92.2 → 83.6 % | 0.063 → 0.084 | 58 → 137 |
| ETK | sphere 6 cm | 600 | 92.4 → 99.4 % | 0.037 → 0.026 | 60 → 223 |
| ETK | box 20 cm | 600 | 43.5 → 53.4 % | 0.378 → 0.297 | 831 → 369 |

Every other row (30–600 m/s; 6 cm, 15 cm, box; three cars) moved by under 5 points. No
projectile tunnels, before or after. The 6 cm ball no longer leaves spinning faster than
it flew (12,682 rad/s × 6 cm = 761 m/s from a 600 m/s shot). The heavy projectiles at
300–600 m/s (the 15 cm sphere, 111 kg; the 20 cm box, 63 kg) punch through the door either
way (they end past the far side); with the friction no longer overshooting they give the
coarse repo sedan and sedan1 less of their momentum and the ETK more. The ETK
(`RUBBLE_ETK`, through its `.truck`) is new in the table.

After the merge (the merged CPU reference), the 6 cm ball at 30 / 100 / 300 / 600 m/s: the
repo sedan 97.2 / 103.6 / 105.4 / 99.0 %, sedan1 113.5 / 101.6 / 99.6 / 98.8 %, the ETK
107.9 / 100.2 / 97.3 / 94.2 %. No tunnel.
