# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`ViperGTS.car`: 276 nodes, 1469 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## mixed

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | 0.546 [0.545, 0.548] | 0.563 | 0.557 [0.552, 0.561] | 0.809 [0.796, 0.823] | — | — | 546.46 |
| 64 | — | — | 0.735 [0.731, 0.739] | 0.776 | 0.746 [0.741, 0.750] | 1.016 [1.014, 1.017] | — | — | 11.48 |
| 1024 | — | — | 8.415 [8.405, 8.425] | 8.798 | 8.428 [8.421, 8.434] | 8.735 [8.724, 8.746] | — | — | 8.22 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.620 | 0.646 | 0.0 % | 0.956 | 1.057 | — |
| 64 | — | — | — | 0.790 | 0.806 | 0.0 % | 1.172 | 1.316 | — |
| 1024 | — | — | — | 8.923 | 9.141 | 67.6 % | 9.277 | 9.499 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 64, Rubble blocking wall 64.


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 0.391 ms + 7.83 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 205.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 461.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 1014.  

