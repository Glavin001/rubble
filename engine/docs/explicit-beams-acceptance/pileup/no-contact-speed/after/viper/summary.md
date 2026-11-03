# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`ViperGTS.car`: 276 nodes, 1469 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## mixed

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | 0.544 [0.540, 0.549] | 0.560 | 0.554 [0.547, 0.561] | 0.807 [0.800, 0.815] | — | — | 544.16 |
| 64 | — | — | 0.763 [0.740, 0.786] | 0.814 | 0.774 [0.751, 0.797] | 1.030 [0.994, 1.066] | — | — | 11.92 |
| 1024 | — | — | 8.351 [8.323, 8.379] | 8.712 | 8.362 [8.334, 8.391] | 8.661 [8.610, 8.712] | — | — | 8.16 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.617 | 0.644 | 0.0 % | 0.966 | 1.061 | — |
| 64 | — | — | — | 0.828 | 0.845 | 0.0 % | 1.182 | 1.300 | — |
| 1024 | — | — | — | 8.836 | 8.892 | 54.0 % | 9.191 | 9.582 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 64, Rubble blocking wall 64.


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 0.406 ms + 7.75 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 205.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 463.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 1022.  

