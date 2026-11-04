# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`E36Sedan.truck`: 487 nodes, 2898 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## mixed

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | 0.700 [0.677, 0.724] | 0.726 | 0.711 [0.689, 0.732] | 0.967 [0.956, 0.978] | — | — | 700.46 |
| 64 | — | — | 1.598 [1.588, 1.607] | 1.692 | 1.609 [1.599, 1.618] | 1.880 [1.857, 1.902] | — | — | 24.96 |
| 1024 | — | — | 20.515 [20.494, 20.536] | 21.498 | 20.527 [20.505, 20.548] | 20.836 [20.811, 20.861] | — | — | 20.03 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.736 | 0.745 | 0.0 % | 1.104 | 1.180 | — |
| 64 | — | — | — | 1.734 | 1.841 | 0.0 % | 2.124 | 2.850 | — |
| 1024 | — | — | — | 21.732 | 21.926 | 100.0 % | 22.108 | 23.258 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 64, Rubble blocking wall 64.


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 0.520 ms + 19.52 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 75.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 178.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 400.  

