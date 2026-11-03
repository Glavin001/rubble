# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`1982-Ford-Mustang-GT-351W.truck`: 527 nodes, 2664 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## mixed

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | 0.726 [0.725, 0.727] | 0.734 | 0.736 [0.734, 0.738] | 0.976 [0.932, 1.021] | — | — | 725.67 |
| 64 | — | — | 1.656 [1.642, 1.670] | 1.688 | 1.667 [1.653, 1.681] | 1.931 [1.924, 1.939] | — | — | 25.87 |
| 1024 | — | — | 24.775 [23.087, 26.462] | 25.671 | 24.789 [23.097, 26.480] | 25.095 [24.168, 26.021] | — | — | 24.19 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.738 | 0.747 | 0.0 % | 1.161 | 1.403 | — |
| 64 | — | — | — | 1.704 | 1.711 | 0.0 % | 2.077 | 2.195 | — |
| 1024 | — | — | — | 26.246 | 27.144 | 100.0 % | 26.415 | 32.400 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 64, Rubble blocking wall 64.


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 0.428 ms + 23.76 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 66.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 150.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 332.  

