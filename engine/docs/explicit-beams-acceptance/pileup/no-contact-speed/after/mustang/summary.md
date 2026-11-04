# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`1982-Ford-Mustang-GT-351W.truck`: 527 nodes, 2664 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## mixed

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | 0.725 [0.718, 0.733] | 0.735 | 0.735 [0.726, 0.745] | 0.986 [0.984, 0.988] | — | — | 725.25 |
| 64 | — | — | 1.638 [1.632, 1.643] | 1.676 | 1.649 [1.644, 1.653] | 1.930 [1.870, 1.991] | — | — | 25.59 |
| 1024 | — | — | 23.260 [22.789, 23.732] | 23.722 | 23.272 [22.804, 23.741] | 23.534 [23.491, 23.577] | — | — | 22.72 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.740 | 0.749 | 0.0 % | 1.130 | 2.297 | — |
| 64 | — | — | — | 1.686 | 1.714 | 0.0 % | 2.108 | 10.451 | — |
| 1024 | — | — | — | 24.000 | 24.227 | 100.0 % | 24.158 | 32.382 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 64, Rubble blocking wall 64.


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 0.466 ms + 22.24 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 68.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 158.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 353.  

