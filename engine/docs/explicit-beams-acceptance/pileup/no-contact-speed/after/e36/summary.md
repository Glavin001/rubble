# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`E36Sedan.truck`: 487 nodes, 2898 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## mixed

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | 0.706 [0.702, 0.710] | 0.730 | 0.716 [0.714, 0.718] | 0.968 [0.953, 0.984] | — | — | 706.18 |
| 64 | — | — | 1.631 [1.573, 1.689] | 1.728 | 1.647 [1.578, 1.716] | 1.913 [1.849, 1.977] | — | — | 25.49 |
| 1024 | — | — | 20.624 [20.461, 20.788] | 22.044 | 20.638 [20.477, 20.799] | 21.026 [20.892, 21.160] | — | — | 20.14 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | — | — | — | 0.737 | 0.747 | 0.0 % | 1.144 | 1.277 | — |
| 64 | — | — | — | 2.715 | 2.814 | 0.0 % | 2.978 | 3.200 | — |
| 1024 | — | — | — | 22.894 | 23.251 | 100.0 % | 23.300 | 27.516 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 64, Rubble blocking wall 64.


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 0.536 ms + 19.61 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 74.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 176.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 397.  

