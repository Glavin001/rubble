# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`E36Sedan.truck`: 487 nodes, 2898 beams. Apple M3 Max (Metal). 1 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## headon

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.296 | 1.423 | 1.310 | 2.471 | 1.319 | 1.476 | 0.99× | inconclusive | 654.99 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.639 | 2.805 | 0.0 % | 3.285 | 3.454 | 0.0 % | 3.440 | 3.733 | 0.81× |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 2, Rubble blocking wall 2, Rigs of Rods 2 (of counts up to 2).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 512.28 | 1099.32 | 67 / 64 | 443 / 458 | 57.325 / 56.875 | 6.81 / 14.12 |
| 2 | 1 | 583.52 | 1272.51 | 64 / 62 | 468 / 478 | -57.246 / -56.871 | 24.74 / 20.61 |
| 2 | 0 | 512.28 | 1099.32 | 67 / 64 | 443 / 458 | 57.325 / 56.875 | 6.81 / 14.12 |
| 2 | 1 | 583.52 | 1272.51 | 64 / 62 | 468 / 478 | -57.246 / -56.871 | 24.74 / 20.61 |

Accuracy ensembles (24 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 760.37 mm, RoR–RoR 745.29 mm, ratio 1.02 [95 % bootstrap 0.85, 1.16]; broken: Rubble median 281.0, RoR 232.0–356.0; bent: Rubble median 1849.0, RoR 1730.0–1942.0.

## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/e36/E36Sedan.truck ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=headon counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/e36-headon`

| result | criterion |
|---|---|
| PASS | 2 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| PASS | headon N=2 (24 runs per engine): median node RMS Rubble–RoR 760.37 mm <= 1117.94 mm (1.5x RoR–RoR median 745.29 mm, at least 1 mm); ratio 1.02 [95 % 0.85, 1.16] |
| PASS | headon N=2: Rubble median broken 281 within RoR's 232.0–356.0 widened 10% (208–392) |
| PASS | headon N=2: Rubble median bent 1849 within RoR's 1730.0–1942.0 widened 10% (1557–2137) |
| PASS | headon N=2 vehicle 0: median node RMS to the RoR runs 834.90 mm <= 1117.94 mm |
| FAIL | headon N=2: speed-up 0.99x, 95% CI [NaN, NaN] excludes 1 (RoR 1.296 ms, Rubble GPU 1.310 ms) |
| FAIL | headon N=2: Rubble pipelined wall 1.319 ms < RoR mean 1.296 ms |
| FAIL | headon N=2: Rubble p99 3.285 < RoR 2.639 ms, max 3.454 < 2.805 ms |

**NOT MET** (3 failing)
