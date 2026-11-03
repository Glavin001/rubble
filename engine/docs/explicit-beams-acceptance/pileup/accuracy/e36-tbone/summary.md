# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`E36Sedan.truck`: 487 nodes, 2898 beams. Apple M3 Max (Metal). 1 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## tbone

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.264 | 1.399 | 1.240 | 2.212 | 1.248 | 1.542 | 1.02× | inconclusive | 619.76 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.048 | 2.652 | 0.0 % | 2.976 | 3.292 | 0.0 % | 3.366 | 3.540 | 0.81× |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 2, Rubble blocking wall 2, Rigs of Rods 2 (of counts up to 2).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 493.92 | 730.64 | 11 / 11 | 369 / 368 | 65.524 / 65.635 | 12.97 / 15.05 |
| 2 | 1 | 384.07 | 845.81 | 1 / 3 | 364 / 377 | 9.032 / 8.953 | -204.83 / -236.81 |
| 2 | 0 | 493.92 | 730.64 | 11 / 11 | 369 / 368 | 65.524 / 65.635 | 12.97 / 15.05 |
| 2 | 1 | 384.07 | 845.81 | 1 / 3 | 364 / 377 | 9.032 / 8.953 | -204.83 / -236.81 |

Accuracy ensembles (24 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 465.89 mm, RoR–RoR 534.49 mm, ratio 0.87 [95 % bootstrap 0.74, 1.17]; broken: Rubble median 24.0, RoR 24.0–42.0; bent: Rubble median 1472.0, RoR 1422.0–1532.0.

## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/e36/E36Sedan.truck ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=tbone counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/e36-tbone`

| result | criterion |
|---|---|
| PASS | 2 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| PASS | tbone N=2 (24 runs per engine): median node RMS Rubble–RoR 465.89 mm <= 801.74 mm (1.5x RoR–RoR median 534.49 mm, at least 1 mm); ratio 0.87 [95 % 0.74, 1.17] |
| PASS | tbone N=2: Rubble median broken 24 within RoR's 24.0–42.0 widened 10% (21–47) |
| PASS | tbone N=2: Rubble median bent 1472 within RoR's 1422.0–1532.0 widened 10% (1279–1686) |
| PASS | tbone N=2 vehicle 0: median node RMS to the RoR runs 382.57 mm <= 801.74 mm |
| FAIL | tbone N=2: speed-up 1.02x, 95% CI [NaN, NaN] excludes 1 (RoR 1.264 ms, Rubble GPU 1.240 ms) |
| PASS | tbone N=2: Rubble pipelined wall 1.248 ms < RoR mean 1.264 ms |
| FAIL | tbone N=2: Rubble p99 2.976 < RoR 2.048 ms, max 3.292 < 2.652 ms |

**NOT MET** (2 failing)
