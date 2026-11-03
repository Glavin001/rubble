# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`ViperGTS.car`: 276 nodes, 1469 beams. Apple M3 Max (Metal). 1 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## headon

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.893 | 1.033 | 0.820 | 1.931 | 0.830 | 1.087 | 1.09× | inconclusive | 410.18 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.239 | 1.360 | 0.0 % | 2.126 | 2.166 | 0.0 % | 2.411 | 2.470 | 0.63× |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 2, Rubble blocking wall 2, Rigs of Rods 2 (of counts up to 2).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 1758.17 | 2312.29 | 0 / 0 | 81 / 74 | 55.226 / 56.929 | 10.28 / 7.92 |
| 2 | 1 | 1852.70 | 2768.33 | 0 / 0 | 88 / 72 | -55.344 / -57.055 | 22.28 / 1.76 |
| 2 | 0 | 1758.17 | 2312.29 | 0 / 0 | 81 / 74 | 55.226 / 56.929 | 10.28 / 7.92 |
| 2 | 1 | 1852.70 | 2768.33 | 0 / 0 | 88 / 72 | -55.344 / -57.055 | 22.28 / 1.76 |

Accuracy ensembles (24 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 1030.88 mm, RoR–RoR 1236.76 mm, ratio 0.83 [95 % bootstrap 0.66, 1.32]; broken: Rubble median 0.0, RoR 0.0–0.0; bent: Rubble median 337.0, RoR 294.0–330.0.

## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/viper/ViperGTS.car ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=headon counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/viper-headon`

| result | criterion |
|---|---|
| PASS | 2 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| PASS | headon N=2 (24 runs per engine): median node RMS Rubble–RoR 1030.88 mm <= 1855.14 mm (1.5x RoR–RoR median 1236.76 mm, at least 1 mm); ratio 0.83 [95 % 0.66, 1.32] |
| PASS | headon N=2: Rubble median broken 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| PASS | headon N=2: Rubble median bent 337 within RoR's 294.0–330.0 widened 10% (264–364) |
| PASS | headon N=2 vehicle 0: median node RMS to the RoR runs 1310.19 mm <= 1855.14 mm |
| FAIL | headon N=2: speed-up 1.09x, 95% CI [NaN, NaN] excludes 1 (RoR 0.893 ms, Rubble GPU 0.820 ms) |
| PASS | headon N=2: Rubble pipelined wall 0.830 ms < RoR mean 0.893 ms |
| FAIL | headon N=2: Rubble p99 2.126 < RoR 1.239 ms, max 2.166 < 1.360 ms |

**NOT MET** (2 failing)
