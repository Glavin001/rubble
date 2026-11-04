# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`ViperGTS.car`: 276 nodes, 1469 beams. Apple M3 Max (Metal). 1 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## tbone

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.929 | 1.103 | 0.920 | 1.794 | 0.930 | 1.197 | 1.01× | inconclusive | 460.19 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.337 | 1.475 | 0.0 % | 2.140 | 2.203 | 0.0 % | 2.407 | 2.456 | 0.67× |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 2, Rubble blocking wall 2, Rigs of Rods 2 (of counts up to 2).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 574.84 | 880.43 | 0 / 0 | 33 / 25 | 63.586 / 63.485 | 1.52 / -3.94 |
| 2 | 1 | 768.22 | 1323.70 | 1 / 1 | 14 / 13 | 6.427 / 6.486 | -13.54 / -133.08 |
| 2 | 0 | 574.84 | 880.43 | 0 / 0 | 33 / 25 | 63.586 / 63.485 | 1.52 / -3.94 |
| 2 | 1 | 768.22 | 1323.70 | 1 / 1 | 14 / 13 | 6.427 / 6.486 | -13.54 / -133.08 |

Accuracy ensembles (24 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 679.80 mm, RoR–RoR 702.60 mm, ratio 0.97 [95 % bootstrap 0.83, 1.11]; broken: Rubble median 2.0, RoR 0.0–2.0; bent: Rubble median 83.0, RoR 62.0–104.0.

## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/viper/ViperGTS.car ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=tbone counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/viper-tbone`

| result | criterion |
|---|---|
| PASS | 2 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| PASS | tbone N=2 (24 runs per engine): median node RMS Rubble–RoR 679.80 mm <= 1053.91 mm (1.5x RoR–RoR median 702.60 mm, at least 1 mm); ratio 0.97 [95 % 0.83, 1.11] |
| PASS | tbone N=2: Rubble median broken 2 within RoR's 0.0–2.0 widened 10% (0–3) |
| PASS | tbone N=2: Rubble median bent 83 within RoR's 62.0–104.0 widened 10% (55–115) |
| FAIL | tbone N=2 vehicle 0: median node RMS to the RoR runs 1151.64 mm <= 1053.91 mm |
| FAIL | tbone N=2: speed-up 1.01x, 95% CI [NaN, NaN] excludes 1 (RoR 0.929 ms, Rubble GPU 0.920 ms) |
| FAIL | tbone N=2: Rubble pipelined wall 0.930 ms < RoR mean 0.929 ms |
| FAIL | tbone N=2: Rubble p99 2.140 < RoR 1.337 ms, max 2.203 < 1.475 ms |

**NOT MET** (4 failing)
