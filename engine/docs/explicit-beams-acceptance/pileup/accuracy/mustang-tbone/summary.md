# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`1982-Ford-Mustang-GT-351W.truck`: 527 nodes, 2664 beams. Apple M3 Max (Metal). 1 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## tbone

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.083 | 1.417 | 1.263 | 2.461 | 1.272 | 1.552 | 0.86× | inconclusive | 631.49 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.565 | 1.636 | 0.0 % | 2.604 | 2.651 | 0.0 % | 2.862 | 2.878 | 0.62× |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 2, Rubble blocking wall 2, Rigs of Rods 2 (of counts up to 2).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 234.50 | 618.30 | 0 / 0 | 344 / 346 | 63.921 / 64.092 | -1.84 / -1.39 |
| 2 | 1 | 615.02 | 938.89 | 0 / 0 | 239 / 245 | 7.428 / 6.836 | -267.75 / -259.56 |
| 2 | 0 | 234.50 | 618.30 | 0 / 0 | 344 / 346 | 63.921 / 64.092 | -1.84 / -1.39 |
| 2 | 1 | 615.02 | 938.89 | 0 / 0 | 239 / 245 | 7.428 / 6.836 | -267.75 / -259.56 |

Accuracy ensembles (24 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 246.28 mm, RoR–RoR 234.89 mm, ratio 1.05 [95 % bootstrap 0.85, 1.40]; broken: Rubble median 0.0, RoR 0.0–0.0; bent: Rubble median 1152.0, RoR 1100.0–1202.0.

## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=tbone counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/mustang-tbone`

| result | criterion |
|---|---|
| PASS | 2 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| PASS | tbone N=2 (24 runs per engine): median node RMS Rubble–RoR 246.28 mm <= 352.34 mm (1.5x RoR–RoR median 234.89 mm, at least 1 mm); ratio 1.05 [95 % 0.85, 1.40] |
| PASS | tbone N=2: Rubble median broken 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| PASS | tbone N=2: Rubble median bent 1152 within RoR's 1100.0–1202.0 widened 10% (990–1323) |
| PASS | tbone N=2 vehicle 0: median node RMS to the RoR runs 173.97 mm <= 352.34 mm |
| FAIL | tbone N=2: speed-up 0.86x, 95% CI [NaN, NaN] excludes 1 (RoR 1.083 ms, Rubble GPU 1.263 ms) |
| FAIL | tbone N=2: Rubble pipelined wall 1.272 ms < RoR mean 1.083 ms |
| FAIL | tbone N=2: Rubble p99 2.604 < RoR 1.565 ms, max 2.651 < 1.636 ms |

**NOT MET** (3 failing)
