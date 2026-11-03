# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`1982-Ford-Mustang-GT-351W.truck`: 527 nodes, 2664 beams. Apple M3 Max (Metal). 1 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## headon

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.079 | 1.233 | 1.312 | 2.596 | 1.321 | 1.603 | 0.82× | inconclusive | 655.85 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.815 | 2.011 | 0.0 % | 2.745 | 2.889 | 0.0 % | 3.082 | 3.128 | 0.70× |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 2, Rubble blocking wall 2, Rigs of Rods 2 (of counts up to 2).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 169.55 | 569.91 | 0 / 0 | 611 / 575 | 57.737 / 57.726 | 25.72 / 26.04 |
| 2 | 1 | 173.18 | 782.57 | 0 / 0 | 609 / 581 | -57.760 / -57.791 | 26.21 / 27.46 |
| 2 | 0 | 169.55 | 569.91 | 0 / 0 | 611 / 575 | 57.737 / 57.726 | 25.72 / 26.04 |
| 2 | 1 | 173.18 | 782.57 | 0 / 0 | 609 / 581 | -57.760 / -57.791 | 26.21 / 27.46 |

Accuracy ensembles (24 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 213.97 mm, RoR–RoR 211.74 mm, ratio 1.01 [95 % bootstrap 0.95, 1.10]; broken: Rubble median 0.0, RoR 0.0–0.0; bent: Rubble median 2317.0, RoR 2198.0–2396.0.

## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=headon counts=2 rubble_counts= fps=120 frames=360 speed=27.78 drag=0 trials=1 nudges=24 quiet_wait=800 max_load=8 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/mustang-headon`

| result | criterion |
|---|---|
| PASS | 2 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| PASS | headon N=2 (24 runs per engine): median node RMS Rubble–RoR 213.97 mm <= 317.61 mm (1.5x RoR–RoR median 211.74 mm, at least 1 mm); ratio 1.01 [95 % 0.95, 1.10] |
| PASS | headon N=2: Rubble median broken 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| PASS | headon N=2: Rubble median bent 2317 within RoR's 2198.0–2396.0 widened 10% (1978–2636) |
| PASS | headon N=2 vehicle 0: median node RMS to the RoR runs 198.84 mm <= 317.61 mm |
| FAIL | headon N=2: speed-up 0.82x, 95% CI [NaN, NaN] excludes 1 (RoR 1.079 ms, Rubble GPU 1.312 ms) |
| FAIL | headon N=2: Rubble pipelined wall 1.321 ms < RoR mean 1.079 ms |
| FAIL | headon N=2: Rubble p99 2.745 < RoR 1.815 ms, max 2.889 < 2.011 ms |

**NOT MET** (3 failing)
