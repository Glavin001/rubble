# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`rubble_hires_jbeam.truck`: 1110 nodes, 6158 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.282 | 2.658 | 0.466 | 0.483 | 0.476 | 0.781 | 4.89× | inconclusive | 466.47 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.891 | 2.936 | 0.0 % | 0.523 | 0.531 | 0.0 % | 1.110 | 1.307 | 5.53× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2.32 | 21.63 | 7 / 5 | 13 / 13 | -0.010 / -0.010 | -1.62 / -1.62 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 4.57 mm, RoR–RoR 3.94 mm, ratio 1.16 [95 % bootstrap 0.55, 1.40]; broken: Rubble median 7.0, RoR 3.0–12.0; bent: Rubble median 14.0, RoR 13.0–15.0.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.403 | 2.737 | 0.536 | 0.571 | 0.548 | 0.851 | 4.48× | inconclusive | 536.41 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.855 | 3.006 | 0.0 % | 0.579 | 0.614 | 0.0 % | 1.140 | 1.201 | 4.90× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 281.54 | 349.03 | 10 / 8 | 13 / 14 | 68.940 / 68.977 | -4.82 / -4.38 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 1224.18 mm, RoR–RoR 778.30 mm, ratio 1.57 [95 % bootstrap 0.73, 6.69]; broken: Rubble median 12.5, RoR 8.0–22.0; bent: Rubble median 14.5, RoR 9.0–20.0.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.126 | 5.886 | 0.694 | 0.747 | 0.706 | 1.005 | 7.38× | inconclusive | 694.18 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.155 | 6.871 | 0.0 % | 0.764 | 0.905 | 0.0 % | 1.375 | 1.462 | 7.59× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1339.05 | 2759.11 | 341 / 339 | 2104 / 2073 | 13.343 / 13.662 | -92.70 / 125.46 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 1305.60 mm, RoR–RoR 1958.02 mm, ratio 0.67 [95 % bootstrap 0.18, 2.00]; broken: Rubble median 339.5, RoR 319.0–506.0; bent: Rubble median 2111.5, RoR 2021.0–2182.0.

## Acceptance

Command: `./target/release/explicit_beams_scaling docs/explicit-beams-acceptance/vehicles/rubble_hires_jbeam.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 nudges=8 max_load=1000 max_gpu=100000 out=docs/explicit-beams-acceptance/accuracy/jbeam_car`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100000 %): 0 did not |
| PASS | idle N=1 (8 runs per engine): median node RMS Rubble–RoR 4.57 mm <= 5.91 mm (1.5x RoR–RoR median 3.94 mm, at least 1 mm); ratio 1.16 [95 % 0.55, 1.40] |
| PASS | idle N=1: Rubble median broken 7 within RoR's 3.0–12.0 widened 10% (2–14) |
| PASS | idle N=1: Rubble median bent 14 within RoR's 13.0–15.0 widened 10% (11–17) |
| FAIL | idle N=1: speed-up 4.89x, 95% CI [NaN, NaN] excludes 1 (RoR 2.282 ms, Rubble GPU 0.466 ms) |
| PASS | idle N=1: Rubble pipelined wall 0.476 ms < RoR mean 2.282 ms |
| PASS | idle N=1: Rubble p99 0.523 < RoR 2.891 ms, max 0.531 < 2.936 ms |
| FAIL | drive N=1 (8 runs per engine): median node RMS Rubble–RoR 1224.18 mm <= 1167.45 mm (1.5x RoR–RoR median 778.30 mm, at least 1 mm); ratio 1.57 [95 % 0.73, 6.69] |
| PASS | drive N=1: Rubble median broken 12.5 within RoR's 8.0–22.0 widened 10% (7–25) |
| PASS | drive N=1: Rubble median bent 14.5 within RoR's 9.0–20.0 widened 10% (8–22) |
| FAIL | drive N=1: speed-up 4.48x, 95% CI [NaN, NaN] excludes 1 (RoR 2.403 ms, Rubble GPU 0.536 ms) |
| PASS | drive N=1: Rubble pipelined wall 0.548 ms < RoR mean 2.403 ms |
| PASS | drive N=1: Rubble p99 0.579 < RoR 2.855 ms, max 0.614 < 3.006 ms |
| PASS | impact N=1 (8 runs per engine): median node RMS Rubble–RoR 1305.60 mm <= 2937.02 mm (1.5x RoR–RoR median 1958.02 mm, at least 1 mm); ratio 0.67 [95 % 0.18, 2.00] |
| PASS | impact N=1: Rubble median broken 339.5 within RoR's 319.0–506.0 widened 10% (287–557) |
| PASS | impact N=1: Rubble median bent 2111.5 within RoR's 2021.0–2182.0 widened 10% (1818–2401) |
| FAIL | impact N=1: speed-up 7.38x, 95% CI [NaN, NaN] excludes 1 (RoR 5.126 ms, Rubble GPU 0.694 ms) |
| PASS | impact N=1: Rubble pipelined wall 0.706 ms < RoR mean 5.126 ms |
| PASS | impact N=1: Rubble p99 0.764 < RoR 6.155 ms, max 0.905 < 6.871 ms |

**NOT MET** (4 failing)
