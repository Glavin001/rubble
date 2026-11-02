# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`95bbUID-agoral.truck`: 297 nodes, 1328 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.183 | 0.197 | 1.292 | 3.248 | 1.468 | 2.255 | 0.14× | inconclusive | 1292.21 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.228 | 0.247 | 0.0 % | 4.061 | 4.389 | 0.0 % | 6.130 | 6.784 | 0.06× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 11.05 | 18.73 | 0 / 0 | 0 / 0 | 0.026 / 0.030 | -3.56 / -3.54 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 10.57 mm, max 19.90 mm; broken 0 / 0, bent 0 / 0.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.189 | 0.212 | 1.057 | 3.368 | 1.101 | 1.839 | 0.18× | inconclusive | 1056.83 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.253 | 0.284 | 0.0 % | 4.504 | 5.511 | 0.0 % | 5.737 | 6.076 | 0.05× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 7.38 | 16.80 | 0 / 0 | 0 / 0 | 63.377 / 63.380 | -3.62 / -3.75 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 8.94 mm, max 23.94 mm; broken 0 / 0, bent 0 / 0.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.217 | 0.240 | 1.118 | 4.492 | 1.151 | 1.975 | 0.19× | inconclusive | 1117.82 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.275 | 0.303 | 0.0 % | 5.396 | 5.527 | 0.0 % | 5.821 | 6.121 | 0.05× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 194.76 | 521.13 | 30 / 30 | 66 / 65 | 10.824 / 10.816 | 170.81 / 167.87 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 91.87 mm, max 209.56 mm; broken 30 / 30, bent 65 / 67.

## Acceptance

Command: `./target/release/explicit_beams_scaling /Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-content/agora/95bbUID-agoral.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 max_load=1000 max_gpu=100000 out=docs/explicit-beams-acceptance/accuracy/agora_l`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100000 %): 0 did not |
| PASS | idle N=1 vehicle 0: node RMS 11.05 mm <= 15.86 mm (1.5x RoR's own 10.57 mm, at least 1 mm) |
| PASS | idle N=1 vehicle 0: broken 0 within RoR's 0–0 widened 10% (0–0) |
| PASS | idle N=1 vehicle 0: bent 0 within RoR's 0–0 widened 10% (0–0) |
| FAIL | idle N=1: speed-up 0.14x, 95% CI [NaN, NaN] excludes 1 (RoR 0.183 ms, Rubble GPU 1.292 ms) |
| FAIL | idle N=1: Rubble pipelined wall 1.468 ms < RoR mean 0.183 ms |
| FAIL | idle N=1: Rubble p99 4.061 < RoR 0.228 ms, max 4.389 < 0.247 ms |
| PASS | drive N=1 vehicle 0: node RMS 7.38 mm <= 13.41 mm (1.5x RoR's own 8.94 mm, at least 1 mm) |
| PASS | drive N=1 vehicle 0: broken 0 within RoR's 0–0 widened 10% (0–0) |
| PASS | drive N=1 vehicle 0: bent 0 within RoR's 0–0 widened 10% (0–0) |
| FAIL | drive N=1: speed-up 0.18x, 95% CI [NaN, NaN] excludes 1 (RoR 0.189 ms, Rubble GPU 1.057 ms) |
| FAIL | drive N=1: Rubble pipelined wall 1.101 ms < RoR mean 0.189 ms |
| FAIL | drive N=1: Rubble p99 4.504 < RoR 0.253 ms, max 5.511 < 0.284 ms |
| FAIL | impact N=1 vehicle 0: node RMS 194.76 mm <= 137.81 mm (1.5x RoR's own 91.87 mm, at least 1 mm) |
| PASS | impact N=1 vehicle 0: broken 30 within RoR's 30–30 widened 10% (27–33) |
| PASS | impact N=1 vehicle 0: bent 66 within RoR's 65–67 widened 10% (58–74) |
| FAIL | impact N=1: speed-up 0.19x, 95% CI [NaN, NaN] excludes 1 (RoR 0.217 ms, Rubble GPU 1.118 ms) |
| FAIL | impact N=1: Rubble pipelined wall 1.151 ms < RoR mean 0.217 ms |
| FAIL | impact N=1: Rubble p99 5.396 < RoR 0.275 ms, max 5.527 < 0.303 ms |

**NOT MET** (10 failing)
