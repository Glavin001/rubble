# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`rubble_sedan_ror.truck`: 367 nodes, 2170 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.237 | 0.259 | 1.128 | 4.579 | 1.151 | 2.400 | 0.21× | inconclusive | 1127.62 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.276 | 0.337 | 0.0 % | 5.796 | 6.946 | 0.0 % | 8.976 | 9.539 | 0.05× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.22 | 0.34 | 0 / 0 | 0 / 0 | 0.000 / -0.000 | -0.95 / -0.94 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 0.18 mm, max 0.29 mm; broken 0 / 0, bent 0 / 0.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.231 | 0.247 | 1.123 | 4.487 | 1.148 | 2.330 | 0.21× | inconclusive | 1123.15 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.257 | 0.262 | 0.0 % | 5.678 | 6.005 | 0.0 % | 8.340 | 9.092 | 0.04× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.04 | 0.08 | 0 / 0 | 10 / 10 | 47.600 / 47.600 | -1.08 / -1.08 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 0.25 mm, max 0.32 mm; broken 0 / 0, bent 10 / 10.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.241 | 0.293 | 1.140 | 3.666 | 1.170 | 2.399 | 0.21× | inconclusive | 1139.53 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.369 | 0.395 | 0.0 % | 5.606 | 6.003 | 0.0 % | 7.454 | 9.641 | 0.07× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 302.67 | 1936.74 | 344 / 344 | 542 / 540 | 15.348 / 15.363 | 85.99 / 34.67 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 318.11 mm, max 1956.31 mm; broken 344 / 344, bent 540 / 538.

## Acceptance

Command: `./target/release/explicit_beams_scaling docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 max_load=1000 max_gpu=100000 out=docs/explicit-beams-acceptance/accuracy/sedan`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100000 %): 0 did not |
| PASS | idle N=1 vehicle 0: node RMS 0.22 mm <= 1.00 mm (1.5x RoR's own 0.18 mm, at least 1 mm) |
| PASS | idle N=1 vehicle 0: broken 0 within RoR's 0–0 widened 10% (0–0) |
| PASS | idle N=1 vehicle 0: bent 0 within RoR's 0–0 widened 10% (0–0) |
| FAIL | idle N=1: speed-up 0.21x, 95% CI [NaN, NaN] excludes 1 (RoR 0.237 ms, Rubble GPU 1.128 ms) |
| FAIL | idle N=1: Rubble pipelined wall 1.151 ms < RoR mean 0.237 ms |
| FAIL | idle N=1: Rubble p99 5.796 < RoR 0.276 ms, max 6.946 < 0.337 ms |
| PASS | drive N=1 vehicle 0: node RMS 0.04 mm <= 1.00 mm (1.5x RoR's own 0.25 mm, at least 1 mm) |
| PASS | drive N=1 vehicle 0: broken 0 within RoR's 0–0 widened 10% (0–0) |
| PASS | drive N=1 vehicle 0: bent 10 within RoR's 10–10 widened 10% (9–11) |
| FAIL | drive N=1: speed-up 0.21x, 95% CI [NaN, NaN] excludes 1 (RoR 0.231 ms, Rubble GPU 1.123 ms) |
| FAIL | drive N=1: Rubble pipelined wall 1.148 ms < RoR mean 0.231 ms |
| FAIL | drive N=1: Rubble p99 5.678 < RoR 0.257 ms, max 6.005 < 0.262 ms |
| PASS | impact N=1 vehicle 0: node RMS 302.67 mm <= 477.17 mm (1.5x RoR's own 318.11 mm, at least 1 mm) |
| PASS | impact N=1 vehicle 0: broken 344 within RoR's 344–344 widened 10% (309–379) |
| PASS | impact N=1 vehicle 0: bent 542 within RoR's 540–538 widened 10% (484–594) |
| FAIL | impact N=1: speed-up 0.21x, 95% CI [NaN, NaN] excludes 1 (RoR 0.241 ms, Rubble GPU 1.140 ms) |
| FAIL | impact N=1: Rubble pipelined wall 1.170 ms < RoR mean 0.241 ms |
| FAIL | impact N=1: Rubble p99 5.606 < RoR 0.369 ms, max 6.003 < 0.395 ms |

**NOT MET** (9 failing)
