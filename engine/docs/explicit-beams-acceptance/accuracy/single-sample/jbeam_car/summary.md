# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`rubble_hires_jbeam.truck`: 1110 nodes, 6158 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.353 | 2.869 | 0.692 | 1.821 | 0.705 | 1.139 | 3.40× | inconclusive | 691.91 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.997 | 3.116 | 0.0 % | 2.433 | 2.523 | 0.0 % | 3.098 | 3.119 | 1.24× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 3.07 | 22.57 | 3 / 5 | 17 / 13 | -0.009 / -0.010 | -1.60 / -1.62 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 3.78 mm, max 53.98 mm; broken 5 / 5, bent 13 / 15.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.505 | 3.036 | 0.769 | 1.973 | 0.783 | 1.223 | 3.26× | inconclusive | 768.79 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.305 | 3.469 | 0.0 % | 2.383 | 2.630 | 0.0 % | 3.576 | 3.980 | 1.32× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 985.90 | 1042.63 | 13 / 8 | 15 / 14 | 69.121 / 68.977 | -4.21 / -4.38 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 1069.58 mm, max 1118.71 mm; broken 8 / 11, bent 14 / 20.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.454 | 6.360 | 1.097 | 2.738 | 1.113 | 1.660 | 4.97× | inconclusive | 1096.85 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.671 | 6.818 | 0.0 % | 3.429 | 3.936 | 0.0 % | 4.528 | 4.769 | 1.73× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1355.44 | 3600.91 | 303 / 339 | 2052 / 2073 | 13.725 / 13.662 | 169.15 / 125.46 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 796.78 mm, max 2442.07 mm; broken 339 / 346, bent 2073 / 2078.

## Acceptance

Command: `./target/release/explicit_beams_scaling docs/explicit-beams-acceptance/vehicles/rubble_hires_jbeam.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 max_load=1000 max_gpu=100 out=docs/explicit-beams-acceptance/accuracy/jbeam_car`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100 %): 0 did not |
| PASS | idle N=1 vehicle 0: node RMS 3.07 mm <= 5.67 mm (1.5x RoR's own 3.78 mm, at least 1 mm) |
| FAIL | idle N=1 vehicle 0: broken 3 within RoR's 5–5 widened 10% (4–6) |
| PASS | idle N=1 vehicle 0: bent 17 within RoR's 13–15 widened 10% (11–17) |
| FAIL | idle N=1: speed-up 3.40x, 95% CI [NaN, NaN] excludes 1 (RoR 2.353 ms, Rubble GPU 0.692 ms) |
| PASS | idle N=1: Rubble pipelined wall 0.705 ms < RoR mean 2.353 ms |
| PASS | idle N=1: Rubble p99 2.433 < RoR 2.997 ms, max 2.523 < 3.116 ms |
| PASS | drive N=1 vehicle 0: node RMS 985.90 mm <= 1604.37 mm (1.5x RoR's own 1069.58 mm, at least 1 mm) |
| PASS | drive N=1 vehicle 0: broken 13 within RoR's 8–11 widened 10% (7–13) |
| PASS | drive N=1 vehicle 0: bent 15 within RoR's 14–20 widened 10% (12–22) |
| FAIL | drive N=1: speed-up 3.26x, 95% CI [NaN, NaN] excludes 1 (RoR 2.505 ms, Rubble GPU 0.769 ms) |
| PASS | drive N=1: Rubble pipelined wall 0.783 ms < RoR mean 2.505 ms |
| PASS | drive N=1: Rubble p99 2.383 < RoR 3.305 ms, max 2.630 < 3.469 ms |
| FAIL | impact N=1 vehicle 0: node RMS 1355.44 mm <= 1195.17 mm (1.5x RoR's own 796.78 mm, at least 1 mm) |
| FAIL | impact N=1 vehicle 0: broken 303 within RoR's 339–346 widened 10% (305–381) |
| PASS | impact N=1 vehicle 0: bent 2052 within RoR's 2073–2078 widened 10% (1865–2286) |
| FAIL | impact N=1: speed-up 4.97x, 95% CI [NaN, NaN] excludes 1 (RoR 5.454 ms, Rubble GPU 1.097 ms) |
| PASS | impact N=1: Rubble pipelined wall 1.113 ms < RoR mean 5.454 ms |
| PASS | impact N=1: Rubble p99 3.429 < RoR 6.671 ms, max 3.936 < 6.818 ms |

**NOT MET** (6 failing)
