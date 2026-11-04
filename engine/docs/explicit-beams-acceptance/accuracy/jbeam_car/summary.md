# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`rubble_hires_jbeam.truck`: 1110 nodes, 6158 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.449 | 2.918 | 0.506 | 0.605 | 0.517 | 0.872 | 4.84× | inconclusive | 506.30 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.149 | 3.218 | 0.0 % | 0.663 | 0.688 | 0.0 % | 1.333 | 1.424 | 4.67× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2.74 | 24.59 | 4 / 5 | 14 / 13 | -0.011 / -0.010 | -1.59 / -1.62 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 4.58 mm, RoR–RoR 3.94 mm, ratio 1.16 [95 % bootstrap 0.56, 1.53]; broken: Rubble median 4.5, RoR 3.0–12.0; bent: Rubble median 11.5, RoR 13.0–15.0.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.520 | 3.032 | 0.647 | 1.258 | 0.659 | 1.063 | 3.90× | inconclusive | 646.83 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.203 | 3.609 | 0.0 % | 1.300 | 1.309 | 0.0 % | 1.879 | 2.149 | 2.76× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1012.30 | 1063.54 | 8 / 8 | 11 / 14 | 69.132 / 68.977 | -4.19 / -4.38 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 860.23 mm, RoR–RoR 778.30 mm, ratio 1.11 [95 % bootstrap 0.54, 3.01]; broken: Rubble median 12.5, RoR 8.0–22.0; bent: Rubble median 15.5, RoR 9.0–20.0.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.329 | 6.226 | 0.880 | 1.460 | 0.893 | 1.400 | 6.05× | inconclusive | 880.43 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.844 | 7.048 | 0.0 % | 1.527 | 1.543 | 0.0 % | 2.610 | 2.806 | 4.57× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2617.83 | 3712.84 | 363 / 339 | 2108 / 2073 | 11.634 / 13.662 | 95.11 / 125.46 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 1440.29 mm, RoR–RoR 1958.02 mm, ratio 0.74 [95 % bootstrap 0.22, 2.36]; broken: Rubble median 358.0, RoR 319.0–506.0; bent: Rubble median 2122.5, RoR 2021.0–2182.0.

## Acceptance

Command: `./target/release/explicit_beams_scaling docs/explicit-beams-acceptance/vehicles/rubble_hires_jbeam.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 nudges=8 max_load=1000 max_gpu=100000 out=docs/explicit-beams-acceptance/accuracy/jbeam_car`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100000 %): 0 did not |
| PASS | idle N=1 (8 runs per engine): median node RMS Rubble–RoR 4.58 mm <= 5.91 mm (1.5x RoR–RoR median 3.94 mm, at least 1 mm); ratio 1.16 [95 % 0.56, 1.53] |
| PASS | idle N=1: Rubble median broken 4.5 within RoR's 3.0–12.0 widened 10% (2–14) |
| PASS | idle N=1: Rubble median bent 11.5 within RoR's 13.0–15.0 widened 10% (11–17) |
| FAIL | idle N=1: speed-up 4.84x, 95% CI [NaN, NaN] excludes 1 (RoR 2.449 ms, Rubble GPU 0.506 ms) |
| PASS | idle N=1: Rubble pipelined wall 0.517 ms < RoR mean 2.449 ms |
| PASS | idle N=1: Rubble p99 0.663 < RoR 3.149 ms, max 0.688 < 3.218 ms |
| PASS | drive N=1 (8 runs per engine): median node RMS Rubble–RoR 860.23 mm <= 1167.45 mm (1.5x RoR–RoR median 778.30 mm, at least 1 mm); ratio 1.11 [95 % 0.54, 3.01] |
| PASS | drive N=1: Rubble median broken 12.5 within RoR's 8.0–22.0 widened 10% (7–25) |
| PASS | drive N=1: Rubble median bent 15.5 within RoR's 9.0–20.0 widened 10% (8–22) |
| FAIL | drive N=1: speed-up 3.90x, 95% CI [NaN, NaN] excludes 1 (RoR 2.520 ms, Rubble GPU 0.647 ms) |
| PASS | drive N=1: Rubble pipelined wall 0.659 ms < RoR mean 2.520 ms |
| PASS | drive N=1: Rubble p99 1.300 < RoR 3.203 ms, max 1.309 < 3.609 ms |
| PASS | impact N=1 (8 runs per engine): median node RMS Rubble–RoR 1440.29 mm <= 2937.02 mm (1.5x RoR–RoR median 1958.02 mm, at least 1 mm); ratio 0.74 [95 % 0.22, 2.36] |
| PASS | impact N=1: Rubble median broken 358 within RoR's 319.0–506.0 widened 10% (287–557) |
| PASS | impact N=1: Rubble median bent 2122.5 within RoR's 2021.0–2182.0 widened 10% (1818–2401) |
| FAIL | impact N=1: speed-up 6.05x, 95% CI [NaN, NaN] excludes 1 (RoR 5.329 ms, Rubble GPU 0.880 ms) |
| PASS | impact N=1: Rubble pipelined wall 0.893 ms < RoR mean 5.329 ms |
| PASS | impact N=1: Rubble p99 1.527 < RoR 6.844 ms, max 1.543 < 7.048 ms |

**NOT MET** (3 failing)
