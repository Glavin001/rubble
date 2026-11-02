# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`95bbUID-agoras.truck`: 195 nodes, 862 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.126 | 0.136 | 0.159 | 0.161 | 0.170 | 0.414 | 0.79× | inconclusive | 159.36 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.158 | 0.167 | 0.0 % | 0.478 | 0.516 | 0.0 % | 0.807 | 0.879 | 0.32× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.15 | 2.56 | 0 / 0 | 0 / 0 | 0.006 / 0.005 | -1.63 / -1.67 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 3.71 mm, RoR–RoR 1.49 mm, ratio 2.49 [95 % bootstrap 1.31, 4.92]; broken: Rubble median 0.0, RoR 0.0–0.0; bent: Rubble median 0.0, RoR 0.0–0.0.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.127 | 0.141 | 0.158 | 0.162 | 0.170 | 0.377 | 0.80× | inconclusive | 158.31 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.167 | 0.182 | 0.0 % | 0.354 | 0.541 | 0.0 % | 0.870 | 0.897 | 0.34× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 12.61 | 25.42 | 0 / 0 | 0 / 0 | 63.700 / 63.700 | -1.61 / -1.25 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 13.16 mm, RoR–RoR 9.13 mm, ratio 1.44 [95 % bootstrap 1.14, 2.21]; broken: Rubble median 0.0, RoR 0.0–0.0; bent: Rubble median 0.0, RoR 0.0–0.0.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.150 | 0.166 | 0.173 | 0.195 | 0.185 | 0.434 | 0.87× | inconclusive | 172.59 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.183 | 0.209 | 0.0 % | 0.510 | 0.518 | 0.0 % | 0.907 | 0.997 | 0.40× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2148.57 | 2979.06 | 21 / 19 | 49 / 48 | 10.991 / 8.946 | 135.64 / 129.07 |

Accuracy ensembles (8 runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR 807.85 mm, RoR–RoR 548.00 mm, ratio 1.47 [95 % bootstrap 1.03, 6.09]; broken: Rubble median 19.0, RoR 19.0–19.0; bent: Rubble median 50.0, RoR 48.0–51.0.

## Acceptance

Command: `./target/release/explicit_beams_scaling /Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-content/agora/95bbUID-agoras.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 nudges=8 max_load=1000 max_gpu=100000 out=docs/explicit-beams-acceptance/accuracy/agora_s`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100000 %): 0 did not |
| FAIL | idle N=1 (8 runs per engine): median node RMS Rubble–RoR 3.71 mm <= 2.24 mm (1.5x RoR–RoR median 1.49 mm, at least 1 mm); ratio 2.49 [95 % 1.31, 4.92] |
| PASS | idle N=1: Rubble median broken 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| PASS | idle N=1: Rubble median bent 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| FAIL | idle N=1: speed-up 0.79x, 95% CI [NaN, NaN] excludes 1 (RoR 0.126 ms, Rubble GPU 0.159 ms) |
| FAIL | idle N=1: Rubble pipelined wall 0.170 ms < RoR mean 0.126 ms |
| FAIL | idle N=1: Rubble p99 0.478 < RoR 0.158 ms, max 0.516 < 0.167 ms |
| PASS | drive N=1 (8 runs per engine): median node RMS Rubble–RoR 13.16 mm <= 13.70 mm (1.5x RoR–RoR median 9.13 mm, at least 1 mm); ratio 1.44 [95 % 1.14, 2.21] |
| PASS | drive N=1: Rubble median broken 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| PASS | drive N=1: Rubble median bent 0 within RoR's 0.0–0.0 widened 10% (0–0) |
| FAIL | drive N=1: speed-up 0.80x, 95% CI [NaN, NaN] excludes 1 (RoR 0.127 ms, Rubble GPU 0.158 ms) |
| FAIL | drive N=1: Rubble pipelined wall 0.170 ms < RoR mean 0.127 ms |
| FAIL | drive N=1: Rubble p99 0.354 < RoR 0.167 ms, max 0.541 < 0.182 ms |
| PASS | impact N=1 (8 runs per engine): median node RMS Rubble–RoR 807.85 mm <= 822.01 mm (1.5x RoR–RoR median 548.00 mm, at least 1 mm); ratio 1.47 [95 % 1.03, 6.09] |
| PASS | impact N=1: Rubble median broken 19 within RoR's 19.0–19.0 widened 10% (17–21) |
| PASS | impact N=1: Rubble median bent 50 within RoR's 48.0–51.0 widened 10% (43–57) |
| FAIL | impact N=1: speed-up 0.87x, 95% CI [NaN, NaN] excludes 1 (RoR 0.150 ms, Rubble GPU 0.173 ms) |
| FAIL | impact N=1: Rubble pipelined wall 0.185 ms < RoR mean 0.150 ms |
| FAIL | impact N=1: Rubble p99 0.510 < RoR 0.183 ms, max 0.518 < 0.209 ms |

**NOT MET** (10 failing)
