# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`95bbUID-agoras.truck`: 195 nodes, 862 beams. Apple M3 Max (Metal). 1 trials × 240 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

## idle

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.132 | 0.151 | 0.154 | 0.159 | 0.162 | 0.377 | 0.86× | inconclusive | 153.85 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.165 | 0.194 | 0.0 % | 0.171 | 0.233 | 0.0 % | 0.699 | 0.817 | 0.83× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.15 | 2.56 | 0 / 0 | 0 / 0 | 0.006 / 0.005 | -1.63 / -1.67 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 1.29 mm, max 1.93 mm; broken 0 / 0, bent 0 / 0.

## drive

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.139 | 0.155 | 0.154 | 0.158 | 0.163 | 0.370 | 0.91× | inconclusive | 153.95 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.164 | 0.202 | 0.0 % | 0.184 | 0.223 | 0.0 % | 0.626 | 0.716 | 0.90× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 12.61 | 25.42 | 0 / 0 | 0 / 0 | 63.700 / 63.700 | -1.61 / -1.25 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 8.89 mm, max 19.65 mm; broken 0 / 0, bent 0 / 0.

## impact

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.163 | 0.186 | 0.157 | 0.164 | 0.164 | 0.374 | 1.04× | inconclusive | 156.56 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.199 | 0.211 | 0.0 % | 0.179 | 0.187 | 0.0 % | 0.672 | 0.717 | 1.13× |


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU NaN ms + NaN µs × N; Rigs of Rods NaN ms + NaN µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 0, Rigs of Rods 0.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 2148.57 | 2979.06 | 21 / 19 | 49 / 48 | 10.991 / 8.946 | 135.64 / 129.07 |

Rigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS 638.77 mm, max 1042.06 mm; broken 19 / 19, bent 48 / 51.

## Acceptance

Command: `./target/release/explicit_beams_scaling /Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-content/agora/95bbUID-agoras.truck ror=/Users/glavin/Development/rubble-beamng-parity/engine/../inspiration/ror-harness/bin/ror_ref scenarios=idle,drive,impact counts=1 rubble_counts= fps=120 speed=27.78 drag=0 trials=1 max_load=1000 max_gpu=100000 out=docs/explicit-beams-acceptance/accuracy/agora_s`

| result | criterion |
|---|---|
| PASS | 6 runs started on a quiet machine (load < 1000, GPU < 100000 %): 0 did not |
| PASS | idle N=1 vehicle 0: node RMS 1.15 mm <= 1.94 mm (1.5x RoR's own 1.29 mm, at least 1 mm) |
| PASS | idle N=1 vehicle 0: broken 0 within RoR's 0–0 widened 10% (0–0) |
| PASS | idle N=1 vehicle 0: bent 0 within RoR's 0–0 widened 10% (0–0) |
| FAIL | idle N=1: speed-up 0.86x, 95% CI [NaN, NaN] excludes 1 (RoR 0.132 ms, Rubble GPU 0.154 ms) |
| FAIL | idle N=1: Rubble pipelined wall 0.162 ms < RoR mean 0.132 ms |
| FAIL | idle N=1: Rubble p99 0.171 < RoR 0.165 ms, max 0.233 < 0.194 ms |
| PASS | drive N=1 vehicle 0: node RMS 12.61 mm <= 13.33 mm (1.5x RoR's own 8.89 mm, at least 1 mm) |
| PASS | drive N=1 vehicle 0: broken 0 within RoR's 0–0 widened 10% (0–0) |
| PASS | drive N=1 vehicle 0: bent 0 within RoR's 0–0 widened 10% (0–0) |
| FAIL | drive N=1: speed-up 0.91x, 95% CI [NaN, NaN] excludes 1 (RoR 0.139 ms, Rubble GPU 0.154 ms) |
| FAIL | drive N=1: Rubble pipelined wall 0.163 ms < RoR mean 0.139 ms |
| FAIL | drive N=1: Rubble p99 0.184 < RoR 0.164 ms, max 0.223 < 0.202 ms |
| FAIL | impact N=1 vehicle 0: node RMS 2148.57 mm <= 958.15 mm (1.5x RoR's own 638.77 mm, at least 1 mm) |
| PASS | impact N=1 vehicle 0: broken 21 within RoR's 19–19 widened 10% (17–21) |
| PASS | impact N=1 vehicle 0: bent 49 within RoR's 48–51 widened 10% (43–57) |
| FAIL | impact N=1: speed-up 1.04x, 95% CI [NaN, NaN] excludes 1 (RoR 0.163 ms, Rubble GPU 0.157 ms) |
| FAIL | impact N=1: Rubble pipelined wall 0.164 ms < RoR mean 0.163 ms |
| PASS | impact N=1: Rubble p99 0.179 < RoR 0.199 ms, max 0.187 < 0.211 ms |

**NOT MET** (9 failing)
