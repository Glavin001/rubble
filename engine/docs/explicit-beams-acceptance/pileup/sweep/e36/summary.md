# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`E36Sedan.truck`: 487 nodes, 2898 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

Bounds (ceiling 33.3 ms for 10 consecutive frames, max_frame_ms 33.3 predicted, deadline_s 780 of running time): engines stopped or not run from: pileup RoR from 256 vehicles; pileup Rubble from 1024 vehicles. All trials ran.

## pileup

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.506 [0.488, 0.524] | 0.548 | 0.688 [0.684, 0.692] | 0.728 | 0.698 [0.695, 0.700] | 0.935 [0.895, 0.975] | 0.74× [0.71, 0.76] | RoR faster | 688.05 |
| 2 | 1.261 [1.095, 1.427] | 1.416 | 1.233 [1.163, 1.303] | 2.544 | 1.244 [1.175, 1.313] | 1.525 [1.398, 1.652] | 1.02× [0.85, 1.22] | inconclusive | 616.56 |
| 4 | 1.461 [1.455, 1.467] | 2.048 | 1.656 [1.564, 1.748] | 3.049 | 1.667 [1.574, 1.759] | 1.944 [1.892, 1.995] | 0.88× [0.84, 0.93] | RoR faster | 414.00 |
| 8 | 1.858 [1.836, 1.879] | 2.638 | 2.264 [2.214, 2.315] | 3.176 | 2.276 [2.225, 2.326] | 2.534 [2.463, 2.604] | 0.82× [0.79, 0.85] | RoR faster | 283.02 |
| 16 | 2.786 [2.586, 2.986] | 3.627 | 2.399 [2.349, 2.448] | 3.311 | 2.410 [2.361, 2.460] | 2.717 [2.563, 2.872] | 1.16× [1.06, 1.27] | Rubble faster | 149.93 |
| 32 | 4.320 [4.265, 4.376] | 5.225 | 3.206 [3.194, 3.217] | 4.342 | 3.217 [3.202, 3.231] | 3.471 [3.431, 3.512] | 1.35× [1.33, 1.36] | Rubble faster | 100.17 |
| 64 | 8.378 [8.001, 8.754] | 9.368 | 3.471 [3.363, 3.578] | 4.233 | 3.482 [3.375, 3.588] | 3.776 [3.684, 3.867] | 2.41× [2.37, 2.45] | Rubble faster | 54.23 |
| 128 | 17.836 [17.700, 17.971] | 19.442 | 5.598 [5.455, 5.741] | 6.571 | 5.610 [5.466, 5.754] | 6.132 [4.852, 7.412] | 3.19× [3.12, 3.25] | Rubble faster | 43.73 |
| 256 | — | — | 9.658 [9.647, 9.670] | 11.222 | 9.670 [9.658, 9.683] | 9.981 [9.851, 10.112] | — | — | 37.73 |
| 512 | — | — | 17.079 [16.996, 17.162] | 19.246 | 17.092 [17.008, 17.176] | 17.423 [17.244, 17.603] | — | — | 33.36 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.562 | 0.650 | 0.0 % | 0.743 | 0.769 | 0.0 % | 1.101 | 1.220 | 0.85× |
| 2 | 2.729 | 3.088 | 0.0 % | 3.303 | 3.459 | 0.0 % | 3.575 | 3.724 | 0.89× |
| 4 | 2.809 | 2.962 | 0.0 % | 3.335 | 3.556 | 0.0 % | 3.643 | 3.793 | 0.83× |
| 8 | 3.052 | 3.210 | 0.0 % | 3.391 | 3.511 | 0.0 % | 3.695 | 3.811 | 0.91× |
| 16 | 3.810 | 4.086 | 0.0 % | 3.477 | 3.560 | 0.0 % | 3.794 | 3.986 | 1.15× |
| 32 | 5.677 | 5.959 | 0.0 % | 4.966 | 6.566 | 0.0 % | 5.282 | 9.661 | 0.91× |
| 64 | 12.161 | 22.036 | 51.0 % | 4.486 | 4.870 | 0.0 % | 5.021 | 10.043 | 4.53× |
| 128 | 19.891 | 22.251 | 100.0 % | 7.083 | 7.922 | 0.0 % | 9.116 | 11.220 | 2.81× |
| 256 | — | — | — | 11.906 | 13.073 | 84.9 % | 12.037 | 13.076 | — |
| 512 | — | — | — | 20.074 | 20.673 | 100.0 % | 20.184 | 22.054 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 128, Rubble blocking wall 64, Rigs of Rods 32 (of counts up to 512).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 1.596 ms + 30.59 µs × N; Rigs of Rods 0.614 ms + 131.34 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 13, Rigs of Rods 10.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 78, Rigs of Rods 25.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 220, Rigs of Rods 58.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 42.01 | 52.67 | 0 / 0 | 12 / 13 | 76.287 / 76.294 | -0.21 / -0.27 |
| 2 | 0 | 543.76 | 1184.10 | 74 / 76 | 442 / 469 | 56.480 / 56.750 | 6.23 / 3.23 |
| 2 | 1 | 773.47 | 1490.83 | 64 / 62 | 471 / 480 | -56.313 / -57.001 | 30.55 / 29.03 |
| 2 | 0 | 543.76 | 1184.10 | 74 / 76 | 442 / 469 | 56.480 / 56.750 | 6.23 / 3.23 |
| 2 | 1 | 773.47 | 1490.83 | 64 / 62 | 471 / 480 | -56.313 / -57.001 | 30.55 / 29.03 |
| 4 | 0 | 898.19 | 3141.18 | 65 / 81 | 461 / 466 | 57.450 / 56.871 | 7.16 / 31.34 |
| 4 | 1 | 577.27 | 2801.87 | 84 / 71 | 451 / 491 | -57.100 / -56.771 | -32.13 / 15.45 |
| 4 | 2 | 515.62 | 1032.02 | 11 / 11 | 340 / 335 | 45.461 / 45.226 | 9.91 / 11.90 |
| 4 | 3 | 807.94 | 1517.20 | 1 / 1 | 341 / 349 | 10.547 / 9.931 | -235.01 / -184.67 |
| 8 | 0 | 754.78 | 1376.07 | 83 / 53 | 454 / 473 | 56.582 / 57.094 | 17.72 / 34.13 |
| 8 | 1 | 388.29 | 944.88 | 61 / 52 | 472 / 483 | -56.772 / -56.552 | 22.08 / 30.31 |
| 8 | 6 | 1228.46 | 3828.02 | 88 / 74 | 424 / 431 | 14.499 / 13.521 | -79.77 / 4.19 |
| 8 | 7 | 1550.94 | 12340.40 | 75 / 72 | 451 / 448 | -14.557 / -14.281 | -647.92 / -1759.94 |
| 16 | 0 | 931.73 | 1618.53 | 66 / 62 | 486 / 473 | 56.161 / 57.018 | 11.14 / 9.53 |
| 16 | 1 | 1162.75 | 1824.99 | 74 / 51 | 487 / 481 | -56.107 / -57.233 | 4.14 / 19.46 |
| 16 | 14 | 4095.60 | 6552.07 | 67 / 28 | 401 / 428 | 45.980 / 49.821 | 23.08 / 20.54 |
| 16 | 15 | 10556.17 | 13384.31 | 2 / 2 | 422 / 452 | 26.241 / 28.392 | -195.57 / -241.18 |
| 32 | 0 | 787.66 | 2182.25 | 73 / 75 | 447 / 480 | 57.040 / 56.560 | 17.57 / 13.67 |
| 32 | 1 | 849.13 | 1464.47 | 55 / 68 | 488 / 497 | -57.124 / -56.344 | 22.74 / -26.40 |
| 32 | 30 | 3197.87 | 24634.25 | 79 / 78 | 495 / 499 | 22.171 / 20.672 | -2275.20 / 12.60 |
| 32 | 31 | 1882.43 | 15496.63 | 87 / 67 | 501 / 509 | -22.111 / -21.524 | -1405.29 / 37.91 |
| 64 | 0 | 592.73 | 2400.02 | 73 / 62 | 497 / 478 | 56.645 / 56.304 | 21.36 / 7.80 |
| 64 | 1 | 543.94 | 2890.38 | 96 / 47 | 486 / 503 | -56.527 / -56.663 | -116.15 / 15.68 |
| 64 | 62 | 749.00 | 1273.04 | 11 / 11 | 405 / 409 | 49.336 / 49.917 | 22.89 / 16.50 |
| 64 | 63 | 1216.52 | 1617.26 | 2 / 3 | 396 / 420 | 24.775 / 23.815 | -238.56 / -229.39 |
| 128 | 0 | 973.10 | 1776.34 | 62 / 69 | 491 / 474 | 55.904 / 56.786 | 13.94 / 41.10 |
| 128 | 1 | 782.60 | 1451.09 | 71 / 68 | 501 / 474 | -55.849 / -56.563 | -12.60 / 40.56 |
| 128 | 126 | 1659.78 | 4179.46 | 85 / 76 | 452 / 491 | 31.190 / 29.625 | 6.97 / 7.90 |
| 128 | 127 | 2049.28 | 4467.71 | 64 / 81 | 475 / 488 | -31.500 / -29.550 | 14.47 / -91.49 |
## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/e36/E36Sedan.truck ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=pileup counts=1,2,4,8,16,32,64,128,256,512,1024 rubble_counts=2048,4096,8192 fps=120 frames=360 speed=27.78 drag=0 trials=3 nudges=0 quiet_wait=1800 max_load=8 ceiling_ms=33.3 ceiling_frames=10 deadline_s=780 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/sweep/e36`

| result | criterion |
|---|---|
| PASS | 54 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| FAIL | pileup N=1: speed-up 0.74x, 95% CI [0.71, 0.76] excludes 1 (RoR 0.506 ms, Rubble GPU 0.688 ms) |
| FAIL | pileup N=1: Rubble pipelined wall 0.698 ms < RoR mean 0.506 ms |
| FAIL | pileup N=1: Rubble p99 0.743 < RoR 0.562 ms, max 0.769 < 0.650 ms |
| FAIL | pileup N=2: speed-up 1.02x, 95% CI [0.85, 1.22] excludes 1 (RoR 1.261 ms, Rubble GPU 1.233 ms) |
| PASS | pileup N=2: Rubble pipelined wall 1.244 ms < RoR mean 1.261 ms |
| FAIL | pileup N=2: Rubble p99 3.303 < RoR 2.729 ms, max 3.459 < 3.088 ms |
| FAIL | pileup N=4: speed-up 0.88x, 95% CI [0.84, 0.93] excludes 1 (RoR 1.461 ms, Rubble GPU 1.656 ms) |
| FAIL | pileup N=4: Rubble pipelined wall 1.667 ms < RoR mean 1.461 ms |
| FAIL | pileup N=4: Rubble p99 3.335 < RoR 2.809 ms, max 3.556 < 2.962 ms |
| FAIL | pileup N=8: speed-up 0.82x, 95% CI [0.79, 0.85] excludes 1 (RoR 1.858 ms, Rubble GPU 2.264 ms) |
| FAIL | pileup N=8: Rubble pipelined wall 2.276 ms < RoR mean 1.858 ms |
| FAIL | pileup N=8: Rubble p99 3.391 < RoR 3.052 ms, max 3.511 < 3.210 ms |
| PASS | pileup N=16: speed-up 1.16x, 95% CI [1.06, 1.27] excludes 1 (RoR 2.786 ms, Rubble GPU 2.399 ms) |
| PASS | pileup N=16: Rubble pipelined wall 2.410 ms < RoR mean 2.786 ms |
| PASS | pileup N=16: Rubble p99 3.477 < RoR 3.810 ms, max 3.560 < 4.086 ms |
| PASS | pileup N=32: speed-up 1.35x, 95% CI [1.33, 1.36] excludes 1 (RoR 4.320 ms, Rubble GPU 3.206 ms) |
| PASS | pileup N=32: Rubble pipelined wall 3.217 ms < RoR mean 4.320 ms |
| FAIL | pileup N=32: Rubble p99 4.966 < RoR 5.677 ms, max 6.566 < 5.959 ms |
| PASS | pileup N=64: speed-up 2.41x, 95% CI [2.37, 2.45] excludes 1 (RoR 8.378 ms, Rubble GPU 3.471 ms) |
| PASS | pileup N=64: Rubble pipelined wall 3.482 ms < RoR mean 8.378 ms |
| PASS | pileup N=64: Rubble p99 4.486 < RoR 12.161 ms, max 4.870 < 22.036 ms |
| PASS | pileup N=128: speed-up 3.19x, 95% CI [3.12, 3.25] excludes 1 (RoR 17.836 ms, Rubble GPU 5.598 ms) |
| PASS | pileup N=128: Rubble pipelined wall 5.610 ms < RoR mean 17.836 ms |
| PASS | pileup N=128: Rubble p99 7.083 < RoR 19.891 ms, max 7.922 < 22.251 ms |

**NOT MET** (12 failing)
