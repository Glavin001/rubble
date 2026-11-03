# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`ViperGTS.car`: 276 nodes, 1469 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

Bounds (ceiling 33.3 ms for 10 consecutive frames, max_frame_ms 33.3 predicted, deadline_s 780 of running time): engines stopped or not run from: pileup RoR from 512 vehicles; pileup Rubble from 4096 vehicles. All trials ran.

## pileup

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.244 [0.228, 0.259] | 0.268 | 0.533 [0.532, 0.535] | 0.557 | 0.543 [0.541, 0.546] | 0.786 [0.720, 0.851] | 0.46× [0.43, 0.49] | RoR faster | 533.13 |
| 2 | 0.943 [0.939, 0.947] | 1.087 | 0.820 [0.797, 0.843] | 1.948 | 0.831 [0.807, 0.855] | 1.082 [1.044, 1.119] | 1.15× [1.11, 1.19] | Rubble faster | 410.11 |
| 4 | 1.055 [1.028, 1.082] | 1.331 | 1.145 [1.122, 1.169] | 2.042 | 1.157 [1.134, 1.179] | 1.414 [1.396, 1.432] | 0.92× [0.90, 0.95] | RoR faster | 286.36 |
| 8 | 1.322 [1.317, 1.327] | 1.577 | 1.803 [1.766, 1.839] | 2.093 | 1.814 [1.777, 1.851] | 2.069 [2.029, 2.108] | 0.73× [0.72, 0.75] | RoR faster | 225.32 |
| 16 | 1.773 [1.650, 1.895] | 2.029 | 1.884 [1.853, 1.915] | 2.178 | 1.895 [1.864, 1.926] | 2.171 [2.162, 2.180] | 0.94× [0.88, 1.00] | inconclusive | 117.74 |
| 32 | 2.584 [2.535, 2.633] | 2.894 | 1.873 [1.835, 1.911] | 2.213 | 1.884 [1.848, 1.921] | 2.191 [2.109, 2.274] | 1.38× [1.34, 1.43] | Rubble faster | 58.52 |
| 64 | 4.551 [4.479, 4.624] | 4.997 | 2.098 [2.079, 2.117] | 2.354 | 2.109 [2.089, 2.129] | 2.380 [2.353, 2.406] | 2.17× [2.15, 2.19] | Rubble faster | 32.78 |
| 128 | 9.946 [9.227, 10.665] | 11.205 | 2.693 [2.637, 2.749] | 3.104 | 2.704 [2.649, 2.759] | 3.004 [2.978, 3.031] | 3.69× [3.47, 3.93] | Rubble faster | 21.04 |
| 256 | 25.543 [25.266, 25.819] | 28.273 | 3.811 [3.799, 3.823] | 4.246 | 3.823 [3.811, 3.835] | 4.135 [4.097, 4.173] | 6.70× [6.65, 6.76] | Rubble faster | 14.89 |
| 512 | — | — | 6.937 [6.811, 7.062] | 7.849 | 6.950 [6.825, 7.075] | 7.298 [7.193, 7.404] | — | — | 13.55 |
| 1024 | — | — | 11.015 [10.914, 11.115] | 11.830 | 11.027 [10.927, 11.127] | 11.340 [11.265, 11.415] | — | — | 10.76 |
| 2048 | — | — | 20.416 [20.333, 20.499] | 22.228 | 20.428 [20.343, 20.513] | 20.760 [20.726, 20.794] | — | — | 9.97 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.282 | 0.294 | 0.0 % | 0.614 | 0.634 | 0.0 % | 0.934 | 0.997 | 0.46× |
| 2 | 1.409 | 1.505 | 0.0 % | 2.131 | 2.165 | 0.0 % | 2.392 | 2.605 | 0.69× |
| 4 | 1.490 | 1.623 | 0.0 % | 2.174 | 2.237 | 0.0 % | 2.458 | 2.588 | 0.73× |
| 8 | 1.728 | 1.823 | 0.0 % | 2.203 | 2.301 | 0.0 % | 2.504 | 2.570 | 0.79× |
| 16 | 2.524 | 3.873 | 0.0 % | 2.285 | 2.365 | 0.0 % | 2.601 | 2.661 | 1.64× |
| 32 | 3.047 | 3.186 | 0.0 % | 2.277 | 2.321 | 0.0 % | 2.600 | 2.641 | 1.37× |
| 64 | 5.320 | 5.490 | 0.0 % | 2.448 | 2.526 | 0.0 % | 2.754 | 2.893 | 2.17× |
| 128 | 11.869 | 13.381 | 100.0 % | 3.163 | 3.235 | 0.0 % | 3.486 | 6.109 | 4.14× |
| 256 | 28.704 | 31.770 | 100.0 % | 4.469 | 5.034 | 0.0 % | 4.949 | 8.242 | 6.31× |
| 512 | — | — | — | 8.165 | 8.601 | 0.6 % | 8.550 | 9.101 | — |
| 1024 | — | — | — | 12.080 | 12.572 | 100.0 % | 12.437 | 12.981 | — |
| 2048 | — | — | — | 22.560 | 23.005 | 100.0 % | 22.919 | 23.312 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 512, Rubble blocking wall 256, Rigs of Rods 64 (of counts up to 1024).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 1.378 ms + 9.40 µs × N; Rigs of Rods -0.029 ms + 94.37 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 66, Rigs of Rods 21.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 278, Rigs of Rods 42.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 739, Rigs of Rods 88.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 19.84 | 36.94 | 0 / 0 | 0 / 0 | 71.529 / 71.522 | -1.59 / -1.52 |
| 2 | 0 | 2067.91 | 2749.83 | 0 / 0 | 79 / 71 | 55.157 / 57.165 | 10.58 / 2.00 |
| 2 | 1 | 1780.41 | 2419.78 | 0 / 0 | 88 / 77 | -55.463 / -57.189 | 16.30 / 7.25 |
| 2 | 0 | 2067.91 | 2749.83 | 0 / 0 | 79 / 71 | 55.157 / 57.165 | 10.58 / 2.00 |
| 2 | 1 | 1780.41 | 2419.78 | 0 / 0 | 88 / 77 | -55.463 / -57.189 | 16.30 / 7.25 |
| 4 | 0 | 2242.79 | 2827.52 | 0 / 0 | 79 / 70 | 55.034 / 57.172 | 13.19 / 2.40 |
| 4 | 1 | 2209.54 | 2697.61 | 0 / 0 | 88 / 74 | -55.189 / -57.365 | 21.16 / 15.02 |
| 4 | 2 | 441.78 | 619.39 | 0 / 0 | 17 / 18 | 43.144 / 43.236 | -1.81 / -2.29 |
| 4 | 3 | 952.89 | 1685.25 | 1 / 0 | 6 / 5 | 8.073 / 8.548 | -28.51 / -129.77 |
| 8 | 0 | 2289.09 | 3062.54 | 0 / 0 | 79 / 72 | 55.026 / 57.210 | 13.40 / 3.88 |
| 8 | 1 | 2138.77 | 2909.95 | 0 / 0 | 88 / 73 | -55.189 / -57.258 | 21.99 / 4.85 |
| 8 | 6 | 541.12 | 1033.11 | 0 / 0 | 104 / 100 | 14.513 / 14.112 | 18.07 / 22.85 |
| 8 | 7 | 465.75 | 879.52 | 0 / 0 | 95 / 102 | -14.718 / -14.485 | 27.97 / 21.65 |
| 16 | 0 | 1975.76 | 2750.81 | 0 / 0 | 79 / 72 | 55.041 / 56.885 | 13.64 / 2.33 |
| 16 | 1 | 2023.38 | 2542.58 | 0 / 0 | 88 / 72 | -55.200 / -57.122 | 22.27 / 2.59 |
| 16 | 14 | 3055.25 | 3420.68 | 0 / 0 | 62 / 55 | 43.142 / 43.344 | 4.06 / 2.87 |
| 16 | 15 | 3570.85 | 4144.72 | 1 / 0 | 27 / 25 | 30.151 / 30.122 | -238.24 / -230.42 |
| 32 | 0 | 1001.95 | 1663.23 | 0 / 0 | 79 / 69 | 57.515 / 57.091 | 20.18 / 5.61 |
| 32 | 1 | 827.53 | 1374.48 | 0 / 0 | 87 / 75 | -57.108 / -56.974 | 47.46 / 6.83 |
| 32 | 30 | 1534.82 | 1990.89 | 0 / 0 | 126 / 126 | 19.819 / 21.242 | 87.64 / 36.10 |
| 32 | 31 | 1393.48 | 2067.98 | 0 / 0 | 122 / 127 | -19.781 / -21.122 | 41.58 / 25.00 |
| 64 | 0 | 2097.68 | 2635.27 | 0 / 0 | 79 / 74 | 55.030 / 57.039 | 13.20 / 3.39 |
| 64 | 1 | 1974.35 | 2494.56 | 0 / 0 | 88 / 72 | -55.199 / -57.119 | 22.42 / 5.77 |
| 64 | 62 | 1317.34 | 1800.37 | 0 / 0 | 62 / 57 | 44.890 / 43.673 | 8.45 / 3.41 |
| 64 | 63 | 9054.41 | 13153.75 | 0 / 1 | 24 / 22 | 25.382 / 17.456 | -235.53 / -235.90 |
| 128 | 0 | 2344.60 | 3163.68 | 0 / 0 | 79 / 68 | 55.044 / 57.157 | 12.80 / 20.44 |
| 128 | 1 | 1741.20 | 2296.59 | 0 / 0 | 88 / 71 | -55.198 / -56.802 | 20.56 / 40.73 |
| 128 | 126 | 1326.64 | 1995.75 | 0 / 0 | 102 / 92 | 29.949 / 31.066 | 17.99 / 30.51 |
| 128 | 127 | 1559.78 | 2413.13 | 0 / 0 | 98 / 93 | -29.476 / -30.895 | 17.59 / 15.60 |
| 256 | 0 | 2546.87 | 3524.10 | 0 / 0 | 81 / 71 | 55.066 / 57.431 | 9.19 / 19.96 |
| 256 | 1 | 1856.44 | 2622.65 | 0 / 0 | 88 / 78 | -55.289 / -57.049 | 22.84 / 45.15 |
| 256 | 254 | 310.92 | 560.99 | 0 / 0 | 12 / 14 | 43.295 / 43.511 | -4.00 / 0.00 |
| 256 | 255 | 507.44 | 1069.55 | 1 / 0 | 10 / 7 | 5.943 / 5.868 | -9.08 / -23.68 |
## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/viper/ViperGTS.car ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=pileup counts=1,2,4,8,16,32,64,128,256,512,1024 rubble_counts=2048,4096,8192 fps=120 frames=360 speed=27.78 drag=0 trials=3 nudges=0 quiet_wait=1800 max_load=8 ceiling_ms=33.3 ceiling_frames=10 deadline_s=780 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/sweep/viper`

| result | criterion |
|---|---|
| PASS | 63 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| FAIL | pileup N=1: speed-up 0.46x, 95% CI [0.43, 0.49] excludes 1 (RoR 0.244 ms, Rubble GPU 0.533 ms) |
| FAIL | pileup N=1: Rubble pipelined wall 0.543 ms < RoR mean 0.244 ms |
| FAIL | pileup N=1: Rubble p99 0.614 < RoR 0.282 ms, max 0.634 < 0.294 ms |
| PASS | pileup N=2: speed-up 1.15x, 95% CI [1.11, 1.19] excludes 1 (RoR 0.943 ms, Rubble GPU 0.820 ms) |
| PASS | pileup N=2: Rubble pipelined wall 0.831 ms < RoR mean 0.943 ms |
| FAIL | pileup N=2: Rubble p99 2.131 < RoR 1.409 ms, max 2.165 < 1.505 ms |
| FAIL | pileup N=4: speed-up 0.92x, 95% CI [0.90, 0.95] excludes 1 (RoR 1.055 ms, Rubble GPU 1.145 ms) |
| FAIL | pileup N=4: Rubble pipelined wall 1.157 ms < RoR mean 1.055 ms |
| FAIL | pileup N=4: Rubble p99 2.174 < RoR 1.490 ms, max 2.237 < 1.623 ms |
| FAIL | pileup N=8: speed-up 0.73x, 95% CI [0.72, 0.75] excludes 1 (RoR 1.322 ms, Rubble GPU 1.803 ms) |
| FAIL | pileup N=8: Rubble pipelined wall 1.814 ms < RoR mean 1.322 ms |
| FAIL | pileup N=8: Rubble p99 2.203 < RoR 1.728 ms, max 2.301 < 1.823 ms |
| FAIL | pileup N=16: speed-up 0.94x, 95% CI [0.88, 1.00] excludes 1 (RoR 1.773 ms, Rubble GPU 1.884 ms) |
| FAIL | pileup N=16: Rubble pipelined wall 1.895 ms < RoR mean 1.773 ms |
| PASS | pileup N=16: Rubble p99 2.285 < RoR 2.524 ms, max 2.365 < 3.873 ms |
| PASS | pileup N=32: speed-up 1.38x, 95% CI [1.34, 1.43] excludes 1 (RoR 2.584 ms, Rubble GPU 1.873 ms) |
| PASS | pileup N=32: Rubble pipelined wall 1.884 ms < RoR mean 2.584 ms |
| PASS | pileup N=32: Rubble p99 2.277 < RoR 3.047 ms, max 2.321 < 3.186 ms |
| PASS | pileup N=64: speed-up 2.17x, 95% CI [2.15, 2.19] excludes 1 (RoR 4.551 ms, Rubble GPU 2.098 ms) |
| PASS | pileup N=64: Rubble pipelined wall 2.109 ms < RoR mean 4.551 ms |
| PASS | pileup N=64: Rubble p99 2.448 < RoR 5.320 ms, max 2.526 < 5.490 ms |
| PASS | pileup N=128: speed-up 3.69x, 95% CI [3.47, 3.93] excludes 1 (RoR 9.946 ms, Rubble GPU 2.693 ms) |
| PASS | pileup N=128: Rubble pipelined wall 2.704 ms < RoR mean 9.946 ms |
| PASS | pileup N=128: Rubble p99 3.163 < RoR 11.869 ms, max 3.235 < 13.381 ms |
| PASS | pileup N=256: speed-up 6.70x, 95% CI [6.65, 6.76] excludes 1 (RoR 25.543 ms, Rubble GPU 3.811 ms) |
| PASS | pileup N=256: Rubble pipelined wall 3.823 ms < RoR mean 25.543 ms |
| PASS | pileup N=256: Rubble p99 4.469 < RoR 28.704 ms, max 5.034 < 31.770 ms |

**NOT MET** (12 failing)
