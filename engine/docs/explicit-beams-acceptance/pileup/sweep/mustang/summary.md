# Explicit beams scaling: Rubble GPU vs Rigs of Rods

`1982-Ford-Mustang-GT-351W.truck`: 527 nodes, 2664 beams. Apple M3 Max (Metal). 3 trials × 360 timed frames (60 untimed first) at 120 fps: 17 steps of 0.5 ms per frame, a 8.33 ms frame budget; drive and impact at 27.78 m/s.

Bounds (ceiling 33.3 ms for 10 consecutive frames, max_frame_ms 33.3 predicted, deadline_s 780 of running time): engines stopped or not run from: pileup RoR from 256 vehicles; pileup Rubble from 1024 vehicles. All trials ran.

## pileup

ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].

| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.356 [0.344, 0.368] | 0.380 | 0.723 [0.717, 0.729] | 0.732 | 0.733 [0.725, 0.740] | 0.983 [0.959, 1.007] | 0.49× [0.48, 0.51] | RoR faster | 722.99 |
| 2 | 1.068 [1.066, 1.071] | 1.265 | 1.311 [1.296, 1.325] | 2.602 | 1.322 [1.307, 1.336] | 1.588 [1.582, 1.593] | 0.82× [0.81, 0.82] | RoR faster | 655.34 |
| 4 | 1.155 [1.053, 1.257] | 1.502 | 1.696 [1.688, 1.703] | 2.628 | 1.707 [1.699, 1.715] | 1.975 [1.970, 1.980] | 0.68× [0.63, 0.74] | RoR faster | 423.92 |
| 8 | 1.592 [1.533, 1.651] | 1.847 | 2.478 [2.447, 2.509] | 2.626 | 2.489 [2.459, 2.520] | 2.759 [2.734, 2.783] | 0.64× [0.62, 0.67] | RoR faster | 309.76 |
| 16 | 2.122 [2.117, 2.127] | 2.537 | 2.525 [2.482, 2.567] | 2.723 | 2.536 [2.494, 2.578] | 2.829 [2.787, 2.871] | 0.84× [0.82, 0.86] | RoR faster | 157.80 |
| 32 | 3.339 [3.197, 3.482] | 3.693 | 3.485 [3.371, 3.598] | 4.895 | 3.497 [3.383, 3.610] | 3.760 [3.700, 3.819] | 0.96× [0.92, 1.00] | inconclusive | 108.90 |
| 64 | 6.221 [6.128, 6.314] | 7.109 | 3.480 [3.446, 3.513] | 4.119 | 3.491 [3.458, 3.524] | 3.792 [3.778, 3.807] | 1.79× [1.77, 1.81] | Rubble faster | 54.37 |
| 128 | 13.877 [13.330, 14.423] | 16.234 | 5.840 [5.791, 5.890] | 7.082 | 5.852 [5.804, 5.900] | 6.154 [6.127, 6.182] | 2.38× [2.27, 2.48] | Rubble faster | 45.63 |
| 256 | — | — | 10.442 [10.406, 10.479] | 12.000 | 10.455 [10.418, 10.491] | 10.781 [10.723, 10.839] | — | — | 40.79 |
| 512 | — | — | 19.050 [18.965, 19.135] | 21.121 | 19.064 [18.982, 19.146] | 19.374 [19.324, 19.425] | — | — | 37.21 |

Peak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the 8.33 ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.

| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.396 | 0.422 | 0.0 % | 0.737 | 0.743 | 0.0 % | 1.144 | 1.206 | 0.57× |
| 2 | 1.908 | 2.067 | 0.0 % | 2.734 | 2.889 | 0.0 % | 3.035 | 3.183 | 0.72× |
| 4 | 1.904 | 2.116 | 0.0 % | 2.774 | 2.960 | 0.0 % | 3.053 | 3.280 | 0.71× |
| 8 | 2.253 | 2.621 | 0.0 % | 2.744 | 2.910 | 0.0 % | 3.037 | 3.236 | 0.90× |
| 16 | 2.778 | 2.974 | 0.0 % | 2.840 | 2.919 | 0.0 % | 3.207 | 9.863 | 1.02× |
| 32 | 3.950 | 4.271 | 0.0 % | 5.019 | 5.218 | 0.0 % | 5.360 | 9.627 | 0.82× |
| 64 | 7.352 | 8.516 | 0.1 % | 4.205 | 4.521 | 0.0 % | 4.564 | 5.637 | 1.88× |
| 128 | 16.646 | 19.643 | 100.0 % | 7.512 | 7.760 | 0.0 % | 7.836 | 10.408 | 2.53× |
| 256 | — | — | — | 12.365 | 12.947 | 86.4 % | 12.889 | 13.379 | — |
| 512 | — | — | — | 21.593 | 22.576 | 100.0 % | 22.055 | 23.265 | — |

Real-time capacity (largest count with p99 ≤ 8.33 ms and every frame ≤ 33.3 ms, all smaller counts too): Rubble GPU time 128, Rubble blocking wall 128, Rigs of Rods 64 (of counts up to 512).


Scaling (least squares, time = fixed + per vehicle × N): Rubble GPU 1.611 ms + 34.13 µs × N; Rigs of Rods 0.492 ms + 101.14 µs × N.

Vehicles in a 2 ms physics budget (from the fit): Rubble 11, Rigs of Rods 14.  
Vehicles in a 4 ms physics budget (from the fit): Rubble 69, Rigs of Rods 34.  
Vehicles in a 8.33 ms physics budget (from the fit): Rubble 196, Rigs of Rods 77.  

Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.

| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 31.92 | 71.72 | 0 / 0 | 0 / 0 | 73.533 / 73.560 | -0.41 / -0.42 |
| 2 | 0 | 248.69 | 819.96 | 0 / 0 | 611 / 571 | 57.737 / 57.640 | 25.72 / 19.75 |
| 2 | 1 | 242.53 | 748.88 | 0 / 0 | 609 / 549 | -57.760 / -57.648 | 26.21 / 22.91 |
| 2 | 0 | 248.69 | 819.96 | 0 / 0 | 611 / 571 | 57.737 / 57.640 | 25.72 / 19.75 |
| 2 | 1 | 242.53 | 748.88 | 0 / 0 | 609 / 549 | -57.760 / -57.648 | 26.21 / 22.91 |
| 4 | 0 | 138.30 | 662.55 | 0 / 0 | 611 / 562 | 57.737 / 57.748 | 25.72 / 27.68 |
| 4 | 1 | 177.87 | 569.36 | 0 / 0 | 609 / 579 | -57.760 / -57.691 | 26.21 / 28.63 |
| 4 | 2 | 365.76 | 573.34 | 0 / 0 | 241 / 233 | 43.762 / 43.730 | 2.94 / -1.20 |
| 4 | 3 | 455.48 | 1025.37 | 0 / 0 | 206 / 208 | 8.222 / 8.195 | -36.61 / -112.07 |
| 8 | 0 | 242.13 | 831.53 | 0 / 0 | 611 / 572 | 57.737 / 57.914 | 25.72 / 24.74 |
| 8 | 1 | 278.00 | 713.06 | 0 / 0 | 609 / 567 | -57.760 / -57.533 | 26.21 / 24.23 |
| 8 | 6 | 193.40 | 644.07 | 0 / 0 | 664 / 627 | 15.020 / 14.880 | 51.34 / 51.53 |
| 8 | 7 | 236.54 | 697.19 | 0 / 0 | 621 / 634 | -15.017 / -15.211 | 47.33 / 46.45 |
| 16 | 0 | 136.11 | 694.47 | 0 / 0 | 611 / 574 | 57.737 / 57.695 | 25.72 / 24.52 |
| 16 | 1 | 122.99 | 686.05 | 0 / 0 | 609 / 567 | -57.760 / -57.705 | 26.21 / 27.22 |
| 16 | 14 | 1634.41 | 2072.35 | 0 / 0 | 469 / 459 | 38.197 / 36.572 | 10.52 / 7.44 |
| 16 | 15 | 1700.67 | 1980.23 | 0 / 0 | 316 / 309 | 18.580 / 16.905 | -32.61 / -3.06 |
| 32 | 0 | 214.34 | 723.56 | 0 / 0 | 611 / 562 | 57.737 / 57.682 | 25.72 / 23.76 |
| 32 | 1 | 206.07 | 667.44 | 0 / 0 | 609 / 571 | -57.760 / -57.665 | 26.21 / 26.68 |
| 32 | 30 | 208.42 | 679.54 | 0 / 0 | 806 / 779 | 21.788 / 21.947 | 65.59 / 52.21 |
| 32 | 31 | 235.65 | 682.33 | 0 / 0 | 811 / 817 | -21.796 / -21.830 | 66.83 / 60.70 |
| 64 | 0 | 171.42 | 673.15 | 0 / 0 | 611 / 571 | 57.737 / 57.669 | 25.72 / 26.25 |
| 64 | 1 | 97.96 | 374.21 | 0 / 0 | 609 / 571 | -57.760 / -57.796 | 26.21 / 25.45 |
| 64 | 62 | 1652.41 | 2233.11 | 0 / 0 | 416 / 418 | 40.187 / 39.302 | 0.05 / 6.20 |
| 64 | 63 | 1999.81 | 2458.95 | 0 / 0 | 302 / 296 | 14.729 / 13.884 | -3.87 / -10.61 |
| 128 | 0 | 221.35 | 778.16 | 0 / 0 | 611 / 592 | 57.737 / 57.848 | 25.72 / 22.91 |
| 128 | 1 | 141.67 | 446.41 | 0 / 0 | 609 / 584 | -57.760 / -57.680 | 26.21 / 24.85 |
| 128 | 126 | 157.36 | 640.11 | 0 / 0 | 666 / 649 | 31.683 / 31.624 | 58.29 / 50.31 |
| 128 | 127 | 160.69 | 639.91 | 0 / 0 | 623 / 646 | -31.705 / -31.731 | 33.97 / 36.51 |
## Acceptance

Command: `/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/acc/ebs /Users/glavin/Development/rubble-beamng-parity/inspiration/ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck ror=/Users/glavin/Development/rubble-beamng-parity/inspiration/ror-harness/bin/ror_ref scenarios=pileup counts=1,2,4,8,16,32,64,128,256,512,1024 rubble_counts=2048,4096,8192 fps=120 frames=360 speed=27.78 drag=0 trials=3 nudges=0 quiet_wait=1800 max_load=8 ceiling_ms=33.3 ceiling_frames=10 deadline_s=780 out=/private/tmp/claude-501/-Users-glavin-Development-rubble/04aea2e9-7df2-4b66-83b5-d09006713df9/scratchpad/sweep/mustang`

| result | criterion |
|---|---|
| PASS | 55 runs started on a quiet machine (load < 8, GPU < 10 %): 0 did not |
| FAIL | pileup N=1: speed-up 0.49x, 95% CI [0.48, 0.51] excludes 1 (RoR 0.356 ms, Rubble GPU 0.723 ms) |
| FAIL | pileup N=1: Rubble pipelined wall 0.733 ms < RoR mean 0.356 ms |
| FAIL | pileup N=1: Rubble p99 0.737 < RoR 0.396 ms, max 0.743 < 0.422 ms |
| FAIL | pileup N=2: speed-up 0.82x, 95% CI [0.81, 0.82] excludes 1 (RoR 1.068 ms, Rubble GPU 1.311 ms) |
| FAIL | pileup N=2: Rubble pipelined wall 1.322 ms < RoR mean 1.068 ms |
| FAIL | pileup N=2: Rubble p99 2.734 < RoR 1.908 ms, max 2.889 < 2.067 ms |
| FAIL | pileup N=4: speed-up 0.68x, 95% CI [0.63, 0.74] excludes 1 (RoR 1.155 ms, Rubble GPU 1.696 ms) |
| FAIL | pileup N=4: Rubble pipelined wall 1.707 ms < RoR mean 1.155 ms |
| FAIL | pileup N=4: Rubble p99 2.774 < RoR 1.904 ms, max 2.960 < 2.116 ms |
| FAIL | pileup N=8: speed-up 0.64x, 95% CI [0.62, 0.67] excludes 1 (RoR 1.592 ms, Rubble GPU 2.478 ms) |
| FAIL | pileup N=8: Rubble pipelined wall 2.489 ms < RoR mean 1.592 ms |
| FAIL | pileup N=8: Rubble p99 2.744 < RoR 2.253 ms, max 2.910 < 2.621 ms |
| FAIL | pileup N=16: speed-up 0.84x, 95% CI [0.82, 0.86] excludes 1 (RoR 2.122 ms, Rubble GPU 2.525 ms) |
| FAIL | pileup N=16: Rubble pipelined wall 2.536 ms < RoR mean 2.122 ms |
| FAIL | pileup N=16: Rubble p99 2.840 < RoR 2.778 ms, max 2.919 < 2.974 ms |
| FAIL | pileup N=32: speed-up 0.96x, 95% CI [0.92, 1.00] excludes 1 (RoR 3.339 ms, Rubble GPU 3.485 ms) |
| FAIL | pileup N=32: Rubble pipelined wall 3.497 ms < RoR mean 3.339 ms |
| FAIL | pileup N=32: Rubble p99 5.019 < RoR 3.950 ms, max 5.218 < 4.271 ms |
| PASS | pileup N=64: speed-up 1.79x, 95% CI [1.77, 1.81] excludes 1 (RoR 6.221 ms, Rubble GPU 3.480 ms) |
| PASS | pileup N=64: Rubble pipelined wall 3.491 ms < RoR mean 6.221 ms |
| PASS | pileup N=64: Rubble p99 4.205 < RoR 7.352 ms, max 4.521 < 8.516 ms |
| PASS | pileup N=128: speed-up 2.38x, 95% CI [2.27, 2.48] excludes 1 (RoR 13.877 ms, Rubble GPU 5.840 ms) |
| PASS | pileup N=128: Rubble pipelined wall 5.852 ms < RoR mean 13.877 ms |
| PASS | pileup N=128: Rubble p99 7.512 < RoR 16.646 ms, max 7.760 < 19.643 ms |

**NOT MET** (18 failing)
