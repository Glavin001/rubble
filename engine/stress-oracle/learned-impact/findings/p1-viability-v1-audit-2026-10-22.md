# Impact-outcome dataset audit

Events: **982**; failures: **0**; integrity problems: **0**.

Oracle configuration: `a12_realtime` (provisional: True; stress_rubble::scenes::realtime (A12); P0 step 1 (p0-feasibility-2026-10-20.md)).

| Check | Value | Rule |
|---|---|---|
| Events still breaking in their last 0.5 s | 0 (0.0%) | must be 0 |
| Truncated at the cap | 7 (0.7%) | excluded from outcome metrics |
| Ensembles (complete / groups, members) | 53 / 54, 430 | all members recorded |
| Events per split | test 120, train 552, val 310 | layout groups held out whole |
| Storage | 7.87 GB; 0 thinned training events keep 99674 of 99674 frames | dropped frames regenerate bit-identically (`--regenerate`) |

## Event durations (simulated s)

| Range | Events |
|---|---|
| [0.25, 0.5) | 8 |
| [0.5, 0.75) | 179 |
| [0.75, 1) | 65 |
| [1, 1.5) | 261 |
| [1.5, 2) | 246 |
| [2, 3) | 135 |
| [3, 5) | 62 |
| [5, 10) | 25 |
| [10, inf) | 1 |

## Frames per phase

| Phase | Frames | Share |
|---|---|---|
| pre_contact | 724 | 0.7% |
| contact | 6247 | 6.3% |
| secondary | 64629 | 64.8% |
| separation | 16331 | 16.4% |
| settling | 11743 | 11.8% |

## Cost per event by family (wall-clock s, one thread per event)

| Family | Events | Chunks (median) | Sim s (median) | Mean | Median | p90 | Max | Oracle / QS / punch / prior (mean) | No-break share | Regimes | MB/event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchored_wall/t1 | 261 | 120 | 1.28 | 49.0 | 35.5 | 103.8 | 346.5 | 23.7 / 14.7 / 0.3 / 2.27 | 22% | arrest 29, deflect 4, perforate 122, rebound 106 | 6.9 |
| anchored_wall/t2 | 222 | 125 | 1.47 | 56.8 | 40.2 | 132.2 | 325.7 | 27.5 / 19.5 / 0.9 / 1.24 | 0% | arrest 114, deflect 12, perforate 399, rebound 179 | 6.1 |
| bl_anchored_base/t1 | 10 | 120 | 1.98 | 129.4 | 127.5 | 170.5 | 205.1 | 64.7 / 46.4 / 0.0 / 3.34 | 0% | perforate 4, rebound 6 | 9.0 |
| bl_anchored_sides/t1 | 80 | 240 | 0.92 | 148.5 | 141.2 | 207.1 | 338.6 | 60.9 / 58.3 / 0.6 / 15.63 | 0% | arrest 38, perforate 24, rebound 18 | 8.5 |
| bl_coarse_heavy/t1 | 10 | 64 | 0.90 | 22.3 | 22.6 | 35.7 | 38.2 | 10.1 / 8.0 / 0.2 / 0.22 | 0% | arrest 1, perforate 9 | 2.2 |
| bl_ground_thick/t1 | 80 | 200 | 1.85 | 153.4 | 155.6 | 193.7 | 238.9 | 86.7 / 41.2 / 0.2 / 6.33 | 0% | arrest 58, deflect 1, perforate 12, rebound 9 | 16.0 |
| bl_ground_thin/t1 | 10 | 100 | 1.13 | 72.8 | 77.8 | 86.0 | 89.1 | 40.2 / 23.4 / 0.0 / 2.20 | 0% | arrest 1, deflect 1, perforate 4, rebound 4 | 4.8 |
| bl_standing/t1 | 10 | 80 | 1.71 | 100.0 | 77.5 | 168.9 | 278.1 | 49.2 / 38.6 / 0.0 / 1.39 | 0% | arrest 1, perforate 4, rebound 5 | 6.5 |
| standing_wall/t1 | 158 | 132 | 1.67 | 85.1 | 75.5 | 168.2 | 327.5 | 43.5 / 22.7 / 0.3 / 5.78 | 23% | arrest 47, deflect 9, perforate 57, rebound 45 | 9.5 |
| standing_wall/t2 | 141 | 128 | 1.45 | 88.2 | 59.1 | 145.5 | 1414.0 | 44.2 / 28.0 / 1.1 / 2.15 | 1% | arrest 55, deflect 9, perforate 335, rebound 29 | 7.4 |
