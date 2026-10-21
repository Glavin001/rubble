# Impact-outcome dataset audit

Events: **640**; failures: **0**; integrity problems: **0**.

Oracle configuration: `a12_realtime` (provisional: True; stress_rubble::scenes::realtime (A12); P0 step 1 (p0-feasibility-2026-10-20.md)).

| Check | Value | Rule |
|---|---|---|
| Events still breaking in their last 0.5 s | 0 (0.0%) | must be 0 |
| Truncated at the cap | 9 (1.4%) | excluded from outcome metrics |
| Ensembles (complete / groups, members) | 12 / 12, 96 | all members recorded |
| Events per split | None 242, sealed 2, test 50, train 292, val 54 | layout groups held out whole |

## Event durations (simulated s)

| Range | Events |
|---|---|
| [0.25, 0.5) | 9 |
| [0.5, 0.75) | 367 |
| [0.75, 1) | 49 |
| [1, 1.5) | 93 |
| [1.5, 2) | 57 |
| [2, 3) | 31 |
| [3, 5) | 16 |
| [5, 10) | 9 |
| [10, inf) | 9 |

## Frames per phase

| Phase | Frames | Share |
|---|---|---|
| pre_contact | 657 | 1.5% |
| contact | 1753 | 4.0% |
| secondary | 9042 | 20.4% |
| separation | 6438 | 14.5% |
| settling | 26374 | 59.6% |

## Cost per event by family (wall-clock s, one thread per event)

| Family | Events | Chunks (median) | Sim s (median) | Mean | Median | p90 | Max | Oracle / QS / punch / prior (mean) | No-break share | Regimes | MB/event |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anchored_wall/t1 | 176 | 132 | 1.26 | 47.9 | 13.1 | 66.3 | 1738.0 | 16.9 / 27.0 / 0.3 / 3.63 | 16% | arrest 22, deflect 1, perforate 103, rebound 50 | 5.0 |
| anchored_wall/t2 | 40 | 118 | 1.42 | 21.3 | 18.6 | 45.4 | 64.2 | 11.7 / 8.5 / 0.6 / 0.45 | 0% | arrest 15, deflect 1, perforate 76, rebound 30 | 3.1 |
| free_wall/t1 | 380 | 112 | 0.55 | 25.2 | 6.2 | 49.1 | 529.6 | 10.1 / 7.6 / 0.4 / 6.98 | 3% | arrest 13, deflect 22, perforate 326, rebound 19 | 4.3 |
| free_wall/t2 | 40 | 120 | 0.60 | 8.8 | 7.0 | 15.2 | 42.4 | 5.4 / 2.6 / 0.5 / 0.28 | 0% | arrest 3, deflect 7, perforate 100, rebound 7 | 2.2 |
| m5_reference/t1 | 4 | 330 | 0.52 | 23.7 | 23.6 | 29.7 | 30.0 | 12.4 / 10.1 / 0.7 / 0.41 | 0% | perforate 4 | 4.3 |
