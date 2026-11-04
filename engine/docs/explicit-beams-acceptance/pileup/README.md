# Vehicle–vehicle contact: accuracy, no-contact speed, pileup sweep (2026-11-03)

Branch `feat/explicit-vv-contact`. Spec: [../../explicit-beams-vehicle-contact-spec.md](../../explicit-beams-vehicle-contact-spec.md).
Reference: `ror_ref` `scale` mode running Rigs of Rods' inter-actor collision through its
own functions (`vv=1`, the default). Apple M3 Max (Metal). All runs: 120 fps (17 steps
of 0.5 ms), 60 warm + 360 timed frames, `speed=27.78`, `drag=0`. Scripts in
[scripts/](scripts/) (`$SCRATCH` was the session's scratch directory).

Every run used `max_load=8` instead of the default 3: a desktop app outside this work
held the 1-minute load average above 3 throughout. The GPU-busy limit (10 %) was kept,
and every run started under both limits. The before and after speed runs used the same
setting, back to back.

## Accuracy (two-car ensembles, `nudges=24`, RoR `threads=1`)

[ACCURACY-RULE.md](../ACCURACY-RULE.md), per scenario on one crash group (node RMS over
both cars; broken and bent counts summed). The moving car is spawned 0.1 mm/s × k
faster in run k. Head-on: car 2 turned 180° and driving at the same speed; T-bone: car
2 turned 90° and standing across car 1's lane. Car 1 drives at 0.85 × 27.78 m/s; the
hit lands about 2.4 s into the 3.5 s run.

| car | scenario | M_XR (mm) | M_RR (mm) | ratio [95 %] | broken: Rubble / RoR range | bent: Rubble / RoR range | verdict |
|---|---|---|---|---|---|---|---|
| Viper | head-on | 1030.9 | 1236.8 | 0.83 [0.66, 1.32] | 0 / 0–0 | 337 / 294–330 | PASS |
| Viper | T-bone | 679.8 | 702.6 | 0.97 [0.83, 1.11] | 2 / 0–2 | 83 / 62–104 | PASS (fleet line: see below) |
| E36 | head-on | 760.4 | 745.3 | 1.02 [0.85, 1.16] | 281 / 232–356 | 1849 / 1730–1942 | PASS |
| E36 | T-bone | 465.9 | 534.5 | 0.87 [0.74, 1.17] | 24 / 24–42 | 1472 / 1422–1532 | PASS |
| Mustang | head-on | 214.0 | 211.7 | 1.01 [0.95, 1.10] | 0 / 0–0 | 2317 / 2198–2396 | PASS |
| Mustang | T-bone | 246.3 | 234.9 | 1.05 [0.85, 1.40] | 0 / 0–0 | 1152 / 1100–1202 | PASS |

Fleet line (rule 6, the single timing-run group): it passes in 5 of 6 cases. It fails for
the Viper T-bone, at 1152 mm against a 1054 mm limit. That line is one draw from a
chaotic case. 2 of Rubble's 24 ensemble runs exceed the limit, and so do 3 of RoR's 24
runs measured against the other RoR runs. Raw data: [accuracy/](accuracy/).

## No-contact fleets: before (eaa1ddcc) and after (vehicle contact on)

Scenario `mixed`, where vehicles never meet. Rubble GPU ms per frame, mean of 3 trials
[95 % CI]. Raw data: [no-contact-speed/](no-contact-speed/).

| car | N | before | after | change |
|---|---|---|---|---|
| Viper | 1 | 0.546 [0.545, 0.548] | 0.544 [0.540, 0.549] | 0 |
| Viper | 64 | 0.735 [0.731, 0.739] | 0.763 [0.740, 0.786] | +3.8 % |
| Viper | 1024 | 8.415 [8.405, 8.425] | 8.351 [8.323, 8.379] | −0.8 % |
| E36 | 1 | 0.700 [0.677, 0.724] | 0.706 [0.702, 0.710] | within noise |
| E36 | 64 | 1.598 [1.588, 1.607] | 1.631 [1.573, 1.689] | +2 %, CIs overlap |
| E36 | 1024 | 20.52 [20.49, 20.54] | 20.62 [20.46, 20.79] | +0.5 %, CIs overlap |
| Mustang | 1 | 0.726 [0.725, 0.727] | 0.725 [0.718, 0.733] | 0 |
| Mustang | 64 | 1.656 [1.642, 1.670] | 1.638 [1.632, 1.643] | −1 % |
| Mustang | 1024 | 24.78 [23.09, 26.46] | 23.26 [22.79, 23.73] | within noise |

N=1 runs the old kernel, unchanged. The cost that remains at N=64 is about 20–30 µs
per frame. It is the per-step check that each node stays inside the frame bounds the
contact plan assumed. That check is what makes skipping the contact work lossless,
since a node's speed is not bounded within a step. `set_vehicle_contact(false)` runs
the old frame code exactly.

## Pileup sweep (`scenarios=pileup`, 3 trials, ceiling 33.3 ms × 10 frames, deadline 780 s)

| car | Real-time capacity: Rubble GPU time | Rubble blocking wall | Rigs of Rods | scaling fit (Rubble / RoR) |
|---|---|---|---|---|
| Viper | 512 | 256 | 64 | 1.38 ms + 9.40 µs·N / −0.03 ms + 94.4 µs·N |
| E36 | 128 | 64 | 32 | 1.60 ms + 30.6 µs·N / 0.61 ms + 131.3 µs·N |
| Mustang | 128 | 128 | 64 | 1.61 ms + 34.1 µs·N / 0.49 ms + 101.1 µs·N |

Capacity is the largest count whose p99 frame is at most 8.33 ms and whose every frame
is at most 33.3 ms, with every smaller count passing as well. Rubble ran up to 2048
Vipers (20.4 ms) and up to 512 E36s or Mustangs (17–19 ms); the frame prediction and
ceiling stopped larger counts. RoR stopped at 512 Vipers, and at 256 E36s or Mustangs.
At small counts RoR is mostly faster (1–16 vehicles; Mustang to 32), because Rubble
steps a crashing pair in one workgroup, one member after the other. Rubble is faster
from 32 vehicles on (E36 from 16, Mustang from 64): Viper 2.2× at 64 and 6.7× at 256,
E36 3.2× at 128, Mustang 2.4× at 128. Full tables: [sweep/](sweep/).
