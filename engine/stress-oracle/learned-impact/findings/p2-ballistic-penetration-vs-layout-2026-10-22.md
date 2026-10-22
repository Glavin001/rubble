# P2: is the ballistic failure penetration or layout? (2026-10-22)

Planning asked whether v7 W64 seed 0's ballistic failure (`bl_ground_thick` 0/7 on ram, partition
and sizes) means the model cannot learn penetration or cannot extrapolate to a new wall layout.

**Answer: layout.**
- The model fits the perforation transition and the exit speed in every
  ballistic family it was trained on, all of them one chunk thick.
- It fails only the two-chunk-thick ballistic layouts: `bl_ground_thick` (val) and
  `bl_anchored_sides` (test). On those it decelerates the impactor like a thin wall and
  predicts perforation too early.
- The remedy is ballistic coverage of thick walls in the data. Caveat: the training-family
  curves are a fit, not held-out generalisation; the in-distribution held-out test is
  below, as a proposal.

## Layout ranges (P1's recipes, viability v1)

| family | split | layout | chunks deep | chunk (m) | thickness (m) | supports | impactor | v0 (m/s) |
|---|---|---|---:|---:|---:|---|---|---|
| bl_ground_thin | train | 10×1×10 | 1 | 0.10 | 0.10 | free (ground) | sphere 100 kg | 1.25–5 |
| bl_anchored_base | train | 12×1×10 | 1 | 0.15 | 0.15 | base | sphere 300 kg | 0.9–3.6 |
| bl_standing | train | 10×1×8 | 1 | 0.15 | 0.15 | free | sphere 500 kg | 1.1–4.5 |
| bl_coarse_heavy | train | 8×1×8 | 1 | 0.20 | 0.20 | base | box 1000 kg | 3.75–15 |
| **bl_ground_thick** | **val** | 10×**2**×10 | **2** | 0.10 | 0.20 | free | box 200 kg | 2–11 |
| bl_anchored_sides | test | 12×**2**×10 | **2** | 0.10 | 0.20 | base + sides | box 150 kg | 4.5–18 |
| standing_wall | train | various | 1 or 2 (201 of 406 two deep) | 0.10/0.15 | 0.10–0.30 | free | box/sphere 50–1500 kg | 0.1–60 |
| anchored_wall | train | various | 1 or 2 (197 of 393) | 0.10/0.15 | 0.10–0.30 | base / base+sides | box/sphere 50–1500 kg | 0.1–60 |

- **Every parameter of `bl_ground_thick` lies inside the training ranges.**
  - Thickness 0.2 m two chunks deep, 0.1 m chunks, free on the ground, a 200 kg box at 2–11 m/s:
    `standing_wall` has 176 training runs that are two deep with 0.1 m chunks, free, at
    0.1–60 m/s and 50–1500 kg.
  - Its overall mass and size are interpolations too.
- **What training lacks is its combination near the transition.** No ballistic family is two
  chunks deep. The wall families sample speed log-uniformly over 0.1–60 m/s, so very few of
  their two-deep runs land near a perforation transition (3–7.5 m/s here).

## Per-speed curves, v7 W64 seed 0 (teacher-forced, the aligned impact call of each ensemble)

Perforation probability against the ensemble's frequency, and the impactor's frame-end speed
along its incoming direction against the members (`scripts/ballistic_curve.py`; raw rows in
`runs/v7/w64-s0/ballistic-curve.json`). Families with fewer rows have ensembles that
were incomplete or not aligned at the impact call.

### Training families (fit; trained on these ensembles)

**bl_ground_thin**

| v0 (m/s) | perforation freq, oracle | p_perforate, model | frame-end v, oracle mean [min, max] | model |
|---:|---:|---:|---|---:|
| 1.80 | 0.00 | 0.01 | 0.96 [0.96, 0.96] | 0.97 |
| 2.12 | 0.00 | 0.03 | 1.20 [1.20, 1.20] | 1.19 |
| 2.51 | 0.25 | 0.26 | 1.47 [1.47, 1.47] | 1.46 |
| 2.96 | 1.00 | 0.93 | 1.77 [1.76, 1.77] | 1.78 |

**bl_anchored_base**

| v0 (m/s) | perforation freq, oracle | p_perforate, model | frame-end v, oracle mean [min, max] | model |
|---:|---:|---:|---|---:|
| 1.30 | 0.00 | 0.02 | 0.24 [0.24, 0.24] | 0.24 |
| 1.63 | 0.00 | 0.02 | 0.46 [0.46, 0.46] | 0.45 |
| 2.04 | 0.00 | 0.14 | 0.74 [0.74, 0.74] | 0.75 |
| 2.55 | 1.00 | 0.96 | 1.20 [1.20, 1.20] | 1.19 |
| 3.20 | 1.00 | 0.99 | 1.74 [1.73, 1.75] | 1.75 |

**bl_standing**

| v0 (m/s) | perforation freq, oracle | p_perforate, model | frame-end v, oracle mean [min, max] | model |
|---:|---:|---:|---|---:|
| 1.60 | 0.00 | 0.02 | 1.12 [1.12, 1.13] | 1.12 |
| 1.87 | 0.00 | 0.03 | 1.34 [1.34, 1.34] | 1.35 |
| 2.19 | 0.00 | 0.12 | 1.62 [1.62, 1.63] | 1.62 |
| 2.56 | 0.88 | 0.73 | 1.94 [1.94, 1.95] | 1.94 |
| 3.00 | 1.00 | 0.99 | 2.32 [2.32, 2.33] | 2.32 |

**bl_coarse_heavy**

| v0 (m/s) | perforation freq, oracle | p_perforate, model | frame-end v, oracle mean [min, max] | model |
|---:|---:|---:|---|---:|
| 9.99 | 0.75 | 0.81 | 8.74 [8.58, 8.91] | 8.78 |

### Held-out thick layouts

**bl_ground_thick (val)**

| v0 (m/s) | perforation freq, oracle | p_perforate, model | frame-end v, oracle mean [min, max] | model |
|---:|---:|---:|---|---:|
| 2.00 | 0.00 | 0.01 | 0.51 [0.47, 0.52] | 1.18 |
| 2.42 | 0.00 | 0.03 | 0.67 [0.62, 0.68] | 1.48 |
| 2.92 | 0.00 | 0.09 | 0.89 [0.83, 0.90] | 1.80 |
| 3.53 | 0.00 | 0.32 | 1.16 [1.09, 1.19] | 2.31 |
| 4.27 | 0.00 | 0.74 | 1.51 [1.43, 1.55] | 3.16 |
| 6.23 | 0.00 | 0.79 | 4.47 [4.31, 4.65] | 3.55 |
| 7.53 | 0.38 | 0.92 | 4.97 [4.52, 5.36] | 4.40 |

**bl_anchored_sides (test, shown for diagnosis only; not a selection signal)**

| v0 (m/s) | perforation freq, oracle | p_perforate, model | frame-end v, oracle mean [min, max] | model |
|---:|---:|---:|---|---:|
| 6.12 | 0.00 | 0.26 | 0.29 [-0.06, 0.85] | 2.82 |
| 7.14 | 0.00 | 0.38 | 1.08 [0.13, 2.97] | 3.22 |
| 8.33 | 0.00 | 0.47 | 0.94 [0.12, 2.89] | 3.78 |
| 18.00 | 1.00 | 0.95 | 11.13 [9.72, 13.85] | 10.67 |

### Reading

- **Training families:** the transition is where the oracle puts it, and the exit speed
  matches to 0.01–0.02 m/s. The model can represent the penetration decision and its
  kinematics.
- **Thick layouts, deceleration:** at 2–4.3 m/s the impactor keeps about twice the oracle's
  frame-end speed (1.18 vs 0.51 at 2 m/s), the thin walls' response.
- **Thick layouts, perforation:** p_perforate passes 0.3 from 3.5 m/s, while the oracle first
  perforates at 7.5 m/s (0.38). This is false perforation, the failure the gate counts.
- `bl_anchored_sides` shows the same pattern.
- **Remaining uncertainty:** whether it would also fail an unseen speed of a seen one-deep
  layout. Answering that needs the held-out test below.

## The in-distribution held-out test (proposal)

All of P1's training ballistic ensembles (4 families × 5 transition speeds × 8) were in v7's
training set, so no held-out in-distribution ballistic data exist for seeds 0 and 1.

- **Data, at no oracle cost:** hold out 2 of the 5 speeds per training family (the 2nd and
  4th) as validation, with their own split group, since they share the layout with the
  training speeds by design.
- **Run:** a dedicated W64 run on that split, not a change to the v7 seed set (which would
  confound the 3 seeds).
- **Readout:** if it passes there and fails only on thick layouts, the data remedy is
  2-deep ballistic training ensembles near the transition.
