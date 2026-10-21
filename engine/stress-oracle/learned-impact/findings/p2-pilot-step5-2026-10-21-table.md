# Pilot learnability and data scaling (P0 step 5)

Ensemble-calibrated gates: True. Gates (≤): `skill_over_bar_ram` 1.0, `skill_over_bar_partition` 1.0, `skill_over_bar_sizes` 1.0

## Variant A

| metric | N=64 | N=131 | N=208 | fit a·N^-b + c | events to the gate |
|---|---|---|---|---|---|
| `skill_over_bar_ram` | 1.162 [1.014, 1.446] ×3 | 0.945 [0.759, 1.230] ×3 | 1.230 [1.139, 1.325] ×3 | a=1.01 b=-0.0189 c=0 | ∞ (no improvement with data (b ≤ 0)) |
| `skill_over_bar_partition` | 1.002 [0.973, 1.022] ×3 | 0.984 [0.925, 1.014] ×3 | 0.951 [0.899, 1.016] ×3 | a=1.2 b=0.0431 c=0 | 73 |
| `skill_over_bar_sizes` | 2.587 [2.136, 2.945] ×3 | 2.270 [1.544, 2.682] ×3 | 2.237 [1.747, 2.799] ×3 | a=7.91e+03 b=2.41 c=2.22 | ∞ (asymptote 2.22 ≥ bar 1.0: more data alone will not pass) |
| `score` | 1.812 [1.759, 1.892] ×3 | 1.612 [1.299, 1.842] ×3 | 1.697 [1.476, 1.947] ×3 |  |  |
| `bond_f1` | 0.217 [0.000, 0.371] ×3 | 0.402 [0.179, 0.648] ×3 | 0.431 [0.196, 0.593] ×3 |  |  |
| `bond_recall` | 0.161 [0.000, 0.318] ×3 | 0.337 [0.133, 0.624] ×3 | 0.342 [0.110, 0.533] ×3 |  |  |
| `false_break_rate` | 0.001 [0.000, 0.003] ×3 | 0.001 [0.000, 0.004] ×3 | 0.000 [0.000, 0.001] ×3 |  |  |
| `regime_brier` | nan [nan, nan] ×3 | nan [nan, nan] ×3 | nan [nan, nan] ×3 |  |  |

Three-way (per run: family → superiority / non-inferiority):

- N/seed 76/s0: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.0% of 3064 calls not worse)
- N/seed 138/s0: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.7% of 3064 calls not worse)
- N/seed 208/s0: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.6% of 3064 calls not worse)
- N/seed 55/s1: free_wall: superior to QS and punch=False, non-inferior on QS-correct calls=True (99.8% of 3064 calls not worse)
- N/seed 117/s1: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (97.9% of 3064 calls not worse)
- N/seed 208/s1: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.7% of 3064 calls not worse)
- N/seed 62/s2: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.3% of 3064 calls not worse)
- N/seed 138/s2: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.2% of 3064 calls not worse)
- N/seed 208/s2: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.0% of 3064 calls not worse)

## Variant B

| metric | N=64 | N=131 | N=208 | fit a·N^-b + c | events to the gate |
|---|---|---|---|---|---|
| `skill_over_bar_ram` | 0.905 [0.553, 1.089] ×3 | 0.578 [0.427, 0.664] ×3 | 0.514 [0.389, 0.665] ×3 | a=1.22e+03 b=1.91 c=0.468 | 58 |
| `skill_over_bar_partition` | 1.004 [0.993, 1.022] ×3 | 1.023 [1.002, 1.042] ×3 | 0.982 [0.960, 1.027] ×3 | a=1.08 b=0.0146 c=0 | 148 |
| `skill_over_bar_sizes` | 2.265 [1.911, 2.663] ×3 | 2.690 [2.450, 2.899] ×3 | 2.351 [1.863, 2.794] ×3 | a=1.92 b=-0.0493 c=0 | ∞ (no improvement with data (b ≤ 0)) |
| `score` | 1.597 [1.566, 1.639] ×3 | 1.600 [1.531, 1.644] ×3 | 1.439 [1.331, 1.596] ×3 |  |  |
| `bond_f1` | 0.349 [0.091, 0.544] ×3 | 0.213 [0.051, 0.353] ×3 | 0.397 [0.286, 0.455] ×3 |  |  |
| `bond_recall` | 0.489 [0.081, 0.926] ×3 | 0.144 [0.030, 0.243] ×3 | 0.319 [0.182, 0.441] ×3 |  |  |
| `false_break_rate` | 0.005 [0.002, 0.008] ×3 | 0.001 [0.001, 0.002] ×3 | 0.002 [0.001, 0.004] ×3 |  |  |
| `regime_brier` | nan [nan, nan] ×3 | nan [nan, nan] ×3 | nan [nan, nan] ×3 |  |  |

Three-way (per run: family → superiority / non-inferiority):

- N/seed 76/s0: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (93.0% of 3064 calls not worse)
- N/seed 138/s0: free_wall: superior to QS and punch=False, non-inferior on QS-correct calls=True (99.2% of 3064 calls not worse)
- N/seed 208/s0: free_wall: superior to QS and punch=False, non-inferior on QS-correct calls=True (99.9% of 3064 calls not worse)
- N/seed 55/s1: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (90.5% of 3064 calls not worse)
- N/seed 117/s1: free_wall: superior to QS and punch=False, non-inferior on QS-correct calls=True (98.6% of 3064 calls not worse)
- N/seed 208/s1: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (96.7% of 3064 calls not worse)
- N/seed 62/s2: free_wall: superior to QS and punch=False, non-inferior on QS-correct calls=True (97.7% of 3064 calls not worse)
- N/seed 138/s2: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.4% of 3064 calls not worse)
- N/seed 208/s2: free_wall: superior to QS and punch=True, non-inferior on QS-correct calls=True (99.9% of 3064 calls not worse)
