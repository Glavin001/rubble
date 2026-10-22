# /5 viability run: early-warning diagnostics (2026-10-22, planning session request)

**Trigger.** At step 6,000 of 12,000, the first `/5` viability run (W64/12, full-state and cluster
heads, P1's final calls-v5 snapshot: 850 events, 27,464 training and 20,630 validation calls)
validated at:
- bond F1 0.47 (P 0.54, R 0.42);
- partition 4.7× bar, sizes 4.0×, ram 1.5×;
- training-sample loss about 16, flat on validation (19.4 → 19.1 from step 4k to 6k).

## 1. Pipeline sanity: sound

The oracle's own `/5` targets, pushed through the decode and the metrics on the same
validation ensembles, give:
- bond F1 0.98, selection score 0.004;
- skill/bar: partition 0.35, sizes 0.28, ram 0.16.

Per family (pass rate oracle-as-model vs held-out member; mean skill/bar likewise):

| family (aligned ens.) | ram | partition | sizes |
|---|---|---|---|
| anchored_wall (5) | 1.00 vs 0.95; 0.20 vs 0.24 | 1.00 vs 0.93; 0.49 vs 0.65 | 1.00 vs 0.93; 0.38 vs 0.56 |
| bl_ground_thick (7) | 1.00 vs 0.96; 0.20 vs 0.27 | 1.00 vs 0.91; 0.39 vs 0.51 | 1.00 vs 0.91; 0.32 vs 0.47 |
| standing_wall (9) | 1.00 vs 1.00; 0.09 vs 0.11 | 1.00 vs 1.00; 0.16 vs 0.18 | 1.00 vs 0.97; 0.14 vs 0.18 |

So the decode and the gates are not the problem.

## 2. Apples to apples: the validation mix is harder, not a regression

The old `/3` bulk-w32 checkpoint (a shim drops the 12 deformation columns; rev 7's
principal-moment inertia features and E_in differ slightly), on the same `/5` validation set:
- bond F1 0.49, partition 4.9×, sizes 4.0×, ram 1.35×, essentially the new run's numbers;
- per family:
  - standing_wall: partition 0.34×, sizes 0.57×, ram 0.62× (passes);
  - anchored_wall: 2.86×, 1.07×, 0.93×;
  - bl_ground_thick (ballistic): 11.5×, 10.4×, 2.5×.

The family-averaged headline is dominated by the ballistic validation sweep, which the old
model never trained on and the new one has 4 single-run training families for.

## 3. Gradient balance: broken, and fixed

Shares of the summed per-term gradient norm:
- **In the run's log:** the 16 full-state bond-field terms took 46%, def_motion 25%, and
  margin, break and demand about 0%.
- **At the step-6k checkpoint, on another training batch set:** impactor 40%, energy 23%,
  motion and pose 29%, the decision terms under 2%.

Normalising every term by its zero-predictor baseline gives terms with very different
gradient scales: fields that barely change, with baselines of 1e-4 to 1e-3, are amplified
about 1/baseline. Which term dominates shifts with the batch, so fixed weights cannot fix it.

Fix (25f1afa9):
- **GroupBalancer (`--balance groups`, now the default).** The five term groups get target
  gradient shares: decision 30%, motion 30%, energy 10%, full-state bonds 15%, other full
  state 15%. Per-group multipliers are set from gradient norms measured every 100 steps
  (5 extra backward passes), smoothed geometrically and kept at geometric mean 1.
- **Normaliser floor.** Full-state normalisers are floored at a tenth of their group's median
  baseline.
- **Weights.** Group-total default weights.
- **Check, 200 steps on 33 real `/5` events:** decision 3% → 43% and motion 69% → 16%, moving
  to the targets. A unit test converges every group to within 5 points.

## 4. Training length

Validation loss was flat from step 4k (19.4 → 19.1) while training fell (18.0 → 15.8). So it
was not just undertrained; the balance fix is the lever.

## Action

- Both `/5` runs were restarted with the balancer (unbalanced runs kept as
  `runs/v5/*-unbalanced`): the viability queue (W64 seeds 0–2, W128, scaling) and the matched
  run with break-time and hidden heads.
- The validation table is reported per family as well as averaged, because a family mix like
  this one is dominated by its hardest member.
