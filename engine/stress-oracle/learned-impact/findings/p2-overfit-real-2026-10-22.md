# Real-data overfit milestone (viability step 0), 2026-10-22

**Question.** Can the representation, architecture and decode memorise real oracle events? We
train and score on the *same* events, with no held-out data and no regularisation (no weight
decay, noise, pushforward or label smoothing), at the default W64/12. Anything that does not
fit is a bug to fix before viability training.

**Setup.**
- Tool: `python -m learned_impact.outcome.overfit_real`.
- Events: single members (`_e0`) of P1's bulk v1 evaluation ensembles, full sequences from
  impact through settling.
  - 1 event: an anchored tier-1 wall that the projectile perforates.
  - 10 events: free and anchored, tiers 1 and 2, with a perforation and an arrest.
  - 100 events: family-stratified.
- Calls `/3` read as `/4` with the hidden deformation unobserved (`LEARNED_IMPACT_ACCEPT_V3=1`):
  the `/4` and `/5` data were not built yet.
- Break decision: the deployed margin rule q̂ ≥ τ with **τ = 1 fixed** (no tuned τ).
- Velocity errors are compared with the ensemble spread: the RMS difference between the
  event's members' outcomes at the same call and body.

## Results

| run | calls | bonds TP / FP / FN | F1 | fragments identical | chunk v err ÷ spread (median / p90) | impactor v err ÷ v (max) | momentum residual (free) | KE above bound (max) |
|---|---|---|---|---|---|---|---|---|
| 1 event (final heads) | 54 | 53 / 0 / 0 | 1.000 | 1.000 | 0.012 / 0.11 | 0.0003 | — | -0.050 |
| 10 events, loss v1 (margin: survivors regression only) | 447 | 896 / 96 / 0 | 0.949 | 0.940 | 0.050 / 0.40 | 0.0019 | 3.0e-16 | -0.036 |
| 10 events, loss v2 (+ survivor hinge) | 739 | 896 / 125 / 0 | 0.935 | 0.958 | 0.163 / 0.55 | 0.0017 | 3.5e-16 | -0.008 |
| 10 events, final heads (monotone margin), 12k steps | 739 | 896 / 0 / 0 | 1.000 | 1.000 | 0.090 / 0.34 | 0.0009 | 3.1e-16 | -0.006 |
| 100 events, final heads, 20k steps (not saturated) | 4755 | 9327 / 249 / 432 | 0.965 | 0.956 | 0.036 / 1.21 | 0.0576 | 4.9e-16 | +0.011 |

**100 events are not saturated at 20k steps.**
- The training-sample loss fell 4.68 → 0.61 and F1 0.60 → 0.965. The last evaluations
  still improved (0.921, 0.952, 0.962, 0.965) as the cosine schedule ran out.
- So this is the schedule, not a plateau. A continuation (30k more steps, a cosine restart at
  lr 5e-4) is running: `runs/overfit-real/n100-cont`.
- The decoded KE exceeded the bound by 1.1% of E_in on the worst call, which is within the
  5% tolerance: `ke_excess_over_bound_max` is measured after the tolerance.

Loss floors on the 10 events (each ÷ its zero predictor): demand 7e-3, margin 4e-5, break 2e-7,
time 3e-5, damage 2e-3, motion 3e-3, def_motion 9e-3, pose 1.5e-2, impactor 9e-6, regime 1e-4,
energy 1.2e-2. The total fell from 2.14 to 0.061. `false_breaks.py` finds none.

**The 10 events pass every criterion except the closed loop,** which is pending. This
covers both families, tiers 1 and 2, perforation and arrest:
- breaks: F1 1.000 against the hard labels;
- fragments: identical on every call;
- velocities: errors a tenth of the ensemble spread at the median;
- invariants: momentum exact, energy below the bound.

### Closed loop on the same 10 events (P0, harness A, `runs/cl-eval-a-n10` in P0's worktree)

Per member: final fragments / first impactor exit speed. Recorded oracle → `a_ceiling` (the
oracle) | `a_ceiling_decoded` (a perfect /3 model through the decode) | `a_model` (the
memorised model):

| event | recorded | a_ceiling_decoded | a_model |
|---|---|---|---|
| anchored_00204 | 27, 2.35 | 27, 1.87 | **71, 2.08** (KE gain; energy bound active on 20 calls) |
| free_00502 | 9, 4.87 | 9, 4.87 | 11, 3.25 |
| free_00440 | 1, 0.13 | 1, 0.13 | 1, 0.22 |
| free_00190 | 132, 39.65 | 133, 39.70 | 131, 38.53 |
| free_00611 | 13, 4.08 | 11, 3.70 | **5, 1.10** |
| free_00757 | 99, 56.83 | 98, 56.83 | 98, 56.78 |
| free_00561 | 23, 43.74 | 23, 43.74 | 23, 43.74 |
| free_t2_00112 | 27, 0.23 | 27, 0.26 | 26, **11.85 m/s** |
| free_00576 | 1, 0.05 | 1, 0.05 | 1, 0.04 |
| free_00609 | 1, 0.15 | 1, 0.15 | 1, 0.23 |

- **The boundary and the decode reproduce the oracle when their inputs are the oracle's:**
  the perfect /3 model lands on or beside it on 9 of 10.
- **The memorised model lands near the ceiling on 6 of 10 and fails on 3,** although every
  call is memorised. After its first imperfect call it sees its own states, which training
  never showed it, and the errors compound.
- So the open milestone criterion (closed loop ≈ ceiling) is not met. The cause is the known
  distribution shift, not a representation bug.
- The remedy is DAgger on the model's rollout states (built: `dagger.py` and P0's
  `learned_impact_relabel`, verified bit for bit). It is demonstrated next on these 10 events
  on `/4`. The /3-trained checkpoint cannot be mixed with rev-6 /4 relabels, so the 10-event
  overfit is redone on `/4` first.

### DAgger round 1 on the 10 events

Steps:
- Relabelled all 779 rollout states with the oracle's island step (P0's
  `learned_impact_relabel`, 515 s on the CPU).
- Admitted 736: 43 failed the oracle's own energy balance, gaining up to 400× their input
  energy. In those, the rollout had left bodies interpenetrating and the oracle launched them.
- Fine-tuned 4,000 steps (lr 3e-4) on the originals plus the admitted relabels.
- Re-ran harness A.

Per-call memorisation slipped (F1 0.974, 48 false breaks). Final fragments and impactor
speed, oracle → round 0 → round 1:

| event | oracle | r0 | r1 |
|---|---|---|---|
| free_00611 | 13, 4.08 | 5, 1.10 | **12, 3.64** |
| free_t2_00112 | 27, 0.23 | 26, 11.85 | 35, 5.03 |
| anchored_00204 | 27, 2.35 | 71 (KE +933 E0) | 73 (KE +1,440 E0) |
| free_00190 | 132, 39.65 | 131, 38.53 | 133, 18.82 |
| free_00502 | 9, 4.87 | 11, 3.25 | 28, 3.16 |
| free_00440, 00609 | 1, 0.13 / 0.15 | KE excess 6.0 / 1.6 E0 | KE excess 0.0 / 0.0 |

One short round is mixed: one failure fixed, one improved, two regressed. Rounds 2–4 now
fine-tune to saturation (10k steps) on all relabels so far before each rollout
(`scripts/dagger_rounds_n10.sh`).

### Full-state (`/5`) heads on P1's 3-event sample

Pooled loss, 3k steps (`runs/overfit-v5/sample3-pooled`):
- breaks 143/0/0, fragments identical, mode accuracy 1.0;
- but several fields ended worse than persistence: ductility_c 28×, strain rate 4×,
  ductility 2.7×, peak margin 1.8×.
- Cause: one averaged term over 15 fields gave rarely changing fields almost no gradient.

Per-field terms, each normalised by its own persistence loss (3e9ba1d7), 6k steps:
- breaks 143/2/0, fragments identical, mode 1.0, reactions 0.21× persistence;
- every bond field now beats persistence: κ 0.12, κ_c 0.05, fail_ts / fail_c 0.10,
  crush 0.04, dissipated 0.05, peak margin 0.11, ductility 0.10, ductility_c 0.54,
  plastic 0.08–0.13, governing stress 0.09, strain rate 0.33, utilisation 0.20.

## Bugs found and fixed

1. **The survivor side of the break decision was not trained as a decision** (dd068fd4).
   - Before: broken bonds had a hinge margin (q̂ ≥ e^0.2), but survivors only had a Huber
     regression on log q. A survivor at q = 0.99 predicted at 1.01 cost about 4e-4, yet
     flipped the decision.
   - Fix: a survivor hinge, q̂ ≤ e^−0.05.
   - This alone did not fix the 10 events (row loss v2).
2. **The margin head could not represent "unchanged"** (2429f4e7, the decisive fix).
   - q = κ / κ_fail with κ a running maximum, so a surviving bond's frame peak is never below
     its frame-start margin (one event: 0 of 8,987 survivors). In quiet frames it stays put.
   - The free regression put about 5% noise on persistently near-critical damaged bonds. On
     the 10 events, 122 of the 125 false breaks were damaged bonds the oracle never breaks
     (`scripts/false_breaks.py`):
     - q̂ median 1.045, oracle q 0.96;
     - all in separation and settling frames, 100 of them in one event;
     - 3 were one-frame timing errors.
   - The break-probability head had every one of them right (p_b median 3e-6).
   - Fix: q̂ = q_start · e^softplus(·), monotone by construction, ported to Rust and GPU by P0
     (parity passes).

## Decisions recorded tonight (the user asleep; the plan's principles applied)

- **Pushforward stays off.**
  - Its successors are a partial rewrite of the carried columns: geometry, contacts and
    topology summaries remain the oracle's.
  - The planning session requires a full rebuild with the call builder's logic. That logic
    includes contact detection, cracked-face proximity, topology summaries and, for
    variant B, the quasi-static frame solve, so a lossless torch reimplementation is not
    feasible tonight.
  - The principled route is the Rust builder itself, which DAgger also needs (P0's oracle
    island relabel, requested). So the GPU-rebuild work is deferred until that path exists.
  - The topology gate (eb1bde07) already prevents contradictory labels.
- **Viability moves to `/5` (05:35, superseding the next item).**
  - P1's `/5` viability recording stops on the disk budget unless the `/4` calls go. Viability
    is defined on the full contract, and `/4` was only the interim.
  - So `/4` is freed, except for a local copy of the 10 overfit events for the DAgger
    demonstration.
  - Viability runs with `--full-state` on chunk A of `calls-v5` when P1 marks it complete.
- **Viability runs use calls `/4`** (hidden deformation inputs, full inertia) as soon as P1's
  `/4` training set completes, and are repeated on `/5` when the full-state data lands. `/4`
  is the closest available contract.
