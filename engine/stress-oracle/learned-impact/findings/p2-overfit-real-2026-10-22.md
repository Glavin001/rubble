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
N100_ROW

Loss floors on the 10 events (each ÷ its zero predictor): demand 7e-3, margin 4e-5, break 2e-7,
time 3e-5, damage 2e-3, motion 3e-3, def_motion 9e-3, pose 1.5e-2, impactor 9e-6, regime 1e-4,
energy 1.2e-2. The total fell from 2.14 to 0.061. `false_breaks.py` finds none.

**The 10 events pass every criterion except the closed loop,** which is pending. This
covers both families, tiers 1 and 2, perforation and arrest:
- breaks: F1 1.000 against the hard labels;
- fragments: identical on every call;
- velocities: errors a tenth of the ensemble spread at the median;
- invariants: momentum exact, energy below the bound.

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
- **Viability runs use calls `/4`** (hidden deformation inputs, full inertia) as soon as P1's
  `/4` training set completes, and are repeated on `/5` when the full-state data lands. `/4`
  is the closest available contract.
