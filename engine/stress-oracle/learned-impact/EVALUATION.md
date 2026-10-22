# Evaluating the learned impact-outcome model without a human reviewer

Goal: the agent that trains, selects and deploys the model can decide on its own, and be
trusted, whether a checkpoint is good. That needs four things:
- ground truth that carries its own spread;
- gates fixed before results are seen;
- a test of the deployed behaviour (closed loop), not only of the per-frame prediction;
- controls proving the tests themselves can tell good from bad.

## 1. Rules that keep the evaluation honest

- **Pre-registered gates.** `findings/gates-v1.json` (written 2026-10-21, before the first real
  training run) fixes, per gate:
  - the metric;
  - the bar: the oracle-calibrated skill rule, max(1.25, the held-out member's mean);
  - the scenes and families it applies to;
  - the aggregation: per call within a family, against the held-out member (below).
  - Changing it after results exist is a new version with the reason recorded, and the
    old verdict stays reported.
- **The pre-registered ensemble gate (family acceptance, v1, 2026-10-21).** This is
  `metrics.family_acceptance`, reported by `report.py` (`acceptance`, `accepted`).
  - **Unit.** Every active ensemble call: an impact or secondary contact, or a break in any
    member, with ≥ 3 aligned members. The gates are ram exit speed, partition and fragment
    sizes, each scored by `calibrated_gate` (skill ≤ max(1.25, held-out member mean), with
    the `TOLERANCE` floors).
  - **Reference.** The held-out oracle member, scored by the identical rule. Each member in
    turn is the candidate against the other members, and its own bar comes from those
    remaining members (leave-one-out). So the oracle's own pass rate and mean skill ÷ bar
    are what a candidate must match. They are not 100% and 1: P0 found the oracle fails
    the literal bar on some scenes.
  - **Rule, per family and per metric.** Accept ⇔ the model's pass rate ≥ the held-out
    member pass rate **and** the model's mean skill ÷ bar ≤ the held-out member mean, over
    the family's active ensemble calls. A family is accepted when all three metrics accept.
    The checkpoint is accepted (`accepted`) when every family is.
  - **Uncertainty.** We report 95% percentile bootstrap CIs (2,000 resamples, seed 0) for
    each rate and for the paired differences. The bootstrap resamples whole ensembles,
    because calls of one ensemble are correlated. `confident` additionally requires each
    difference's CI to exclude the failing side. It is reported, not required: with about
    10² ensembles per family the CIs are wide, and requiring it would reject an
    oracle-equivalent model about half the time.
- **gates-v1.1 (2026-10-21, the current gate; adopts P0's aggregate rule).** Same unit,
  reference and bootstrap as v1, but a non-inferiority margin replaces the point comparison.
  Per family and metric, accept ⇔ both of these hold:
  - the one-sided 95% upper bootstrap bound of (held-out member pass rate − model pass
    rate) is ≤ 5 points;
  - the one-sided 95% upper bootstrap bound of (model mean skill ÷ bar ÷ member mean − 1)
    is ≤ 5%.
  - Family and checkpoint acceptance are as in v1. `accept` is v1.1; v1's verdict stays
    reported as `accept_v1`.
  - Reason: v1 compared point estimates, so a model equivalent to the oracle failed about
    half the time on sampling noise. The bound puts the burden of proof on the candidate
    and allows a stated, small margin (P0's closed-loop rule, so the per-call and full-event
    gates agree).
- **Selection is on dense metrics, not on the gate.**
  - Checkpoints, τ and the temperature are chosen on the validation selection score
    (`EvalResult.score`).
  - The score is computed over active calls from:
    - CRPS of the ram exit speed ÷ v_ref;
    - 1 − ARI on mass, against every member;
    - W1 on log mass, against every member.
  - Each term is ÷ its tolerance (0.05, 0.05, 0.1). The score is the mean over families plus
    ¼ of the worst family.
  - Single runs (no ensemble) use their errors, which is CRPS with one member.
  - The pass/fail gate is too coarse to select on at these sample sizes, so it is used for
    reporting only.
- **Split discipline.**
  - Validation selects checkpoints, λ, τ and the temperature. It uses the bulk v1
    evaluation ensembles (`calls-v3-eval`, val split; about 130 ensembles).
  - **τ is frozen on validation** (stored in the checkpoint) and **confirmed on test**.
    Confirmation means `report.py --split test` evaluates the acceptance gate at the frozen
    τ. Its sweep over the same τ grid (`tau_sweep`, with the regret of the frozen choice)
    is informational and never re-selects.
  - The test set is for confirming the frozen checkpoint and τ, and for reporting.
  - The sealed set is opened once, at P4: `report.py` refuses `--split sealed` without
    `--confirm-sealed`.
  - Splits are by layout group, and `split_group` leakage fails loudly (`train.Data`).
- **Worst case next to the median.** Every gate is per scene and per family. Failures are
  listed by event and call (`three_way` → `worse_than_qs`), never averaged away.

## 2. What exists now (P2, offline, per call)

| Tool | What it decides |
|---|---|
| `metrics.calibrated_gate`, `evaluate` | skill against the oracle ensemble on ram speed, partition (mass-weighted ARI) and fragment sizes (W1 on log mass), with the held-out-member bar; fragments in range; bond P/R/F1 against consensus; ECE/Brier of breaks and regime; false breaks on no-break calls; energy-bound activation rate |
| `metrics.three_way`, `report.py` | model vs `blast_parity` QS vs the CPU punch on identical calls: superiority and non-inferiority per family, plus the list of calls where the model is worse than QS |
| `metrics.fit_ballistic_limit` | the Lambert–Jonas v_bl from (v0, exit speed) points |
| decode invariants (tests) | momentum exact, KE bound, monotone damage, invariance, masking across cracks |
| training monitors | per-term gradient shares with a dominance warning (V34), and `status.json` |

**Output-side ceiling (`ceiling.py`, provider `--mode ceiling`).** This is the first test on real
calls, run before any training. The oracle's outcome is converted into raw head outputs, pushed
through the deployed decode (projection, KE bound at the configured tolerance, τ, bins,
damage clamp), and scored with the real metrics. It also attributes the error to each stage:
- the encoding round trip;
- the projection (non-zero when the oracle's "free" island does not conserve momentum as
  assumed);
- the bound, with its activation rate at tolerances 0–0.1 (P0: ledger gains up to 4.7%);
- τ on the oracle's own peak demand.

In closed loop the same decode answers every call from the harness's `oracle.json`, so the
full-event metrics show what the output side alone costs. Synthetic toy: exact. (It found
that the toy itself conserved angular momentum with frame-start instead of frame-end
geometry; fixed.) Real data: pending the call builder.

These are necessary but **not sufficient**. P0 and the V31 lesson show that one-frame
accuracy can hide a catastrophic full run: 1,258 breaks against 324, with the projectile
only 0.5 mm off.

## 3. What is still needed, and from whom

| # | Need | Why | Owner | Status |
|---|---|---|---|---|
| 1 | **Closed-loop rollout harness**: every routed island's frame comes from the model, handed off at window end with the contact filtered (P0's `window_reduced`), through to the end of the life cycle; it writes outcome.json and a frame record | the only test of what ships: fragment distribution, support survival, exit speed and trajectories over 0.5–1 s, multi-wall cascades | P0 (`learned_p0 ceiling --variant provider`, e6d54614 on P0's branch) + P2 `outcome.provider` | **built**; self-checks pass (P0). My provider in echo mode through P0's harness reproduces wall_row 23.622 / 133 / 401 exactly. Model mode waits on the call builder's `call_record` |
| 2 | **Harness self-check**: the same harness with the *oracle* as the model must reproduce P0's ceiling (pass on both M5 scenes), and with the punch it must fail as P0 measured | proves the harness, not the model, is right before trusting any verdict | P2, once #1 exists | — |
| 3 | **Ensembles** (8 members) on val, test, sealed and the M5 scenes | the truth is a distribution; the bar is calibrated by it | P1 | agreed |
| 4 | **Ballistic-limit sweeps**: fine v0 grids across each family's transition, with ensembles | the curve gates (v_bl inside the ensemble spread; calibrated perforation frequency in the ambiguous band) | P1 recipes, P2 fit | to request with the first families |
| 5 | **DAgger and topology-true pushforward**: building a call from an arbitrary state, and oracle restarts from the decoded state | without them, validation only sees oracle-state inputs and over-trusts the model | build-from-state: the Rust call builder (own session, not started); restarts: P1 recorder mode `--restart` | agreed; restart waits on decoded states (format sent) |
| 6 | **Controls**: a shuffled-target model and a zero-velocity model must fail every gate; a held-out oracle member must pass at its calibrated rate | proves the gates discriminate | P2 | to add with real data |
| 7 | **Out-of-distribution gate**: input ranges from training, plus residuals (energy-bound activation, regime/cut inconsistency), with a logged fallback rate on test | a deployed model must know when not to answer | P2 (Python reference), P3 (engine) | to build |
| 8 | **Deployment parity**: PyTorch vs Slang (Metal and WGSL) within fp32, identical break sets on validation; guard digests unchanged with the flag off | the checkpoint evaluated must be the checkpoint that runs | P3 | later |
| 9 | **Cost in the same report**: per-call ms on Metal and Chrome at M5 and town sizes, one job at a time | accuracy per millisecond decides routing | P3 / P0 probe | later |
| 10 | **Visual playback**: fracture-lab screenshots of oracle / QS / punch / model side by side for every gate scene, captured in the browser pane | catches failures metrics miss (drifting walls, frozen debris) | P2, once #1 exists | — |

## 4. The verdict

One command, after #1, will produce `findings/<run>-verdict.{json,md}`. It runs:
- the offline report on val (and test);
- the closed loop on the M5 ensembles and the val families;
- the ballistic-limit curves;
- the controls.

Each gate in `gates-v1.json` passes or fails per scene, with links to the failing events
and screenshots. A checkpoint is deployable only if every pre-registered gate passes, the
controls behave, and parity holds. Otherwise the verdict says which gate failed and the
checkpoint stays behind the flag. Adoption into a profile remains the user's decision, as
the plan says.
