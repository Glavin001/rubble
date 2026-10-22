# P2 bulk v1 retrain: gates-v1.1 on the evaluation ensembles (2026-10-22)

Checkpoint: `runs/v4/bulk-w32/best.pt`, with the weights export in `runs/v4/bulk-w32/export`.
- **Model:** W32, rounds 2,1,2,1,2, levels 2, variant B, input standardisation.
- **Loss:** break/margin weight 5, no class reweighting, pushforward 0.3.
- **Training:** 10,000 steps on MPS, 94 min, on P1's 828 bulk v1 training events (`calls-v3-train`, calls `/3`).
- **Validation:** the bulk v1 evaluation ensembles (`calls-v3-eval`, val split): 1,055 events, 49,266 calls, 125 complete 8-member ensembles with aligned calls (28 anchored-wall, 97 free-wall).
- **Selection:** on the dense score (EVALUATION.md §1). Best at step 8,000, score 2.19. τ was chosen on validation and frozen: 1.0 (scores 3.41 / 2.75 / 2.37 / 2.24 / 2.19 / 2.19 for τ 0.5–1.0).

**Verdict: not accepted (gates-v1.1).** Neither family passes; the test split stays unopened for a verdict until one does.

## gates-v1.1 per family (active ensemble calls, held-out oracle member by the identical rule)

| family | gate | model pass | member pass | shortfall, 95% upper bound (≤ 0.05) | model skill/bar | member skill/bar | excess, 95% upper bound (≤ 5%) |
|---|---|---|---|---|---|---|---|
| anchored_wall (28 ens, 467 calls) | ram | 0.903 | 0.940 | +0.098 ✗ | 0.48 | 0.27 | +145% ✗ |
| | partition | 0.891 | 0.840 | +0.058 ✗ | 0.66 | 0.54 | +69% ✗ |
| | sizes | 0.899 | 0.840 | +0.036 ✓ | 0.72 | 0.52 | +95% ✗ |
| free_wall (97 ens, 550 calls) | ram | 0.945 | 0.980 | +0.052 ✗ | 0.34 | 0.21 | +103% ✗ |
| | partition | 0.773 | 0.868 | +0.160 ✗ | 1.47 | 0.73 | +138% ✗ |
| | sizes | 0.809 | 0.859 | +0.089 ✗ | 1.32 | 0.69 | +120% ✗ |

Other validation metrics:
- bond F1 0.59 (P 0.70, R 0.51); false breaks on no-break calls 0.0%; fragment count in the ensemble range 0.87; energy bound active on 0.7% of calls.
- Three-way: superiority over `blast_parity` QS and the CPU punch on both families; non-inferiority to QS passes on both.
  - On ram, skill/bar is 0.48 vs QS 1.89 (anchored) and 0.34 vs 1.34 (free).
  - On free-wall partition and sizes it is 1.47 / 1.32 vs QS 1.65 / 1.71.

## What it says

1. **The binding constraint is the mean skill ratio, not the pass rate.**
   - Pass rates are within a few points of the oracle's. On anchored walls the model passes partition and sizes *more* often than a held-out member.
   - But its mean skill ÷ bar is 1.3–2.4× the member's. The member's own mean is far below 1 (0.2–0.7), because the bar is max(1.25, held-out mean).
   - So v1.1's "within 5% of the member's mean" asks for oracle-level agreement on every call, not just inside the bar. That is the gate as registered.
2. **Free-wall partition is the weakest point:** pass 0.77 vs 0.87, skill/bar 1.47. This matches the closed-loop videos (wrong fragment partition, over-breaking on new geometry) and the flat 1 − ARI (0.14) through training.
3. **Training plateaus early.** The dense score was 2.27 at step 1,000 and 2.19 at 8,000. More steps on this data are not the lever.
4. **Boundary.** These are `/3` calls. P0's cl-eval-v2 showed that the reduced boundary (no hidden deformation) fails fragment motion even for a perfect model. The planning session has since asked for the full-state contract (`/5`) before any simplification, so the next model is trained on that.

## Reproduce

```bash
cd engine/stress-oracle/learned-impact/python
.venv/bin/python -m learned_impact.outcome.train --dataset <P1>/runs/calls-v3-train --val-dataset <P1>/runs/calls-v3-eval \
  --out ../runs/v4/bulk-w32 --steps 10000 --eval-every 1000 --pushforward 0.3 --device mps
.venv/bin/python -m learned_impact.outcome.report --checkpoint ../runs/v4/bulk-w32/best.pt --dataset <P1>/runs/calls-v3-eval --split val --out report-val.json
```
(The `/3` reports run from the code at e25b2649–a6b9d4ec; the working tree has since moved to `/4`.)
