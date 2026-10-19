# Training notes (plan + papers + what we measured)

Sources: `engine/docs/learned-stress-solver-plan.md` §5–7; Pfaff et al. MeshGraphNets (ICLR 2021); MeshGraphNet-Transformer (arXiv:2601.23177).

## What the model likes

| Signal | Why (paper / plan) | What we saw |
|---|---|---|
| **Residual on QS** | Physics carries statics; net only learns dynamic correction | Overfit converges; util/fail heads learn |
| **Dense coverage in one family** | MGN-T used 18 trajectories *one* family | Wall speed holdout F1 0.84 with dense sweep; sparse mini wall-holdout failed (~0.94 loss) |
| **Teacher forcing → rollout FT** | MeshGraphNets / pushforward: unroll 4–16 steps with input noise | We have teacher-forced only; 8-frame clips already help F1 |
| **Focal fail + util near 1** | Failures rare; threshold matters most | Fail head learns first; log1p util stabilizes impact peaks |
| **Attention / global tokens** | Stress wave spans a building in one frame; local MP under-reaches | MGN beats MLP slightly on multi-event fit |
| **Whole-type / whole-load splits** | Plan: never random frames | Structure + load holdouts are the gate |
| **Scaled features** | Plan §4.1 lengths/stresses/loads | ∞ ductility / raw util≫1 blew loss until packed/normalized |

## What it dislikes

| Anti-pattern | Why | Mitigation |
|---|---|---|
| Sparse one-shot of a hard family | No coverage → holdout collapses | Speed sweep + houses + cantilevers before claiming transfer |
| Cross-family with tiny data | Plan puts structure transfer at ~45% confidence | Hold out house *after* walls+frames in train; don't expect miracle |
| Autoregressive without rollout FT | Compounding error (MeshGraphNets) | Next: 4–16 step unroll + noise |
| Feeding sentinel ductility / huge util | Dominates Huber/MSE | Pack non-finite ductility as 0; log1p util |
| Random frame splits | Leakage across time | Hold out event / structure / load type |
| Oversized H for M0 | Plan: H 32–64, ≤1M params | Stay ~104k (H48) until Slang speed proof |

## Small-data prove protocol (this track)

1. Generate `--prove` (walls push/ram/punch, cantilevers, arch loop, frame, masonry house).
2. Train with `--holdout-structure-type masonry_house` (and separately `wall_b5`).
3. Gate: holdout fail F1 within ~0.1 of train-family F1 (plan M2); train loss drops; no NaNs.
4. Inference bench: `python -m learned_impact.bench_infer` for N=1…100 (PyTorch proxy; Slang is the runtime target).

## Inference cost (plan §9, M3 Max estimates)

| Scenario | Active bonds | Lean H32 | Best H64 |
|---|---|---|---|
| 1 large town building | 5k | 0.5 ms | 0.6 ms |
| 20 impacted | 34k | 2.3 ms | 3.8 ms |
| 100 @ 30% active | 46k | 3.1 ms | 5.1 ms |
| 100 fully active | 152k | 9.7 ms | 17 ms |

**Targets:** ≤4 ms @ 20 impacted; ≤8 ms @ 100 town (30% active). Measured PyTorch numbers replace nothing until Slang M0 — they only trend CPU/MPS cost.
