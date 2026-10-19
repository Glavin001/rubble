# Learned impact — small-data prove (2026-10-19)

## Setup
- Model: MGN-T residual, H=48 P=16 (~104k params)
- Set: `--prove` → `/tmp/li-prove-set` · **17 events / 43 frames / 1.4 min** generate
  - walls push/ram/punch, cantilevers n=6/10/14, arch loop, frame column, masonry house (impact+settle), analytic anchors
- Training notes: `learned-impact/TRAINING.md` (MeshGraphNets / MGN-T / plan §6–9)

## Holdouts (structure / load)

| Holdout | Train→last | Holdout loss | Holdout fail F1 |
|---|---|---|---|
| `masonry_house` | 2.62→0.021 | 0.55 | **0.42** |
| `wall_b5` (no walls in train) | 2.24→0.051 | 0.89 | **0.25** |
| `cantilever` | 2.62→0.024 | 0.16 | **1.00** |
| punch loads `ram_25/40` | 2.62→0.023 | 0.47 | **0.73** |

Earlier wall-only dense speed holdout (separate set): F1 **0.84**.

## Inference (PyTorch MPS, serial N× one graph — not Slang)

| N buildings | Wall 2018 bonds total_ms | House 1299 bonds total_ms |
|---|---|---|
| 1 | 1.2 | 1.1 |
| 20 | 21.7 | 21.1 |
| 100 | 109 | 102 |

≈ **1.05–1.1 ms/building** on MPS for this graph size. Plan §9 GPU Slang targets are ≤4 ms @20 / ≤8 ms @100 (30% active) — **not met by this PyTorch proxy**; M0 Slang measurement still required. Linear scaling is healthy.

## Verdict
- **E2E path works** on a small diverse set including a real house.
- **Within-family** (punch speeds, cantilevers) looks learnable.
- **Cross-family to walls without wall data** still fails (F1 0.25) — expected; walls must stay in the mix.
- **House holdout F1 0.42** is partial transfer from walls/frames — promising for small data, not M2-gated yet (want within ~0.1 of seen-type F1).
