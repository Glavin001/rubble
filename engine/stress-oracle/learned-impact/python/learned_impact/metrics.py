"""Eval metrics: fail F1, util MAE, delta RMSE (teacher-forced per frame)."""

from __future__ import annotations

import torch


@torch.no_grad()
def binary_fail_f1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Fail = class > 0. Returns F1 in [0, 1]; 1.0 if both sides empty."""
    pred = logits.argmax(dim=-1) > 0
    tgt = target > 0
    tp = int((pred & tgt).sum())
    fp = int((pred & ~tgt).sum())
    fn = int((~pred & tgt).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


@torch.no_grad()
def util_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean())


@torch.no_grad()
def delta_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).pow(2).mean().sqrt())


@torch.no_grad()
def eval_batch(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "fail_f1": binary_fail_f1(pred["bond_fail_logits"], batch["bond_fail_class"]),
        "util_mae": util_mae(pred["bond_peak_util"], batch["bond_peak_util"]),
        "delta_rmse": delta_rmse(pred["chunk_state_delta"], batch["chunk_state_delta"]),
    }


@torch.no_grad()
def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"fail_f1": 0.0, "util_mae": 0.0, "delta_rmse": 0.0, "n": 0.0}
    keys = [k for k in rows[0] if k != "n"]
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["n"] = float(len(rows))
    return out
