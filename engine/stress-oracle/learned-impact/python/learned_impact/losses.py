"""Training losses (focal fail-class, util near 1, state MSE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import K


def focal_ce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    pt = p.gather(1, target[:, None]).squeeze(1)
    log_pt = log_p.gather(1, target[:, None]).squeeze(1)
    return (-((1.0 - pt) ** gamma) * log_pt).mean()


def util_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Targets are log1p(peak_util). Heavier weight near fail threshold (util≈1 → log1p≈0.693).
    fail_log = 0.693147  # log1p(1)
    w = 1.0 + 4.0 * torch.exp(-((target - fail_log) ** 2) / 0.05)
    return (w * F.huber_loss(pred, target, reduction="none", delta=0.5)).mean()


def total_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    w_fail: float = 1.0,
    w_util: float = 1.0,
    w_delta: float = 1.0,
) -> dict[str, torch.Tensor]:
    fail = focal_ce(pred["bond_fail_logits"], batch["bond_fail_class"].clamp(0, K))
    util = util_loss(pred["bond_peak_util"], batch["bond_peak_util"])
    delta = F.mse_loss(pred["chunk_state_delta"], batch["chunk_state_delta"])
    total = w_fail * fail + w_util * util + w_delta * delta
    return {"loss": total, "fail": fail.detach(), "util": util.detach(), "delta": delta.detach()}
