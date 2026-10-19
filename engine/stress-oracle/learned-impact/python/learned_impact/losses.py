"""Training losses (focal fail-class, util near 1, state MSE)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import K


def focal_ce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    if not target.numel():
        return logits.sum() * 0
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    pt = p.gather(1, target[:, None]).squeeze(1)
    log_pt = log_p.gather(1, target[:, None]).squeeze(1)
    return (-((1.0 - pt) ** gamma) * log_pt).mean()


def util_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not target.numel():
        return pred.sum() * 0
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


def fracture_cdf_squared_error(break_logit, time_logits, labels):
    """Ranked probability score terms for bins 1..K, then no break in interval.

    CDF[k] = P(break by boundary k), including the no-break probability mass.
    A confident error spanning three bins costs three times a one-bin error.
    Scores every live bond; no conditioning on successful fracture detection.
    Labels 0 mean right-censored beyond this interval, not 'never fractures'.
    """
    probability = break_logit.sigmoid()[:, None] * time_logits.softmax(-1).cumsum(-1)
    boundaries = torch.arange(1, time_logits.shape[-1]+1, device=labels.device)
    target = ((labels[:, None] > 0) & (labels[:, None] <= boundaries)).to(probability.dtype)
    return (probability-target).square()


def fracture_event_nll(break_logit, time_logits, labels):
    """Per-bond negative log likelihood of no break or a first-break time bin.

    P(0)=1-sigmoid(b); P(k)=sigmoid(b)*softmax(t)[k-1].
    Average over all eligible bonds, including intact intervals. This is a
    normalized categorical likelihood; it introduces no positive-class weight.
    Intact means no observed break during this interval, not infinite survival.
    """
    if break_logit.shape != labels.shape or time_logits.shape[:-1] != labels.shape:
        raise ValueError('fracture event likelihood shape mismatch')
    if labels.dtype != torch.int64 or ((labels < 0) | (labels > time_logits.shape[-1])).any():
        raise ValueError('fracture event labels outside no-break/time-bin range')
    positive = labels > 0
    binary = F.binary_cross_entropy_with_logits(break_logit, positive.to(break_logit.dtype), reduction='none')
    timing = -F.log_softmax(time_logits, dim=-1).gather(-1, (labels-1).clamp_min(0)[..., None]).squeeze(-1)
    return binary + torch.where(positive, timing, torch.zeros_like(timing))
