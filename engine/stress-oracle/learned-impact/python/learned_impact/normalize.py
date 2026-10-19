"""Per-batch feature scaling (plan §4.1: lengths / stresses / loads)."""

from __future__ import annotations

import torch


def _rms(x: torch.Tensor, floor: float) -> float:
    if not x.numel():
        return floor
    v = float(x.detach().float().pow(2).mean().sqrt().item())
    return max(v, floor)


def normalize_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a shallow-copied batch with float inputs/targets scaled to ~O(1)."""
    out = dict(batch)
    # Characteristic sizes / forces from this graph.
    he = batch["chunk_half_extents"].abs().mean().clamp_min(1e-3)
    mass_scale = batch["chunk_mass"].abs().mean().clamp_min(1e-3)
    load_scale = _rms(batch["chunk_load_bins"], 1.0)
    qs_scale = _rms(batch["bond_qs_force"], 1.0)
    joint_scale = _rms(batch["bond_joint_state"], 1.0)

    out["chunk_mass"] = batch["chunk_mass"] / mass_scale
    out["chunk_inertia"] = batch["chunk_inertia"] / (mass_scale * he * he)
    out["chunk_half_extents"] = batch["chunk_half_extents"] / he
    # u,θ ~ length/1; v,ω ~ length (crude)
    cs = batch["chunk_state"].clone()
    cs[:, 0:3] = cs[:, 0:3] / he
    cs[:, 6:9] = cs[:, 6:9] / he
    out["chunk_state"] = cs
    cd = batch["chunk_state_delta"].clone()
    cd[:, 0:3] = cd[:, 0:3] / he
    cd[:, 6:9] = cd[:, 6:9] / he
    out["chunk_state_delta"] = cd

    out["chunk_load_bins"] = batch["chunk_load_bins"] / load_scale
    out["chunk_gravity_local"] = batch["chunk_gravity_local"] / 9.81
    out["bond_joint_state"] = batch["bond_joint_state"] / joint_scale
    out["bond_qs_force"] = batch["bond_qs_force"] / qs_scale
    # Impact peak util can be ≫1; log1p keeps targets O(1) without crushing the tail.
    out["bond_peak_util"] = torch.log1p(batch["bond_peak_util"].clamp_min(0.0))
    out["_scales"] = {
        "he": float(he),
        "mass": float(mass_scale),
        "load": load_scale,
        "qs": qs_scale,
        "joint": joint_scale,
    }
    return out
