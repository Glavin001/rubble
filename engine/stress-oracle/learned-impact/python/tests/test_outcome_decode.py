"""Decode: exact momentum after projection, energy bound, monotone damage, break modes."""

import numpy as np
import pytest
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome.geometry import world_inertia
from learned_impact.outcome.model import ImpactOutcomeModel
from tests.conftest import batch_of, make_calls


def _outputs(b, scale=3.0, seed=0):
    torch.manual_seed(seed)
    m = ImpactOutcomeModel()
    with torch.no_grad():
        o = m(b)
        g = torch.Generator().manual_seed(seed)
        o["chunk_out"] = torch.randn(o["chunk_out"].shape, generator=g) * scale
        o["imp_out"] = torch.randn(o["imp_out"].shape, generator=g) * scale
        o["bond_ddamage"] = torch.rand(o["bond_ddamage"].shape, generator=g)
    return o


def _momenta(ph, G, end):
    if end:
        return D.momentum(ph, ph["v_end"], ph["w_end"], ph["pos_end"], ph["inertia_end"], G)
    i0 = world_inertia(ph["q"], ph["inertia"])
    return D.momentum(ph, ph["v"], ph["w"], D._com_rel(ph, ph["pos"], G), i0, G)


@pytest.mark.parametrize("gravity", [False, True])
def test_momentum_exact_after_projection(gravity):
    calls, st = make_calls(range(6), anchored=False, gravity=gravity)
    b = batch_of(calls, st)
    dec = D.decode(_outputs(b), b, D.DecodeConfig(energy_bound=False))
    ph, G = dec["_phys"], b.n_calls
    assert dec["free"].all()
    p0, l0 = _momenta(ph, G, end=False)
    p1, l1 = _momenta(ph, G, end=True)
    mt = D.scatter_sum(ph["mass"].unsqueeze(-1), ph["call"], G)
    dt = 1 / 60
    g = b.t["gravity"]
    scale = (ph["mass"].unsqueeze(-1) * ph["v"].abs()).sum() + 1.0
    assert ((p1 - p0 - mt * g * dt).abs().max() / scale) < 1e-12
    assert ((l1 - l0).abs().max() / scale) < 1e-11
    # centre-of-mass displacement exact: Σ m Δx = P0 dt + ½ M g dt²
    mdx = D.scatter_sum(ph["mass"].unsqueeze(-1) * ph["dx"], ph["call"], G)
    torch.testing.assert_close(mdx, p0 * dt + 0.5 * mt * g * dt * dt, rtol=1e-12, atol=1e-9)


def test_momentum_still_exact_after_energy_bound():
    calls, st = make_calls(range(6), anchored=False)
    b = batch_of(calls, st)
    dec = D.decode(_outputs(b, scale=10.0), b, D.DecodeConfig(energy_tolerance=0.0))
    ph, G = dec["_phys"], b.n_calls
    assert dec["energy_bound_active"].any()
    p0, l0 = _momenta(ph, G, end=False)
    p1, l1 = _momenta(ph, G, end=True)
    scale = (ph["mass"].unsqueeze(-1) * ph["v"].abs()).sum() + 1.0
    assert ((p1 - p0).abs().max() / scale) < 1e-12
    assert ((l1 - l0).abs().max() / scale) < 1e-11
    i1 = ph["inertia_end"]
    ke1 = D.kinetic(ph, ph["v_end"], ph["w_end"], i1, G)
    ke0 = D.kinetic(ph, ph["v"], ph["w"], world_inertia(ph["q"], ph["inertia"]), G)
    feasible = ~dec["energy_bound_infeasible"]
    assert torch.all(ke1[feasible] <= ke0[feasible] * (1 + 1e-9) + 1e-9)


def test_anchored_islands_are_not_projected_but_bounded():
    calls, st = make_calls(range(4), anchored=True)
    b = batch_of(calls, st)
    o = _outputs(b, scale=10.0)
    dec = D.decode(o, b, D.DecodeConfig(energy_tolerance=0.05))
    assert not dec["free"].any()
    ph = dec["_phys"]
    anch = ph["anchored"]
    assert torch.all(ph["v_end"][anch] == 0)
    raw = D.to_physical(o, b)
    # anchored: only a uniform scale s ≤ 1 per island
    ratio = (ph["v_end"].norm(dim=-1) / raw["v_end"].norm(dim=-1).clamp(min=1e-30))[~anch]
    assert torch.all(ratio <= 1 + 1e-12)


def test_damage_is_monotone():
    calls, st = make_calls([3])
    b = batch_of(calls, st)
    o = _outputs(b)
    dec = D.decode(o, b)
    d0 = b.edges["bond"]["x"][0::2, D.S.col("bond_feat", "damage")].double()
    assert torch.all(o["bond_ddamage"] >= 0)
    assert torch.all(dec["bond_damage_end"].double() >= d0 - 1e-7)
    assert torch.all(dec["bond_damage_end"] < 1.0)  # survivors never reach 1 (only q̂ ≥ τ breaks)


def test_threshold_and_hash_sampling():
    calls, st = make_calls(range(3))
    b = batch_of(calls, st)
    o = _outputs(b)
    o["bond_log_q"] = torch.linspace(-1, 1, o["bond_log_q"].shape[0])
    o["bond_log_r"] = torch.linspace(1, -1, o["bond_log_r"].shape[0])
    br = D.breaks(o, b, D.DecodeConfig(tau=1.0))
    assert torch.equal(br, o["bond_log_q"] >= 0)  # the failure margin decides
    bu = D.breaks(o, b, D.DecodeConfig(tau=1.0, break_mode="utilization"))
    assert torch.equal(bu, o["bond_log_r"] >= 0)  # plan v1's test, kept as a mode
    o["bond_break_logit"] = torch.zeros_like(o["bond_log_r"])
    s1 = D.breaks(o, b, D.DecodeConfig(break_mode="sample", sample_seed=7))
    s2 = D.breaks(o, b, D.DecodeConfig(break_mode="sample", sample_seed=7))
    s3 = D.breaks(o, b, D.DecodeConfig(break_mode="sample", sample_seed=8))
    assert torch.equal(s1, s2) and not torch.equal(s1, s3)
    u = D.hash_uniform(torch.arange(100_000, dtype=torch.int64) | (3 << 32), 1).numpy()
    assert 0 <= u.min() and u.max() < 1
    assert abs(u.mean() - 0.5) < 0.01 and abs(np.mean(u < 0.1) - 0.1) < 0.01
