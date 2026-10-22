"""Censored demand loss, normalisation, and L_energy's independence of booked contact
dissipation (P0)."""

import numpy as np
import torch

from learned_impact.outcome import losses as Lo
from learned_impact.outcome.model import ImpactOutcomeModel
from tests.conftest import batch_of, make_calls


def _batch():
    calls, st = make_calls(range(8))
    b = batch_of(calls, st)
    assert (b.t["y_bond_broke"] > 0).any() and (b.t["y_bond_broke"] == 0).any()
    return b, calls, st


def test_censored_demand():
    b, _, _ = _batch()
    cfg = Lo.LossConfig(alpha_cut=0, beta_support=0)
    z = Lo.zero_outputs(b)
    broke = b.t["y_bond_broke"] > 0
    y = torch.log(b.t["y_bond_peak_util"].clamp(min=cfg.log_r_floor))
    # predicting the survivors exactly and any r̂ ≥ 1 on broken bonds costs nothing
    o = dict(z)
    o["bond_log_r"] = torch.where(broke, y + 5.0, y)
    assert float(Lo.loss_terms(o, b, cfg)["demand"]) < 1e-10
    # r̂ < 1 on a broken bond is penalised, however large its saturated label is
    o["bond_log_r"] = torch.where(broke, torch.full_like(y, -0.5), y)
    assert float(Lo.loss_terms(o, b, cfg)["demand"]) > 0
    # a survivor is penalised on both sides
    for s in (+0.3, -0.3):
        o["bond_log_r"] = torch.where(broke, y + 5.0, y + s)
        assert float(Lo.loss_terms(o, b, cfg)["demand"]) > 0


def test_censored_margin():
    b, _, _ = _batch()
    cfg = Lo.LossConfig(alpha_cut=0, beta_support=0)
    z = Lo.zero_outputs(b)
    broke = b.t["y_bond_broke"] > 0
    y = torch.log(b.t["y_bond_fail_margin"].clamp(min=cfg.log_r_floor))
    o = dict(z)
    o["bond_log_q"] = torch.where(broke, y + 5.0, y)
    assert float(Lo.loss_terms(o, b, cfg)["margin"]) < 1e-10
    o["bond_log_q"] = torch.where(broke, torch.full_like(y, -0.5), y)
    assert float(Lo.loss_terms(o, b, cfg)["margin"]) > 0


def test_normalised_terms_start_near_one():
    b, _, _ = _batch()
    cfg = Lo.LossConfig()
    norms = Lo.zero_baselines([b], cfg)
    torch.manual_seed(0)
    out = ImpactOutcomeModel()(b)
    _, normed = Lo.total(Lo.loss_terms(out, b, cfg), norms, cfg)
    for k, v in normed.items():
        assert 0.5 < float(v) < 2.0, (k, float(v))
    # the full-state terms (fs_*) exist only with /5 calls
    assert set(normed) == {k for k in Lo.TERMS if not (k.startswith("fs_") or k.startswith("hid_") or k == "break_t")}


def test_energy_loss_ignores_booked_contact_dissipation():
    b, calls, st = _batch()
    cfg = Lo.LossConfig()
    torch.manual_seed(0)
    m = ImpactOutcomeModel()
    with torch.no_grad():
        e1 = float(Lo.loss_terms(m(b), b, cfg)["energy"])
        for c in calls:
            c.energy["D_contact_booked"] *= 100.0
            c.energy["D_friction_booked"] += 1e6
        b2 = batch_of(calls, st)
        e2 = float(Lo.loss_terms(m(b2), b2, cfg)["energy"])
    assert e1 == e2


def test_per_term_gradients_reach_parameters():
    from learned_impact.outcome.train import grad_norms

    b, _, _ = _batch()
    cfg = Lo.LossConfig()
    torch.manual_seed(0)
    m = ImpactOutcomeModel()
    norms = Lo.zero_baselines([b], cfg)
    _, normed = Lo.total(Lo.loss_terms(m(b), b, cfg), norms, cfg)
    g = grad_norms(normed, list(m.parameters()))
    assert all(v > 0 for v in g.values()), g


def test_dominance_monitor_flags_a_persistent_term():
    mon = Lo.DominanceMonitor(0.6, 3)
    for i in range(2):
        _, f = mon.update({"demand": 10.0, "motion": 1.0, "break": 1.0})
        assert f == []
    _, f = mon.update({"demand": 10.0, "motion": 1.0, "break": 1.0})
    assert f == ["demand"]
    _, f = mon.update({"demand": 1.0, "motion": 1.0, "break": 1.0})
    assert f == []


def test_uncertainty_weights_start_at_the_fixed_weighting():
    uw = Lo.UncertaintyWeights()
    normed = {k: torch.tensor(float(i + 2)) for i, k in enumerate(Lo.TERMS)}  # L≠1: s-grad ≠ 0
    w = Lo.LossConfig().weights
    fixed = sum(w[k] * v for k, v in normed.items())
    torch.testing.assert_close(uw(normed, w), 0.5 * fixed)
    uw(normed, w).backward()
    assert uw.log_var.grad is not None and torch.all(uw.log_var.grad != 0)


def test_body_relative_rates_rigid_zero_and_momentum_free():
    import torch

    from learned_impact.outcome.losses import body_relative_rates

    torch.manual_seed(0)
    d = torch.float64
    n = 7
    m = torch.rand(n, dtype=d) + 0.5
    x = torch.randn(n, 3, dtype=d)
    inertia = torch.diag_embed(torch.rand(n, 3, dtype=d) + 0.1)
    body = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    M = torch.zeros(2, 1, dtype=d).index_add_(0, body, m[:, None])
    X = torch.zeros(2, 3, dtype=d).index_add_(0, body, m[:, None] * x) / M
    vb, wb = torch.randn(2, 3, dtype=d), torch.randn(2, 3, dtype=d)
    v = vb[body] + torch.cross(wb[body], x - X[body], dim=-1)
    w = wb[body]
    dv, dw = body_relative_rates(v, w, x, m, inertia, body, 2)
    assert dv.abs().max() < 1e-10 and dw.abs().max() < 1e-10  # rigid: no hidden rates
    dv, dw = body_relative_rates(v + 0.01 * torch.randn(n, 3, dtype=d),
                                 w + 0.01 * torch.randn(n, 3, dtype=d), x, m, inertia, body, 2)
    r = x - X[body]
    P = torch.zeros(2, 3, dtype=d).index_add_(0, body, m[:, None] * dv)
    L = torch.zeros(2, 3, dtype=d).index_add_(
        0, body, torch.cross(r, m[:, None] * dv, dim=-1) + (inertia @ dw[..., None])[..., 0])
    assert P.abs().max() < 1e-12 and L.abs().max() < 1e-10  # zero net momentum per body


def test_group_balancer_reaches_the_target_shares():
    gb = Lo.GroupBalancer()
    true = {"margin": 0.02, "motion": 5.0, "energy": 1.0, "fs_mode": 0.5, "fs_chunk": 0.5}
    for _ in range(3):  # gradient norms scale with the multipliers; closed form: on target at once
        g = {k: v * gb.mult[gb.group_of[k]] for k, v in true.items()}
        shares = gb.update(g)
    for grp, sh in shares.items():
        assert abs(sh - Lo.GROUP_TARGET[grp]) < 0.05, (grp, sh)
    assert abs(sum(np.log(gb.mult[g]) for g in shares)) < 1e-6  # geometric mean 1
