"""Censored demand loss, normalisation, and L_energy's independence of booked contact
dissipation (P0)."""

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


def test_normalised_terms_start_near_one():
    b, _, _ = _batch()
    cfg = Lo.LossConfig()
    norms = Lo.zero_baselines([b], cfg)
    torch.manual_seed(0)
    out = ImpactOutcomeModel()(b)
    _, normed = Lo.total(Lo.loss_terms(out, b, cfg), norms, cfg)
    for k, v in normed.items():
        assert 0.5 < float(v) < 2.0, (k, float(v))
    assert set(normed) == set(Lo.TERMS)


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
