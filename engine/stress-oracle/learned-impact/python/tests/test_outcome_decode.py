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
    if gravity:
        # with gravity the deployed rule cannot rule out ground contact: not free; the
        # projection's own exactness (P0 + M g dt) is checked by calling it directly
        assert not D.causal_freedom(b)[0].any()
        ph = D.project(D.to_physical(_outputs(b), b), b, torch.ones(b.n_calls, dtype=torch.bool))
        dec = {"_phys": ph}
    else:
        # (the energy envelope, the hard admissibility constraint, overrides exact momentum
        # when it fires: off here)
        dec = D.decode(_outputs(b), b, D.DecodeConfig(energy_bound=False, energy_envelope=None))
        assert dec["free"].all()
    ph, G = dec["_phys"], b.n_calls
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


def test_decode_reads_inputs_only():
    """Decode output is identical with the oracle's targets present and with dummy targets
    (closed loop): nothing at inference reads y_*, ens_*, base_* or the energy ledger."""
    from learned_impact.outcome import schema as S

    for kw in (dict(anchored=False), dict(anchored=True), dict(anchored=False, gravity=True)):
        calls, st = make_calls(range(3), **kw)
        blind = []
        for c in calls:
            # the ledger's KE_in and E_elastic_start are inputs: the frame-start kinetic and
            # stored energy (the energy envelope's E)
            stripped = S.Call(**{**c.__dict__, "arrays": {
                k: v for k, v in c.arrays.items() if not S._is_target(k)},
                "energy": {k: c.energy[k] for k in ("KE_in", "E_elastic_start", "E_contact_start", "E_budget_ref",
                                                     "E_budget_gain", "E_budget_remaining")
                           if k in c.energy}})
            blind.append(S.with_dummy_targets(stripped))
        b1, b2 = batch_of(calls, st), batch_of(blind, st)
        o = _outputs(b1)
        d1, d2 = D.decode(o, b1), D.decode(o, b2)
        for k in ("chunk_v_end", "chunk_w_end", "chunk_dx", "chunk_drot", "imp_v_end",
                  "imp_dx", "bond_broke", "bond_damage_end", "energy_bound_active", "free"):
            assert torch.equal(d1[k], d2[k]), (kw, k)


def test_chunk_inertia_uses_the_body_rotation_not_the_rest_rotation():
    """The solver rotates ChunkData.inertia (structure body frame) by the body rotation only
    (R_body I R_bodyᵀ); with /4's full inertia and body quaternion the decode does the same,
    whatever the chunk's own (rest-rotated) orientation."""
    import numpy as np
    import torch

    from learned_impact.outcome import decode as D
    from learned_impact.outcome.batch import HierarchyCache, collate
    from learned_impact.outcome.geometry import quat_to_matrix, random_quat, sym6_to_matrix
    from learned_impact.outcome.synthetic import make_event

    m, calls, st = make_event(2)
    c = calls[0]
    nc = c.counts["chunks"]
    rng = np.random.default_rng(0)
    i6 = np.concatenate([rng.uniform(1, 2, (nc, 3)), rng.uniform(-0.2, 0.2, (nc, 3))], 1)
    body_q = random_quat(rng, nc)
    c.arrays = dict(c.arrays, chunk_inertia_struct=i6.astype(np.float32),
                    chunk_body_q=body_q.astype(np.float32),
                    chunk_q_world=random_quat(rng, nc).astype(np.float32))  # rest-rotated
    b = collate([c], {c.event_id: st}, HierarchyCache(2))
    iw = D.node_world_inertia(b)[:nc]
    r = quat_to_matrix(torch.tensor(body_q))
    want = r @ sym6_to_matrix(torch.tensor(i6)) @ r.transpose(-1, -2)
    assert torch.allclose(iw, want, atol=1e-5)


def test_energy_envelope_bounds_teleports_and_never_touches_admissible_moves():
    from learned_impact.outcome.batch import SCALE_INDEX

    calls, st = make_calls(range(6))
    for c in calls:
        c.energy["E_elastic_start"] = 0.0
    b = batch_of(calls, st)
    nc = b.t["chunk_node"].shape[0]
    torch.manual_seed(0)
    with torch.no_grad():
        out = ImpactOutcomeModel()(b)
        out["imp_out"][:, 6:9] += 500.0  # a teleport: far beyond any speed the energy allows
        on = D.decode(out, b, D.DecodeConfig(energy_bound=False))
        off = D.decode(out, b, D.DecodeConfig(energy_bound=False, energy_envelope=None))
    ph = on["_phys"]
    G = b.n_calls
    e = D.kinetic(ph, ph["v"], ph["w"], ph["inertia_w"], G) * (1 + D.ENERGY_ENVELOPE_TOL)
    dt = b.t["scales"].double()[:, SCALE_INDEX["dt"]]
    g = b.t["gravity"].double().norm(dim=-1)
    call = ph["call"]
    r = dt[call] * (torch.sqrt(2 * e[call] / ph["mass"]) + g[call] * dt[call])
    dx = torch.cat([on["chunk_dx"], on["imp_dx"]])
    free = ~ph["anchored"] & ~on["admissibility_unbounded"][call]
    assert (dx.norm(dim=-1)[free] <= r[free] * (1 + 1e-6)).all()
    assert int(on["admissibility_clamps"].sum()) > 0
    assert not torch.equal(on["imp_dx"], off["imp_dx"])
    # a decode already inside the envelope is unchanged
    with torch.no_grad():
        out0 = ImpactOutcomeModel()(b)
        a0 = D.decode(out0, b, D.DecodeConfig(energy_bound=False, energy_envelope=1e6))
        b0 = D.decode(out0, b, D.DecodeConfig(energy_bound=False, energy_envelope=None))
    assert int(a0["admissibility_clamps"].sum()) == 0
    torch.testing.assert_close(a0["imp_dx"], b0["imp_dx"])
    # without the frame-start stored energy the call is left unbounded (never tighter)
    for c in calls:
        c.energy.pop("E_elastic_start")
    b2 = batch_of(calls, st)
    with torch.no_grad():
        d2 = D.decode(ImpactOutcomeModel()(b2), b2, D.DecodeConfig(energy_bound=False))
    assert bool(d2["admissibility_unbounded"][:len(calls)].all())
    assert int(d2["admissibility_clamps"].sum()) == 0
