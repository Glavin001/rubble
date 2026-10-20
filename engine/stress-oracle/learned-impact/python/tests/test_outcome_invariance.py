"""Permutation, rotation/translation and batching invariance of model + decode, and the
plan's reach witness (a load at one end of a 28-chunk wall changes the other end)."""

import numpy as np
import pytest
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome import schema as S
from learned_impact.outcome.geometry import quat_mul_np, quat_to_matrix_np, random_quat
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from tests.conftest import batch_of, make_calls

WORLD_VEC = ("chunk_pos", "chunk_v", "chunk_w", "impactor_pos", "impactor_v", "impactor_w",
             "y_chunk_v_end", "y_chunk_w_end", "y_chunk_dx", "y_chunk_drot", "y_impactor_v_end",
             "y_impactor_w_end", "y_impactor_dx", "y_impactor_drot")


def rotate_call(c: S.Call, q: np.ndarray) -> S.Call:
    r = quat_to_matrix_np(q)
    a = {k: np.array(v) for k, v in c.arrays.items()}
    for k in WORLD_VEC:
        a[k] = (a[k].astype(np.float64) @ r.T).astype(np.float32)
    for k in ("chunk_q_world", "impactor_q_world"):
        a[k] = quat_mul_np(q[None], a[k].astype(np.float64)).astype(np.float32)
    for t in S.EDGE_TYPES:
        a[f"{t}_frame"] = (a[f"{t}_frame"].astype(np.float64) @ r.T).astype(np.float32)
    out = S.Call(**{**c.__dict__, "arrays": a})
    out.gravity = tuple((r @ np.asarray(c.gravity)).tolist())
    return out


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = ImpactOutcomeModel(ModelConfig(**kw)).double()
    m.eval()
    # larger head outputs than the near-zero init, so the test is not trivially passing
    for head in (m.chunk_head, m.imp_head):
        torch.nn.init.normal_(head.model[-1].weight, std=0.3)
    return m


def _double(b):
    for k, v in b.t.items():
        if v.dtype == torch.float32:
            b.t[k] = v.double()
    for d in b.edges.values():
        for k, v in d.items():
            if v.dtype == torch.float32:
                d[k] = v.double()
    return b


def test_rotation_translation_equivariance():
    calls, st = make_calls([11], gravity=True)
    m = _model()
    rng = np.random.default_rng(5)
    q = random_quat(rng, 1)[0]
    r = quat_to_matrix_np(q)
    st_rot = {eid: {sid: dict(s, chunk_rest_pos=(s["chunk_rest_pos"].astype(np.float64) @ r.T
                                                  + np.array([7.0, -3.0, 100.0])).astype(np.float32))
                    for sid, s in v.items()} for eid, v in st.items()}
    b0 = _double(batch_of(calls, st))
    b1 = _double(batch_of([rotate_call(c, q) for c in calls], st_rot))
    with torch.no_grad():
        o0, o1 = m(b0), m(b1)
        for k in o0:  # inputs are rotated in f32 storage: compare at that rounding
            torch.testing.assert_close(o0[k], o1[k], rtol=1e-5, atol=1e-6)
        d0, d1 = D.decode(o0, b0), D.decode(o1, b1)
    rt = torch.from_numpy(r)
    for k in ("chunk_v_end", "chunk_w_end", "chunk_dx", "chunk_drot", "imp_v_end", "imp_dx"):
        torch.testing.assert_close(d0[k] @ rt.T, d1[k], rtol=1e-5, atol=2e-5)
    assert torch.equal(d0["bond_broke"], d1["bond_broke"])


def _permute_call(c: S.Call, perm: np.ndarray) -> S.Call:
    """Relabel chunks (new index = inv[old]) and reverse the order of undirected bonds."""
    a = {k: np.array(v) for k, v in c.arrays.items()}
    nc = c.counts["chunks"]
    inv = np.empty(nc, np.int64)
    inv[perm] = np.arange(nc)
    for k in list(a):
        if k.startswith("chunk_") or k.startswith("y_chunk_") or k.startswith("base_qs_chunk"):
            a[k] = a[k][perm]
    remap = np.concatenate([inv, np.arange(nc, nc + c.counts["impactors"])])
    for t in S.EDGE_TYPES:
        a[f"{t}_src"] = remap[a[f"{t}_src"]].astype(np.int32)
        a[f"{t}_dst"] = remap[a[f"{t}_dst"]].astype(np.int32)
    e = c.counts["bond"]
    und = np.arange(e)[::-1]
    dirs = np.stack([2 * und, 2 * und + 1], 1).reshape(-1)
    for k in ("bond_src", "bond_dst", "bond_feat", "bond_qs_feat"):
        a[k] = a[k][dirs]
    for k in [n for n in a if (n.startswith("y_bond_") or n.startswith("base_qs_bond"))] + [
            "bond_key", "bond_frame"]:
        a[k] = a[k][und]
    return S.Call(**{**c.__dict__, "arrays": a})


def test_permutation_equivariance():
    calls, st = make_calls([12])
    c = calls[0]
    perm = np.random.default_rng(1).permutation(c.counts["chunks"])
    m = _model()
    b0 = _double(batch_of([c], st))
    b1 = _double(batch_of([_permute_call(c, perm)], st))
    with torch.no_grad():
        o0, o1 = m(b0), m(b1)
    torch.testing.assert_close(o0["chunk_out"][perm], o1["chunk_out"], rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(o0["imp_out"], o1["imp_out"], rtol=1e-7, atol=1e-8)
    e = c.counts["bond"]
    torch.testing.assert_close(o0["bond_log_r"].flip(0), o1["bond_log_r"], rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(o0["energy_ke_loss"], o1["energy_ke_loss"], rtol=1e-7, atol=1e-8)
    assert e > 0


def test_batching_independence_of_islands():
    calls, st = make_calls([20, 21, 22])
    m = _model()
    with torch.no_grad():
        ob = m(_double(batch_of(calls, st)))
        b_all = _double(batch_of(calls, st))
        offs_c = offs_b = 0
        for c in calls:
            o1 = m(_double(batch_of([c], st)))
            nc, nb = c.counts["chunks"], c.counts["bond"]
            torch.testing.assert_close(ob["chunk_out"][offs_c:offs_c + nc], o1["chunk_out"],
                                       rtol=1e-9, atol=1e-10)
            torch.testing.assert_close(ob["bond_log_r"][offs_b:offs_b + nb], o1["bond_log_r"],
                                       rtol=1e-9, atol=1e-10)
            offs_c += nc
            offs_b += nb
        dec = D.decode(ob, b_all)
        assert dec["free"].shape[0] == len(calls)


@pytest.mark.parametrize("cfg", [dict(use_global=True),
                                 dict(use_global=False, levels=3, rounds=(2, 1, 1, 2, 1, 1, 2))])
def test_reach_witness_28_chunk_wall(cfg):
    calls, st = make_calls([30], nx=28, nz=1, anchored=False)
    c = calls[0]
    m = _model(**cfg)
    b0 = _double(batch_of([c], st, levels=m.cfg.levels))
    a = {k: np.array(v) for k, v in c.arrays.items()}
    a["chunk_feat"][0, S.col("chunk_feat", "log_mass")] += 1.0  # a change at one end
    b1 = _double(batch_of([S.Call(**{**c.__dict__, "arrays": a})], st, levels=m.cfg.levels))
    with torch.no_grad():
        far0 = m(b0)["chunk_out"][27]
        far1 = m(b1)["chunk_out"][27]
    assert (far0 - far1).abs().max() > 1e-9, "the far end did not see the change"
