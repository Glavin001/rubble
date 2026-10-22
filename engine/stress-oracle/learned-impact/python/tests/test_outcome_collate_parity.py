"""The cached collate is bit-identical to the reference (per-batch) collate, and so are the
losses; within fp32 tolerance on MPS."""

import numpy as np
import pytest
import torch

from learned_impact.outcome import losses as Lo
from learned_impact.outcome.batch import HierarchyCache, clear_piece_cache, collate, pad_batch
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from learned_impact.outcome.synthetic import make_event
from tests.collate_reference import collate_reference


def _calls(n, full_state=False):
    calls, st = [], {}
    for i in range(n):
        m, cs, s = make_event(500 + i, split="train", full_state=full_state) if full_state else make_event(500 + i, split="train")
        st[m["event_id"]] = s
        calls += cs
    return calls, st


def _equal(b1, b2):
    assert set(b1.t) == set(b2.t)
    for k in b1.t:
        assert torch.equal(b1.t[k], b2.t[k]) or (b1.t[k].dtype.is_floating_point and torch.equal(
            torch.nan_to_num(b1.t[k], nan=123.0), torch.nan_to_num(b2.t[k], nan=123.0))), k
    assert set(b1.edges) == set(b2.edges)
    for t in b1.edges:
        for k in b1.edges[t]:
            assert torch.equal(b1.edges[t][k], b2.edges[t][k]), (t, k)
    for name in ("h_edges", "h_ids", "h_chunk"):
        for x, y in zip(getattr(b1, name), getattr(b2, name)):
            assert torch.equal(x, y), name
    assert b1.n_fragments == b2.n_fragments


@pytest.mark.parametrize("full_state", [False, True])
def test_cached_collate_is_bit_identical(full_state):
    try:
        calls, st = _calls(10, full_state)
    except TypeError:
        pytest.skip("synthetic events have no full-state option")
    clear_piece_cache()
    rng = np.random.default_rng(0)
    for _ in range(4):  # different groupings reuse the cached pieces
        idx = rng.permutation(len(calls))[: rng.integers(3, len(calls))]
        bc = [calls[i] for i in idx]
        b_new = collate(bc, st, HierarchyCache(2))
        b_ref = collate_reference(bc, st, HierarchyCache(2))
        _equal(b_new, b_ref)
        _equal(pad_batch(b_new), pad_batch(b_ref))
        torch.manual_seed(0)
        m = ImpactOutcomeModel(ModelConfig())
        cfg = Lo.LossConfig()
        with torch.no_grad():
            t1 = Lo.loss_terms(m(b_new), b_new, cfg)
            t2 = Lo.loss_terms(m(b_ref), b_ref, cfg)
        for k in t1:
            assert torch.equal(t1[k], t2[k]) or (torch.isnan(t1[k]) and torch.isnan(t2[k])), k


def test_a_call_with_replaced_arrays_recomputes_its_piece():
    calls, st = _calls(3)
    c = calls[0]
    b1 = collate([c], st, HierarchyCache(2))
    c.arrays = dict(c.arrays)
    c.arrays["y_bond_break_time"] = np.full(c.counts["bond"], 0.004)
    b2 = collate([c], st, HierarchyCache(2))
    _equal(b2, collate_reference([c], st, HierarchyCache(2)))
    assert not torch.equal(torch.nan_to_num(b1.t["y_bond_break_frac"]), b2.t["y_bond_break_frac"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_losses_match_cpu_within_fp32():
    calls, st = _calls(6)
    b = pad_batch(collate(calls, st, HierarchyCache(2)))
    torch.manual_seed(0)
    m = ImpactOutcomeModel(ModelConfig())
    cfg = Lo.LossConfig()
    with torch.no_grad():
        tc = Lo.loss_terms(m(b), b, cfg)
        mm = m.to("mps")
        bm = b.to("mps")
        tm = Lo.loss_terms(mm(bm), bm, cfg)
    for k in tc:
        a, x = float(tc[k]), float(tm[k].cpu())
        assert abs(a - x) <= 1e-4 * max(1.0, abs(a)), (k, a, x)
