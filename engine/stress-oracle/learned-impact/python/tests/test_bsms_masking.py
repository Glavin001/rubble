"""Coarse-edge masking across a crack, and the degree fix on isolated chunks."""

import numpy as np
import torch

from learned_impact.mgn.bsms import WeightedEdgeConv, degree
from learned_impact.mgn.bsms_graph import build_hierarchy, live_edges


def grid(nx, nz):
    pos = np.array([[i, 0, k] for k in range(nz) for i in range(nx)], np.float32)
    ends = []
    for k in range(nz):
        for i in range(nx):
            n = k * nx + i
            if i + 1 < nx:
                ends.append((n, n + 1))
            if k + 1 < nz:
                ends.append((n, n + nx))
    return pos, np.array(ends)


def _side_labels(h, side0):
    """Per level, which side of the crack each node is on (by its level-0 identity)."""
    out = [side0]
    ids = np.arange(len(side0))
    for lv in h.levels:
        ids = ids[lv.ids]
        out.append(side0[ids])
    return out


def test_no_level_carries_messages_across_a_crack():
    nx, nz = 12, 6
    pos, ends = grid(nx, nz)
    h = build_hierarchy(len(pos), ends, pos, num_levels=3)
    x = pos[:, 0]
    crack = (x[ends[:, 0]] <= 5) != (x[ends[:, 1]] <= 5)  # every bond across x = 5.5
    side0 = (x <= 5).astype(int)
    emask, nmask = live_edges(h, ~crack, np.ones(len(pos), bool))
    sides = _side_labels(h, side0)
    edges = [h.edges0] + [lv.edges for lv in h.levels]
    assert all(m.any() for m in emask), "every level keeps some live edges"
    for li, (e, m, s) in enumerate(zip(edges, emask, sides)):
        crossing = s[e[0]] != s[e[1]]
        assert not np.any(crossing & m), f"level {li} has a live edge across the crack"
        if li > 0:
            assert np.any(crossing), f"level {li} had crossing edges to mask"
    # without the crack, crossing edges are live
    emask2, _ = live_edges(h, np.ones(len(ends), bool), np.ones(len(pos), bool))
    assert any(np.any((s[e[0]] != s[e[1]]) & m) for e, m, s in zip(edges, emask2, sides))


def test_partial_crack_keeps_paths_around_it():
    pos, ends = grid(8, 4)
    h = build_hierarchy(len(pos), ends, pos, 2)
    cut = np.zeros(len(ends), bool)
    cut[0] = True  # one broken bond leaves the graph connected
    emask, _ = live_edges(h, ~cut, np.ones(len(pos), bool))
    assert emask[0].sum() == 2 * (len(ends) - 1)
    assert emask[1].sum() > 0.8 * h.levels[0].edges.shape[1]


def test_degree_fix_on_an_isolated_chunk():
    # the last node has no out-edges: the original `degree` sized its output by max(index)+1
    g = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    d = degree(g[0], num_nodes=4)
    assert d.shape == (4,) and d[3] == 0
    conv = WeightedEdgeConv()
    w = torch.ones(4, 1)
    ew, aggr = conv.cal_ew(w, g)
    assert torch.isfinite(ew).all() and torch.isfinite(aggr).all()
    x = torch.randn(4, 5)
    y = conv(x, g, ew)
    assert torch.isfinite(y).all() and torch.all(y[3] == 0)


def test_isolated_chunk_through_the_model(synthetic_batch):
    from learned_impact.outcome.model import ImpactOutcomeModel
    from tests.conftest import batch_of, make_calls

    calls, st = make_calls([7])
    c = calls[0]
    # break every bond of chunk 0: drop its live bonds from the call (twins stay paired)
    a = dict(c.arrays)
    keep_dir = (a["bond_src"] != 0) & (a["bond_dst"] != 0)
    keep_und = keep_dir[0::2]
    for k in ("bond_src", "bond_dst", "bond_feat", "bond_qs_feat"):
        a[k] = a[k][keep_dir]
    for k in ("bond_key", "bond_frame", *[n for n in a if n.startswith("y_bond_")],
              "base_qs_bond_broke", "base_qs_bond_peak_util"):
        a[k] = a[k][keep_und]
    c.arrays = a
    c.counts = dict(c.counts, bond=int(keep_und.sum()))
    b = batch_of([c], st)
    out = ImpactOutcomeModel()(b)
    assert all(torch.isfinite(v).all() for v in out.values())
