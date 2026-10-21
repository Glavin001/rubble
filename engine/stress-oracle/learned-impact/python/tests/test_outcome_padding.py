"""Shape bucketing is exact: padded batches give the same losses, real outputs and metrics."""

import copy

import pytest
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome import losses as Lo
from learned_impact.outcome import metrics as M
from learned_impact.outcome.batch import bucket, pad_batch
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from tests.conftest import batch_of, make_calls


def test_bucket():
    assert bucket(1) == 64 and bucket(64) == 64 and bucket(65) == 80
    sizes = {bucket(n) for n in range(1, 20000)}
    assert len(sizes) <= 27  # 64 … 20k in 27 shapes at growth 1.25


@pytest.mark.parametrize("kw", [dict(), dict(anchored=True), dict(gravity=True, anchored=False),
                                dict(settle_calls=1)])
def test_padding_is_exact(kw):
    calls, st = make_calls(range(5), **kw)
    b0 = batch_of(calls, st)
    b1 = pad_batch(copy.deepcopy(b0))
    assert int(b1.t["n_nodes"]) > int(b0.t["n_nodes"]) and b1.n_calls % 8 == 0
    torch.manual_seed(0)
    m = ImpactOutcomeModel(ModelConfig()).double()
    for head in (m.chunk_head, m.bond_head, m.imp_head):
        torch.nn.init.normal_(head.model[-1].weight, std=0.3)
    for bb in (b0, b1):
        for d in [bb.t, *bb.edges.values()]:
            for k, v in d.items():
                if v.dtype == torch.float32:
                    d[k] = v.double()
    cfg = Lo.LossConfig()
    with torch.no_grad():
        o0, o1 = m(b0), m(b1)
        nc, ni, nb = (b0.t["chunk_node"].shape[0], b0.t["imp_node"].shape[0],
                      b0.t["bond_call"].shape[0])
        torch.testing.assert_close(o1["chunk_out"][:nc], o0["chunk_out"], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(o1["imp_out"][:ni], o0["imp_out"], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(o1["bond_log_q"][:nb], o0["bond_log_q"], rtol=1e-9, atol=1e-10)
        t0, t1 = Lo.loss_terms(o0, b0, cfg), Lo.loss_terms(o1, b1, cfg)
        for k in t0:
            torch.testing.assert_close(t1[k], t0[k], rtol=1e-5, atol=1e-7, msg=k)
        d0, d1 = D.decode(o0, b0), D.decode(o1, b1)
        torch.testing.assert_close(d1["chunk_v_end"][:nc], d0["chunk_v_end"], rtol=1e-9, atol=1e-9)
    e0 = M.evaluate(m, [b0])
    e1 = M.evaluate(m, [b1])
    for k, v in e0.metrics.items():
        assert e1.metrics[k] == pytest.approx(v, rel=1e-6, abs=1e-9), k
