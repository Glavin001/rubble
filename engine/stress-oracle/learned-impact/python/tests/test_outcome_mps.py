"""The model runs on MPS (no PyG / torch_scatter) and agrees with CPU; f16 stays finite."""

import pytest
import torch

from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from tests.conftest import batch_of, make_calls

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")


@mps
@pytest.mark.parametrize("cfg", [ModelConfig(), ModelConfig.reference(), ModelConfig(variant="B")])
def test_mps_matches_cpu(cfg):
    calls, st = make_calls(range(3))
    b = batch_of(calls, st)
    torch.manual_seed(0)
    m = ImpactOutcomeModel(cfg)
    for head in (m.chunk_head, m.bond_head):
        torch.nn.init.normal_(head.model[-1].weight, std=0.3)
    with torch.no_grad():
        oc = m(b)
        om = m.to("mps")(b.to("mps"))
    for k in oc:
        torch.testing.assert_close(om[k].cpu(), oc[k], rtol=2e-3, atol=2e-4)


@mps
def test_f16_forward_is_finite():
    calls, st = make_calls(range(3))
    b = batch_of(calls, st).to("mps")
    for d in [b.t, *b.edges.values()]:
        for k, v in d.items():
            if v.dtype == torch.float32 and k not in ("scales", "energy", "gravity"):
                d[k] = v.half()
    m = ImpactOutcomeModel(ModelConfig()).to("mps").half()
    with torch.no_grad():
        out = m(b)
    assert all(torch.isfinite(v).all() for v in out.values())
