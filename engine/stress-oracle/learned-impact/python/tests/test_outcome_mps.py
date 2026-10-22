"""The model runs on MPS (no PyG / torch_scatter) and agrees with CPU; f16 stays finite."""

import pytest
import torch

from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from tests.conftest import batch_of, make_calls

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")


@mps
@pytest.mark.parametrize("cfg", [ModelConfig.cost(), ModelConfig.reference(), ModelConfig.large()])
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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_step1_loss_parity_cpu_mps(tmp_path):
    """The device-parity check remote training relies on (devcheck): the same seeded model
    on the same batch gives the same per-term loss on CPU and MPS (CUDA is checked on Modal
    against the same reference)."""
    from learned_impact.outcome.devcheck import compare, step1_losses
    from learned_impact.outcome.synthetic import write_dataset

    write_dataset(tmp_path / "d", 6)
    cpu = step1_losses(str(tmp_path / "d"), "cpu", n_calls=8)
    mps = step1_losses(str(tmp_path / "d"), "mps", n_calls=8)
    assert compare(cpu, mps) == [], compare(cpu, mps)
