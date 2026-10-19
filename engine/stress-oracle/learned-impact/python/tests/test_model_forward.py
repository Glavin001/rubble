import torch

from learned_impact.dataset import synthetic_batch
from learned_impact.model import LearnedImpactNet


def test_forward_shapes():
    batch = synthetic_batch(n=8, e=12)
    model = LearnedImpactNet(h=8, p=4)
    model.attn.heads = 2
    pred = model(batch)
    e = batch["bond_peak_util"].shape[0]
    n = batch["chunk_state"].shape[0]
    assert pred["bond_fail_logits"].shape == (e, 5)
    assert pred["bond_peak_util"].shape == (e,)
    assert pred["chunk_state_delta"].shape == (n, 12)
    assert torch.isfinite(pred["bond_peak_util"]).all()
