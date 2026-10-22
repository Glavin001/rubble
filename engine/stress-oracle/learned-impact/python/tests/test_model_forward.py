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


def test_drop_features_makes_the_column_invisible():
    import dataclasses

    import pytest
    import torch

    from learned_impact.outcome import schema as S
    from learned_impact.outcome.batch import HierarchyCache, collate
    from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
    from learned_impact.outcome.synthetic import make_event

    m, calls, st = make_event(5)
    for c in calls:
        c.event_id = m["event_id"]
    b = collate(calls, {m["event_id"]: st}, HierarchyCache(2))
    col = S.col("bond_feat", "log_qs_hertz_util")
    for variant in ("A", "B"):
        cfg = ModelConfig(variant=variant, drop_features=("bond_feat:log_qs_hertz_util",))
        torch.manual_seed(0)
        model = ImpactOutcomeModel(cfg).eval()
        with torch.no_grad():
            o1 = model(b)
            b.edges["bond"]["x"][:, col] += 7.0  # changing the dropped column changes nothing
            o2 = model(b)
            b.edges["bond"]["x"][:, col] -= 7.0
        for k in o1:
            assert torch.equal(o1[k], o2[k]), k
        # the config round-trips through a checkpoint dict
        assert ModelConfig(**cfg.to_dict()).drop_features == cfg.drop_features
    with pytest.raises(ValueError):
        ModelConfig(variant="A", drop_features=("bond_qs_feat:qs_frame_broke",))
    with pytest.raises(ValueError):
        dataclasses.replace(ModelConfig(), drop_features=("bond_feat:nope",))
