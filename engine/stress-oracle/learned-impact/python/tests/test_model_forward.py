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


def test_monotone_margin_never_below_the_frame_start_margin():
    import torch

    from learned_impact.outcome import schema as S
    from learned_impact.outcome.batch import HierarchyCache, collate
    from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
    from learned_impact.outcome.synthetic import make_event

    m, calls, st = make_event(4)
    b = collate(calls, {calls[0].event_id: st}, HierarchyCache(2))
    q0 = b.edges["bond"]["x"][0::2, S.col("bond_feat", "fail_margin")].clamp(min=1e-4)
    torch.manual_seed(0)
    with torch.no_grad():
        out = ImpactOutcomeModel(ModelConfig(monotone_margin=True))(b)
    assert (out["bond_log_q"] >= torch.log(q0) - 1e-6).all()


def test_full_state_encodings_round_trip():
    import numpy as np

    from learned_impact.outcome import fullstate as FS

    rng = np.random.default_rng(0)
    sc = {k: rng.uniform(0.5, 2.0, 5) for k in ("u_c", "theta_c", "gf_area_tension", "f_t")}
    for _f, how, _k in FS.BOND_FIELDS:
        x = rng.uniform(0.0, 3.0, 5) if how in ("log1p", "inv1p") else rng.normal(0, 2, 5)
        back = FS.decode_raw(FS._enc(x, how, sc, 1 / 60), how, sc, 1 / 60)
        assert np.allclose(back, x, rtol=1e-9, atol=1e-12), (_f, how)
    assert np.isinf(FS.decode_raw(FS._enc(np.array([np.inf]), "inv1p", sc, 1 / 60), "inv1p", sc, 1 / 60))[0]
