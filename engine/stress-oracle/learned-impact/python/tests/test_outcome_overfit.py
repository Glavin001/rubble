"""The model + trainer can overfit a small synthetic dataset that matches the schema.
This says nothing about accuracy on oracle data."""

import pytest
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome import losses as Lo
from learned_impact.outcome import metrics as M
from learned_impact.outcome.ensemble import aggregate_dataset
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from learned_impact.outcome.synthetic import write_dataset
from learned_impact.outcome.train import Data, TrainConfig, train
from tests.conftest import DEVICES, batch_of, make_calls


def test_overfit_four_events_cpu():
    torch.manual_seed(0)
    calls, st = make_calls([1, 3, 4, 8])
    calls = [c for c in calls if c.call_index == 0]
    b = batch_of(calls, st)
    assert (b.t["y_bond_broke"] > 0).any()
    m = ImpactOutcomeModel(ModelConfig(width=32))
    cfg = Lo.LossConfig()
    norms = Lo.zero_baselines([b], cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss, normed = Lo.total(Lo.loss_terms(m(b), b, cfg), norms, cfg)
        first = first if first is not None else {k: float(v) for k, v in normed.items()}
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    last = {k: float(v) for k, v in normed.items()}
    for k in ("demand", "motion", "impactor", "time"):
        assert last[k] < 0.2 * first[k], (k, first[k], last[k])
    with torch.no_grad():
        dec = D.decode(m(b), b)
    pred = dec["bond_broke"].numpy()
    truth = (b.t["y_bond_broke"] > 0).numpy()
    assert M.prf(pred, truth)[2] > 0.95


@pytest.mark.parametrize("device", DEVICES)
def test_trainer_end_to_end(tmp_path, device):
    write_dataset(tmp_path / "data", 10, ensembles=1, ensemble_size=3)
    aggregate_dataset(tmp_path / "data")
    data = Data(str(tmp_path / "data"))
    cfg = TrainConfig(dataset=str(tmp_path / "data"), out=str(tmp_path / "run"), steps=12,
                      batch_calls=4, eval_every=6, device=device, pushforward_prob=1.0,
                      noise_sigma=0.01, warmup=2)
    r = train(cfg, data)
    assert r["best"]["step"] is not None
    run = tmp_path / "run"
    assert (run / "progress.log").read_text().count("[train") > 3
    import json
    st = json.loads((run / "status.json").read_text())
    assert st["state"] == "done" and st["step"] == 12
    assert (run / "best.pt").exists() and (run / "last.pt").exists()
    from learned_impact.outcome.train import load_model
    model, ck = load_model(run / "last.pt")
    assert ck["schema_version"] == "learned-impact-outcome-calls/1"
