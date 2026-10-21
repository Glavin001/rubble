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
    assert ck["schema_version"] == "learned-impact-outcome/1"


def test_batches_pack_to_a_budget_without_dropping():
    from learned_impact.outcome.train import _batches, call_size, take_batch

    calls, _ = make_calls(range(10))
    sizes = [call_size(c) for c in calls]
    bs = _batches(calls, budget=3 * max(sizes), max_calls=64)
    assert sum(len(b) for b in bs) == len(calls)
    assert all(sum(call_size(c) for c in b) <= 3 * max(sizes) for b in bs)
    one, rest = take_batch(calls, budget=1, max_calls=64)  # oversized call: alone, not dropped
    assert len(one) == 1 and len(rest) == len(calls) - 1


def test_trainer_chained_topology_pushforward(tmp_path):
    from learned_impact.outcome import schema as S
    from learned_impact.outcome.synthetic import make_event

    for i in range(6):
        m, cs, st = make_event(40 + i, settle_calls=2, split="val" if i == 0 else "train")
        S.write_event(tmp_path / "data" / m["event_id"], m, cs, st)
    cfg = TrainConfig(dataset=str(tmp_path / "data"), out=str(tmp_path / "run"), steps=6,
                      batch_calls=4, eval_every=6, device="cpu", pushforward_prob=1.0,
                      pushforward_steps=3, warmup=1, loss=Lo.LossConfig(balance="uncertainty"))
    train(cfg)
    log = (tmp_path / "run" / "progress.log").read_text()
    depths = [float(x.split("pushforward_depth=")[1].split()[0]) for x in log.splitlines()
              if "pushforward_depth=" in x]
    assert depths and max(depths) > 1.0, "chains should go deeper than one frame"
    assert "λeff" in log
