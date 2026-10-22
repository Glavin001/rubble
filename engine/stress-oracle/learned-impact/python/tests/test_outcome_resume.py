"""A run killed mid-training resumes from its periodic last.pt and finishes."""

import json

import pytest
import torch

from learned_impact.outcome import losses as Lo
from learned_impact.outcome.ensemble import aggregate_dataset
from learned_impact.outcome.synthetic import write_dataset
from learned_impact.outcome.train import Data, TrainConfig, train


def test_resume_after_a_kill(tmp_path, monkeypatch):
    write_dataset(tmp_path / "data", 8, ensembles=1, ensemble_size=3)
    aggregate_dataset(tmp_path / "data")
    data = Data(str(tmp_path / "data"))
    kw = dict(dataset=str(tmp_path / "data"), out=str(tmp_path / "run"), steps=12, batch_calls=4,
              eval_every=4, device="cpu", warmup=2, ckpt_every=4)
    real, n = Lo.total, {"k": 0}

    def dying(*a, **k):
        n["k"] += 1
        if n["k"] == 7 + 4:  # (the zero baselines and evals call loss_terms, not total)
            raise KeyboardInterrupt("instance killed")
        return real(*a, **k)
    monkeypatch.setattr(Lo, "total", dying)
    with pytest.raises(KeyboardInterrupt):
        train(TrainConfig(**kw), data)
    monkeypatch.setattr(Lo, "total", real)
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert "resume" in ck and ck["resume"]["step"] in (4, 8)
    r = train(TrainConfig(**kw, resume=True), Data(str(tmp_path / "data")))
    st = json.loads((tmp_path / "run" / "status.json").read_text())
    assert st["state"] == "done" and st["step"] == 12 and r["best"]["step"] is not None
    assert "resumed from" in (tmp_path / "run" / "progress.log").read_text()
    assert "resume" not in torch.load(tmp_path / "run" / "last.pt", weights_only=False)  # finished
