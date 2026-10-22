"""W&B tracking: offline runs from the trainer and from the backfill (never online here)."""

import json

import pytest

from learned_impact.outcome import tracking as TR


def test_flatten_translates_every_history_kind():
    s, m = TR.flatten({"kind": "train_running", "step": 50, "epoch": 1, "loss": 2.0,
                       "terms": {"margin": 0.5}, "lr": 1e-3, "grad_norm": 3.0,
                       "gshare": {"decision": 0.3}, "mult": {"decision": 1.2}, "steps_per_s": 2.0})
    assert s == 50 and m["train/margin"] == 0.5 and m["gshare/decision"] == 0.3
    assert m["mult/decision"] == 1.2 and m["steps_per_s"] == 2.0 and m["train/loss"] == 2.0
    s, m = TR.flatten({"kind": "eval", "step": 100, "score": 0.6, "metrics": {
        "bond_f1": 0.9, "families": {"anchored_wall": {"skill": 0.4}}, "admissibility_clamp_rate": 0.0},
        "train": {"loss": 1.0, "terms": {"pose": 0.2}}})
    assert m["val/score"] == 0.6 and m["val/families/anchored_wall/skill"] == 0.4
    assert m["curve/train/terms/pose"] == 0.2 and m["val/admissibility_clamp_rate"] == 0.0
    s, m = TR.flatten({"kind": "group_shares", "step": 5, "shares": {"motion": 0.3}, "mult": {"motion": 2.0}})
    assert m == {"gshare_window/motion": 0.3, "mult/motion": 2.0}


def test_resolve_mode():
    assert TR.resolve_mode("off") == "off"
    with pytest.raises(ValueError):
        TR.resolve_mode("cloud")


@pytest.mark.skipif(not TR.available(), reason="wandb not installed")
def test_trainer_and_backfill_write_offline_runs(tmp_path, monkeypatch):
    from learned_impact.outcome.ensemble import aggregate_dataset
    from learned_impact.outcome.synthetic import write_dataset
    from learned_impact.outcome.train import Data, TrainConfig, train
    from learned_impact.wandb_backfill import backfill, find_runs

    monkeypatch.setenv("WANDB_SILENT", "true")
    write_dataset(tmp_path / "data", 6, ensembles=1, ensemble_size=3)
    aggregate_dataset(tmp_path / "data")
    run = tmp_path / "v0" / "w-s0"
    cfg = TrainConfig(dataset=str(tmp_path / "data"), out=str(run), steps=8, batch_calls=4,
                      eval_every=4, device="cpu", warmup=2, wandb="offline")
    train(cfg, Data(str(tmp_path / "data")))
    marker = json.loads((run / "wandb-run.json").read_text())
    assert marker["mode"] == "offline" and list((run / "wandb").glob("offline-run-*"))
    hist = [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines()]
    tr = [h for h in hist if h["kind"] == "train_running"]
    assert tr and "gshare" in tr[0] and "steps_per_s" in tr[0] and "grad_norm" in tr[0]
    # a live-tracked run is skipped by the backfill; an untracked one becomes a run
    assert find_runs([tmp_path / "v0"]) == [run]
    assert backfill(run) is None
    (run / "wandb-run.json").unlink()
    rid = backfill(run)
    assert rid and json.loads((run / "wandb-backfill.json").read_text())["id"] == rid
    assert backfill(run) is None
