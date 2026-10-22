from learned_impact.outcome.ensemble import aggregate_dataset
from learned_impact.outcome.synthetic import write_dataset
from learned_impact.outcome.train import Data, TrainConfig, train
from learned_impact.outcome.viability import build, overfit_summary, read_history


def test_viability_report_from_a_finished_run(tmp_path):
    write_dataset(tmp_path / "data", 10, ensembles=1, ensemble_size=3)
    aggregate_dataset(tmp_path / "data")
    run = tmp_path / "run"
    cfg = TrainConfig(dataset=str(tmp_path / "data"), out=str(run), steps=8, batch_calls=4,
                      eval_every=4, device="cpu", warmup=2, select_tau=False)
    cfg.model = cfg.model.cost()
    train(cfg, Data(str(tmp_path / "data")))
    ov = overfit_summary(read_history(run))
    assert "loss" in ov and "motion" in ov and ov["loss"]["val_min_step"] in (4, 8)
    out = build({"toy": run}, None, tmp_path / "viability.md")
    text = out.read_text()
    assert "## 1. Loss curves" in text and "toy" in text and "## 4. Closed loop" in text
    assert (tmp_path / "viability" / "curves-toy.png").stat().st_size > 1000
