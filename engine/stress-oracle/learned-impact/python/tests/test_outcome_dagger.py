import json

import torch

from learned_impact.outcome.dagger import export
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
from learned_impact.outcome.synthetic import write_dataset
from learned_impact.outcome.train import Data, TrainConfig, save


def test_export_writes_jobs_for_diverged_states(tmp_path):
    write_dataset(tmp_path / "d", 6, val_fraction=0.01, settle_calls=2)
    data = Data(str(tmp_path / "d"))
    cfg = TrainConfig(model=ModelConfig.cost())
    torch.manual_seed(0)
    model = ImpactOutcomeModel(cfg.model)
    save(tmp_path / "m.pt", model, cfg, {}, None, 0, {})
    recs = {e.manifest["event_id"]: tmp_path / "rec" / e.manifest["event_id"] for e in data.events}
    s = export(tmp_path / "m.pt", data, recs, tmp_path / "round")
    jobs = json.loads((tmp_path / "round" / "jobs.json").read_text())
    assert s["calls_with_successor"] > 0 and len(jobs) == s["jobs"] > 0
    by = {(c.event_id, c.call_index): c for c in data.train}
    for j in jobs:
        parent = by[(j["parent_event_id"], j["parent_call"])]
        assert j["frame"] == parent.frame + 1 and j["split_group"] == parent.split_group
        doc = json.loads(open(j["state"]).read())
        assert doc["schema_version"].startswith("learned-impact-decoded-state/")
