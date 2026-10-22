import json

import numpy as np
import torch

from learned_impact.outcome.export import export
from learned_impact.outcome.model import ImpactOutcomeModel
from learned_impact.outcome.train import TrainConfig, save


def test_weights_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = ImpactOutcomeModel()
    save(tmp_path / "m.pt", m, TrainConfig(), {}, {}, 0, {}, tau=0.7)
    out = export(tmp_path / "m.pt", tmp_path / "exp", None, 0)
    man = json.loads((out / "weights.json").read_text())
    blob = (out / "weights.bin").read_bytes()
    sd = m.state_dict()
    assert man["decode"]["tau"] == 0.7 and len(man["tensors"]) == len(sd)
    for t in man["tensors"]:
        a = np.frombuffer(blob, "<f4", t["count"], t["offset"]).reshape(t["shape"])
        np.testing.assert_array_equal(a, sd[t["name"]].numpy())
        assert t["offset"] % 16 == 0
