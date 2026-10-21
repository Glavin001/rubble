import json

import numpy as np

from learned_impact.outcome import pilot
from learned_impact.outcome import schema as S
from learned_impact.outcome.ensemble import aggregate_dataset
from learned_impact.outcome.model import ModelConfig
from learned_impact.outcome.synthetic import make_event


def test_power_law_fit_and_projection():
    n = np.array([250, 1000, 4000, 16000])
    e = 3.0 * n ** -0.5 + 0.2
    f = pilot.fit_power_law(n, e)
    assert abs(f["b"] - 0.5) < 0.05 and abs(f["c"] - 0.2) < 0.02
    p = pilot.project(f, 0.3)
    assert abs(p["n_needed"] - (3.0 / 0.1) ** 2) / 900 < 0.1
    assert pilot.project(f, 0.1)["n_needed"] == float("inf")  # asymptote above the bar


def test_nested_subsets():
    g = [f"g{i}" for i in range(50)]
    s = pilot.nested_subsets(g, [5, 20, 100], seed=1)
    assert s[5] <= s[20] <= s[100] and len(s[100]) == 50


def test_pilot_end_to_end(tmp_path):
    data = tmp_path / "data"
    for i in range(12):
        m, cs, st = make_event(100 + i, split="train")
        S.write_event(data / m["event_id"], m, cs, st)
    for k in range(4):  # one validation ensemble
        m, cs, st = make_event(900, split="val", ensemble=("ens_900", k, 4))
        S.write_event(data / m["event_id"], m, cs, st)
    aggregate_dataset(data)
    s = pilot.run(str(data), str(tmp_path / "out"), [4, 12], [0], ["A", "B"], None,
                  steps_per_event=1.0, min_steps=4, max_steps=8, device="cpu",
                  model_cfg=ModelConfig())
    assert s["ensembles"] and set(s["variants"]) == {"A", "B"}
    md = (tmp_path / "out" / "pilot.md").read_text()
    assert "skill_over_bar_ram" in md and "Variant B" in md
    # resumable: a second run reuses every result
    s2 = pilot.run(str(data), str(tmp_path / "out"), [4, 12], [0], ["A", "B"], None,
                   1.0, 4, 8, "cpu", ModelConfig())
    assert json.dumps(s2, default=float) == json.dumps(s, default=float)
