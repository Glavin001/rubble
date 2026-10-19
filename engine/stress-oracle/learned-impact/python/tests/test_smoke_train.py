from pathlib import Path

from learned_impact.train import run_smoke


def test_smoke_train(tmp_path: Path):
    loss = run_smoke(None, steps=3, h=8, p=4, out=tmp_path / "w")
    assert loss == loss  # finite
    assert (tmp_path / "w" / "weights.bin").exists()
    assert (tmp_path / "w" / "weights.json").exists()
