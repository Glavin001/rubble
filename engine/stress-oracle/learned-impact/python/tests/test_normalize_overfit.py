from pathlib import Path

import torch

from learned_impact.dataset import synthetic_batch
from learned_impact.normalize import normalize_batch
from learned_impact.train import run_overfit


def test_normalize_scales_joint_and_loads():
    b = synthetic_batch()
    b["bond_joint_state"][:, 4] = 1.0e6
    b["chunk_load_bins"][:] = 2.0e4
    n = normalize_batch(b)
    assert float(n["bond_joint_state"].abs().max()) < 100
    assert float(n["chunk_load_bins"].abs().max()) < 10


def test_overfit_drops_on_synthetic(tmp_path: Path, monkeypatch):
    # Build a one-event synthetic set on disk is heavy; instead monkeypatch loader.
    from learned_impact import train as T

    def fake_batches(dataset, structure_type):
        return [synthetic_batch(n=8, e=12, seed=1)]

    monkeypatch.setattr(T, "_batches_for", fake_batches)
    first, last = run_overfit(tmp_path, steps=40, h=16, p=8, out=tmp_path / "w", arch="mgn")
    assert last < first * 0.5 or last < 1.0
