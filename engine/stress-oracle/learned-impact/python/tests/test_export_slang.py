"""Slang weight pack matches kernel layout and weights.json sizes."""

from pathlib import Path

import torch

from learned_impact.export import (
    SLANG_FORMAT,
    export_slang_weights,
    pack_slang_state_dict,
    slang_weight_count,
)
from learned_impact.model import LearnedImpactNet


def test_pack_slang_size_matches_json(tmp_path: Path):
    torch.manual_seed(0)
    h, p = 48, 16
    model = LearnedImpactNet(h=h, p=p)
    blob, cfg = pack_slang_state_dict(model.state_dict(), h=h, p=p)
    assert cfg["format"] == SLANG_FORMAT
    assert cfg["n_floats"] == slang_weight_count(h, p)
    assert cfg["n_bytes"] == len(blob)
    assert cfg["n_bytes"] == sum(
        __import__("math").prod(t["shape"]) * 4 for t in cfg["tensors"]
    )
    path = export_slang_weights(model, tmp_path, h=h, p=p)
    assert path.read_bytes() == blob
    assert (tmp_path / "weights.json").exists()


def test_pack_smoke_dims():
    torch.manual_seed(1)
    h, p = 8, 4
    model = LearnedImpactNet(h=h, p=p)
    blob, cfg = pack_slang_state_dict(model.state_dict(), h=h, p=p)
    assert len(blob) // 4 == slang_weight_count(h, p) == cfg["n_floats"]
