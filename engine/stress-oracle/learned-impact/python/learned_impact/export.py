"""Flatten a state_dict to a binary blob + JSON config for future Slang load."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def export_weights(model: torch.nn.Module, out_dir: Path | str, *, h: int, p: int) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    order: list[dict] = []
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        order.append({"name": name, "shape": list(arr.shape), "dtype": "f32", "offset": len(blob)})
        blob.extend(arr.tobytes(order="C"))
    bin_path = out / "weights.bin"
    bin_path.write_bytes(blob)
    cfg = {
        "format": "learned-impact-weights/1",
        "h": h,
        "p": p,
        "n_bytes": len(blob),
        "tensors": order,
    }
    (out / "weights.json").write_text(json.dumps(cfg, indent=2))
    return bin_path
