"""Load and validate `learned-impact/1` event directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CHUNK_STATE_DIM, JOINT_STATE_DIM, K, LOCAL6_DIM, SCHEMA_VERSION

STATIC_REQUIRED = [
    "chunk_mass",
    "chunk_inertia",
    "chunk_half_extents",
    "chunk_material_id",
    "chunk_support",
    "chunk_level",
    "chunk_coarse_parent",
    "bond_area",
    "bond_width",
    "bond_offset",
    "bond_stiffness",
    "bond_strength",
    "bond_fracture_energy",
    "bond_rebar",
    "bond_weibull",
    "bond_buckling_length",
    "bond_endpoints",
]

FRAME_REQUIRED = [
    "chunk_state",
    "chunk_load_bins",
    "chunk_contact",
    "chunk_gravity_local",
    "bond_joint_state",
    "bond_force_prev",
    "bond_qs_force",
    "bond_fail_class",
    "bond_peak_util",
    "bond_fail_mode",
    "bond_joint_state_end",
    "bond_force_end",
    "chunk_state_delta",
]


def _read_array(root: Path, entry: dict[str, Any]) -> np.ndarray:
    dtype = np.float32 if entry["dtype"] == "f32" else np.int32
    shape = tuple(entry["shape"])
    path = root / entry["path"]
    data = np.fromfile(path, dtype=dtype)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: size {data.size} != shape {shape}")
    return data.reshape(shape)


@dataclass
class EventArrays:
    manifest: dict[str, Any]
    static: dict[str, np.ndarray]
    frames: list[dict[str, np.ndarray]]


def load_event(path: Path | str) -> EventArrays:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version {manifest.get('schema_version')!r}")
    if int(manifest["K"]) != K:
        raise ValueError(f"K={manifest['K']} != {K}")
    n = int(manifest["n_chunks"])
    e = int(manifest["n_bonds"])
    static_cat = json.loads((root / "catalog.json").read_text())["arrays"]
    static = {name: _read_array(root, static_cat[name]) for name in STATIC_REQUIRED}
    if static["bond_endpoints"].shape != (e, 2):
        raise ValueError("bond_endpoints shape")
    frames: list[dict[str, np.ndarray]] = []
    for fi in range(int(manifest["n_frames"])):
        fdir = root / "frames" / f"{fi:06d}"
        cat = json.loads((fdir / "catalog.json").read_text())["arrays"]
        frame = {name: _read_array(fdir, cat[name]) for name in FRAME_REQUIRED}
        if frame["chunk_state"].shape != (n, CHUNK_STATE_DIM):
            raise ValueError(f"frame {fi} chunk_state shape")
        if frame["bond_joint_state"].shape != (e, JOINT_STATE_DIM):
            raise ValueError(f"frame {fi} bond_joint_state shape")
        if frame["chunk_load_bins"].shape != (n, K, LOCAL6_DIM):
            raise ValueError(f"frame {fi} chunk_load_bins shape")
        classes = frame["bond_fail_class"]
        if np.any(classes < 0) or np.any(classes > K):
            raise ValueError(f"frame {fi} bond_fail_class out of range")
        frames.append(frame)
    return EventArrays(manifest=manifest, static=static, frames=frames)


def validate_event(path: Path | str) -> dict[str, Any]:
    ev = load_event(path)
    return ev.manifest
