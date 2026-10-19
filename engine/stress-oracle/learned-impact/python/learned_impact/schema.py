"""Load and validate `learned-impact/1` event directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    context: dict[str, Any] = field(default_factory=dict)
    cluster_history: list[list[int]] = field(default_factory=list)
    world_inputs: list[dict[str, np.ndarray]] = field(default_factory=list)
    world_source_sha256: str | None = None
    world_targets: list[dict[str, np.ndarray]] = field(default_factory=list)
    exact_joint_history: Any | None = None
    bond_response_history: Any | None = None
    material_inputs: dict[str, np.ndarray] = field(default_factory=dict)
    material_source_sha256: str | None = None


def load_event(path: Path | str, *, world_inputs=False, world_targets=False, forcing_inputs=False, exact_joint_history=False, material_inputs=False, projectile_inputs=False, bond_response_history=False) -> EventArrays:
    if projectile_inputs and not world_inputs:raise ValueError('dynamic projectile inputs require validated world inputs')
    if forcing_inputs and not world_inputs:raise ValueError('forcing inputs require validated world inputs')
    if world_targets and not world_inputs:raise ValueError('world targets require validated world inputs')
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
    context_path = root / 'context.json'
    context = json.loads(context_path.read_text()) if context_path.exists() else {}
    preview_format = context.get('load_preview', {}).get('format')
    if preview_format not in (None, 'rigid-contact-preview/1'):
        raise ValueError(f'unknown load preview format: {preview_format}')
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
        for name, shape, dtype in [('chunk_preview_load_bins', (n, K, LOCAL6_DIM), 'f32'),
                                   ('chunk_preview_contact', (n,), 'i32')]:
            if preview_format or name in cat:
                if not preview_format or name not in cat:
                    raise ValueError(f'frame {fi}: preview contract/array mismatch: {name}')
                a = _read_array(fdir, cat[name])
                if a.shape != shape or cat[name]['dtype'] != dtype or not np.isfinite(a).all():
                    raise ValueError(f'frame {fi}: invalid {name}')
                if dtype == 'i32' and not np.isin(a, [0, 1]).all():
                    raise ValueError(f'frame {fi}: invalid preview contact flag')
                frame[name] = a
        classes = frame["bond_fail_class"]
        if np.any(classes < 0) or np.any(classes > K):
            raise ValueError(f"frame {fi} bond_fail_class out of range")
        frames.append(frame)
    replay_path = root / 'replay.json'
    observations,world_digest,labels=[],None,[]
    if world_inputs:
        from .world_inputs import load_world_inputs
        observations,world_digest,clusters,labels=load_world_inputs(root,context,manifest,include_targets=world_targets,include_forcing=forcing_inputs,include_projectiles=projectile_inputs)
    else:
        clusters = [f['clusters'] for f in json.loads(replay_path.read_text())['frames']] if replay_path.exists() else []
    history=None
    if exact_joint_history or context.get('joint_history'):
        from .joint_history import load_joint_history
        history=load_joint_history(root,context,manifest)
    materials,material_digest={},None
    response=None
    if bond_response_history or context.get('bond_response'):
        from .response_history import load_response_history
        response=load_response_history(root,context,manifest,history,frames)
    if material_inputs:
        import hashlib
        from .material_inputs import observations as material_observations
        scene_bytes=(root/'scene.json').read_bytes()
        materials=material_observations(json.loads(scene_bytes),context,manifest,static)
        material_digest=hashlib.sha256(scene_bytes).hexdigest()
    return EventArrays(manifest=manifest, static=static, frames=frames, context=context, cluster_history=clusters,
                       world_inputs=observations,world_source_sha256=world_digest,world_targets=labels,
                       exact_joint_history=history if exact_joint_history else None,
                       bond_response_history=response if bond_response_history else None,
                       material_inputs=materials,material_source_sha256=material_digest)


def validate_event(path: Path | str) -> dict[str, Any]:
    ev = load_event(path)
    return ev.manifest
