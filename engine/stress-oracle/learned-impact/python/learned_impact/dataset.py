"""Dataset helpers: synthetic batch, single event, multi-event directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .constants import CHUNK_STATE_DIM, JOINT_STATE_DIM, K, LOCAL6_DIM
from .schema import load_event


def bidirectional_csr(endpoints: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Undirected (E,2) → CSR row_ptr (N+1), col_idx (2E)."""
    directed_rows = endpoints.reshape(-1).astype(np.int64)
    if directed_rows.size and (directed_rows.min() < 0 or directed_rows.max() >= n):
        raise ValueError('bond endpoint outside chunk array')
    deg = np.bincount(directed_rows, minlength=n)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(deg)
    # Stable grouping exactly preserves the old edge visitation order (including
    # repeated edges/self edges), without Python loops over 10k-body graphs.
    neighbors = endpoints[:, ::-1].reshape(-1).astype(np.int64)
    col = neighbors[np.argsort(directed_rows, kind='stable')]
    return row_ptr, col


def frame_to_tensors(static: dict, frame: dict) -> dict[str, torch.Tensor]:
    """Pack supplied arrays; inference callers need no fabricated future labels."""
    endpoints = static["bond_endpoints"]
    n = static["chunk_mass"].shape[0]
    row_ptr, col = bidirectional_csr(endpoints, n)
    batch = {
        "chunk_mass": torch.from_numpy(static["chunk_mass"].copy()),
        "chunk_inertia": torch.from_numpy(static["chunk_inertia"].copy()),
        "chunk_half_extents": torch.from_numpy(static["chunk_half_extents"].copy()),
        "bond_endpoints": torch.from_numpy(endpoints.astype(np.int64)),
        "csr_row_ptr": torch.from_numpy(row_ptr),
        "csr_col": torch.from_numpy(col),
        "chunk_state": torch.from_numpy(frame["chunk_state"].copy()),
        "chunk_gravity_local": torch.from_numpy(frame["chunk_gravity_local"].copy()),
        "bond_joint_state": torch.from_numpy(frame["bond_joint_state"].copy()),
    }
    # Keep physical properties and auxiliary targets. The legacy encoder ignores
    # these keys; the versioned physics encoder consumes them explicitly.
    for key, array in {**static, **frame}.items():
        if key not in batch:
            batch[key] = torch.from_numpy(array.astype(np.int64) if key=='bond_fail_class' else array.copy())
    return batch


def load_first_frame(event_dir: Path | str) -> dict[str, torch.Tensor]:
    ev = load_event(event_dir)
    return next(event_batches(ev))


def iter_event_frames(event_dir: Path | str):
    ev = load_event(event_dir)
    yield from event_batches(ev)


def event_batches(ev, *, bond_policy="first_failure"):
    """Teacher-forced live graph: remove previously failed bonds before each frame.

    runtime_alive instead retains cracked interfaces while the exact history
    marks them alive, with a separate connected/first-fracture eligibility mask.

    Old recordings repeat class K for dead bonds. Keep the raw labels on disk,
    but never train or score those bonds as new failures in later frames.
    """
    if bond_policy not in ("first_failure","runtime_alive"):raise ValueError("unknown bond policy")
    if bond_policy=="runtime_alive" and ev.exact_joint_history is None:raise ValueError("runtime_alive graph requires exact joint history")
    dead = np.zeros(len(ev.static["bond_endpoints"]), dtype=bool)
    for fi, frame in enumerate(ev.frames):
        ids = np.flatnonzero(ev.exact_joint_history.flags[fi,:,2]==1 if bond_policy=="runtime_alive" else ~dead)
        static = {k: (v[ids] if k.startswith('bond_') else v) for k, v in ev.static.items()}
        live = {k: (v[ids] if k.startswith("bond_") else v) for k, v in frame.items()}
        batch = frame_to_tensors(static, live)
        batch["bond_ids"] = torch.from_numpy(ids)
        if bond_policy=="runtime_alive":
            eligible=ev.exact_joint_history.flags[fi,ids,3]==1
            batch["bond_failure_eligible"]=torch.from_numpy(eligible.copy())
            # An already-disconnected interface may still carry friction/compression.
            # Retain its state/force labels but never train it as a new break.
            batch["bond_fail_class"][~batch["bond_failure_eligible"]]=0
        batch['dt'] = torch.tensor(float(ev.manifest['dt']), dtype=torch.float32)
        # Unknown legacy QS convention is a separate value, never guessed.
        batch['qs_mode'] = torch.tensor(float(ev.context.get('qs_probe', -1)))
        batch['load_basis_v2'] = torch.tensor(float(ev.context.get('load_basis') == 'frame_start_cluster'))
        if 'bond_frames' in ev.context:
            batch['bond_frames'] = torch.tensor(np.asarray(ev.context['bond_frames'])[ids], dtype=torch.float32)
        if ev.cluster_history:
            batch['delta_valid'] = torch.tensor(np.asarray(ev.cluster_history[fi]) == np.asarray(ev.cluster_history[fi+1]))
        if ev.world_inputs:
            batch.update({k:torch.from_numpy(v.copy()) for k,v in ev.world_inputs[fi].items()})
        if ev.world_targets:
            batch.update({k:torch.from_numpy(v.copy()) for k,v in ev.world_targets[fi].items()})
        if ev.material_inputs:
            batch.update({k:torch.from_numpy((v[ids] if k.startswith('bond_') else v).copy())
                          for k,v in ev.material_inputs.items()})
        if ev.exact_joint_history is not None:
            from .joint_history import finite_features
            history=ev.exact_joint_history
            # Both use the same live graph ordering; future history is explicitly
            # a target and must never enter feature preparation.
            batch['bond_exact_history']=torch.from_numpy(history.frame_start(fi)[ids].copy())
            batch['bond_exact_history_end']=torch.from_numpy(finite_features(history.values[fi+1],history.flags[fi+1])[ids].copy())
            # Preserve the exact connectivity boundary before float32 log
            # encoding can round a nearly-complete damage value to log(2).
            batch['bond_damage_complete_start']=torch.from_numpy((history.values[fi,ids,0]>=1.).copy())
            batch['bond_damage_complete_end']=torch.from_numpy((history.values[fi+1,ids,0]>=1.).copy())
        if ev.bond_response_history is not None:
            from .response_history import finite_features as response_features
            response=ev.bond_response_history
            batch['bond_response_encoded']=torch.from_numpy(response.frame_start(fi)[ids].copy())
            batch['bond_response_encoded_end']=torch.from_numpy(response_features(response.values[fi+1])[ids].copy())
            batch['bond_response_alive']=torch.from_numpy((response.flags[fi,ids,0]==1).copy())
            batch['bond_response_alive_end']=torch.from_numpy((response.flags[fi+1,ids,0]==1).copy())
        yield batch
        dead |= frame["bond_fail_class"] > 0


def list_events(dataset_dir: Path | str) -> list[Path]:
    """Event dirs under a set, or a single event dir (uses manifest.json if present)."""
    root = Path(dataset_dir)
    man = root / "manifest.json"
    if man.exists():
        data = json.loads(man.read_text())
        if "events" in data:
            return [root / e["path"] for e in data["events"]]
        # Single-event manifest (event_id + on-disk static/).
        if "event_id" in data and (root / "static").is_dir():
            return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def load_all_frames(dataset_dir: Path | str) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for ev in list_events(dataset_dir):
        batches.extend(iter_event_frames(ev))
    if not batches:
        raise FileNotFoundError(f"no events in {dataset_dir}")
    return batches


def synthetic_batch(n: int = 8, e: int = 12, seed: int = 0) -> dict[str, torch.Tensor]:
    """In-memory batch matching the schema (no disk)."""
    rng = np.random.default_rng(seed)
    endpoints = np.zeros((e, 2), dtype=np.int64)
    for i in range(e):
        a = i % n
        b = (i + 1) % n
        endpoints[i] = (a, b)
    static = {
        "chunk_mass": rng.random(n, dtype=np.float32) + 0.5,
        "chunk_inertia": rng.random((n, 3), dtype=np.float32),
        "chunk_half_extents": np.full((n, 3), 0.2, dtype=np.float32),
        "bond_endpoints": endpoints.astype(np.int32),
    }
    frame = {
        "chunk_state": rng.normal(size=(n, CHUNK_STATE_DIM)).astype(np.float32) * 0.01,
        "chunk_load_bins": rng.normal(size=(n, K, LOCAL6_DIM)).astype(np.float32),
        "chunk_gravity_local": np.tile(np.array([0, 0, -9.81], np.float32), (n, 1)),
        "bond_joint_state": rng.random((e, JOINT_STATE_DIM)).astype(np.float32) * 0.1,
        "bond_qs_force": rng.normal(size=(e, LOCAL6_DIM)).astype(np.float32),
        "bond_fail_class": np.zeros(e, dtype=np.int32),
        "bond_peak_util": rng.random(e, dtype=np.float32),
        "chunk_state_delta": rng.normal(size=(n, CHUNK_STATE_DIM)).astype(np.float32) * 0.001,
    }
    frame["bond_fail_class"][e // 2] = 2
    frame["bond_peak_util"][e // 2] = 1.1
    return frame_to_tensors(static, frame)
