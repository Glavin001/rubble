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
    deg = np.zeros(n, dtype=np.int64)
    for i, j in endpoints:
        deg[int(i)] += 1
        deg[int(j)] += 1
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(deg)
    col = np.empty(int(row_ptr[-1]), dtype=np.int64)
    fill = row_ptr[:-1].copy()
    for ei, (i, j) in enumerate(endpoints):
        i, j = int(i), int(j)
        col[fill[i]] = j
        fill[i] += 1
        col[fill[j]] = i
        fill[j] += 1
    return row_ptr, col


def frame_to_tensors(static: dict, frame: dict) -> dict[str, torch.Tensor]:
    endpoints = static["bond_endpoints"]
    n = static["chunk_mass"].shape[0]
    row_ptr, col = bidirectional_csr(endpoints, n)
    return {
        "chunk_mass": torch.from_numpy(static["chunk_mass"].copy()),
        "chunk_inertia": torch.from_numpy(static["chunk_inertia"].copy()),
        "chunk_half_extents": torch.from_numpy(static["chunk_half_extents"].copy()),
        "bond_endpoints": torch.from_numpy(endpoints.astype(np.int64)),
        "csr_row_ptr": torch.from_numpy(row_ptr),
        "csr_col": torch.from_numpy(col),
        "chunk_state": torch.from_numpy(frame["chunk_state"].copy()),
        "chunk_load_bins": torch.from_numpy(frame["chunk_load_bins"].copy()),
        "chunk_gravity_local": torch.from_numpy(frame["chunk_gravity_local"].copy()),
        "bond_joint_state": torch.from_numpy(frame["bond_joint_state"].copy()),
        "bond_qs_force": torch.from_numpy(frame["bond_qs_force"].copy()),
        "bond_fail_class": torch.from_numpy(frame["bond_fail_class"].astype(np.int64)),
        "bond_peak_util": torch.from_numpy(frame["bond_peak_util"].copy()),
        "chunk_state_delta": torch.from_numpy(frame["chunk_state_delta"].copy()),
    }


def load_first_frame(event_dir: Path | str) -> dict[str, torch.Tensor]:
    ev = load_event(event_dir)
    return frame_to_tensors(ev.static, ev.frames[0])


def iter_event_frames(event_dir: Path | str):
    ev = load_event(event_dir)
    for frame in ev.frames:
        yield frame_to_tensors(ev.static, frame)


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
