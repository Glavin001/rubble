"""Carried-state pushforward (Brandstetter et al. 2022) and GNS input noise.

`rewrite_carried` replaces a successor call's carried columns (`schema.CARRIED_COLUMNS`)
with values derived from a predicted frame outcome, by the direction convention in
OUTCOME_CALLS_SCHEMA.md. Topology, damage summaries and geometry stay the oracle's; DAgger covers
those. This is the one place Python re-derives recorder columns, and only for training:
`tests/test_outcome_pushforward.py` checks that feeding the oracle's own outcome reproduces
the recorded successor exactly.
"""

from __future__ import annotations

import numpy as np

from . import schema as S
from .geometry import quat_to_matrix_np

_V3 = [f"v_rel_body_{i}" for i in range(3)]
_W3 = [f"w_body_{i}" for i in range(3)]
_E3 = [f"v_rel_edge_{i}" for i in range(3)]


def _cols(arr: str, names) -> list[int]:
    return [S.col(arr, n) for n in names]


def _row_frames(frame: np.ndarray) -> np.ndarray:
    """[E,3,3] → [2E,3,3]: row 2k uses F, row 2k+1 uses (−t1, t2, −n)."""
    rev = frame * np.array([-1.0, 1.0, -1.0])[None, :, None]
    out = np.empty((2 * len(frame), 3, 3), dtype=np.float64)
    out[0::2], out[1::2] = frame, rev
    return out


def rewrite_carried(c: S.Call, pred: dict[str, dict[int, np.ndarray]]) -> S.Call:
    """`pred` maps keys to predicted frame-start state of `c` (= frame-end of its parent):
    `chunk_v`, `chunk_w` (by chunk key), `imp_v`, `imp_w` (by impactor key), and
    `bond_damage`, `bond_peak` (by bond key). Missing keys keep the recorded values."""
    a = {k: np.array(v) for k, v in c.arrays.items()}
    nc, ni = c.counts["chunks"], c.counts["impactors"]
    v_ref, L, cw = c.scales["v"], c.scales["L"], c.scales["c"]
    v = np.concatenate([a["chunk_v"], a["impactor_v"]]).astype(np.float64)
    w = np.concatenate([a["chunk_w"], a["impactor_w"]]).astype(np.float64)
    for i, k in enumerate(a["chunk_key"].tolist()):
        if k in pred.get("chunk_v", {}):
            v[i], w[i] = pred["chunk_v"][k], pred["chunk_w"][k]
    for i, k in enumerate(a["impactor_key"].tolist()):
        if k in pred.get("imp_v", {}):
            v[nc + i], w[nc + i] = pred["imp_v"][k], pred["imp_w"][k]
    mass = np.concatenate([a["chunk_mass"], a["impactor_mass"]]).astype(np.float64)
    pos = np.concatenate([a["chunk_pos"], a["impactor_pos"]]).astype(np.float64)
    q = np.concatenate([a["chunk_q_world"], a["impactor_q_world"]]).astype(np.float64)
    rb = quat_to_matrix_np(q)  # body → world
    vcom = (mass[:, None] * v).sum(0) / mass.sum()

    cf = a["chunk_feat"].copy()
    cf[:, _cols("chunk_feat", _V3)] = np.einsum("nji,nj->ni", rb[:nc], v[:nc] - vcom) / v_ref
    cf[:, _cols("chunk_feat", _W3)] = np.einsum("nji,nj->ni", rb[:nc], w[:nc]) * L / v_ref
    a["chunk_feat"] = cf.astype(np.float32)
    if ni:
        f = a["impactor_feat"].copy()
        vrel = np.einsum("nji,nj->ni", rb[nc:], v[nc:] - vcom)
        f[:, _cols("impactor_feat", _V3)] = vrel / v_ref
        f[:, S.col("impactor_feat", "speed_over_c")] = np.linalg.norm(vrel, axis=1) / cw
        a["impactor_feat"] = f.astype(np.float32)

    def point_edges(t: str):
        if not len(a[f"{t}_src"]):
            return
        arr = f"{t}_feat"
        f = a[arr].copy()
        s_, d_ = a[f"{t}_src"].astype(np.int64), a[f"{t}_dst"].astype(np.int64)
        fr = _row_frames(a[f"{t}_frame"].astype(np.float64))
        off = f[:, _cols(arr, [f"offset_{i}" for i in range(3)])].astype(np.float64) * L
        pnt = pos[s_] + np.einsum("nji,nj->ni", fr, off)  # bond point, world
        vs = v[s_] + np.cross(w[s_], pnt - pos[s_])
        vd = v[d_] + np.cross(w[d_], pnt - pos[d_])
        f[:, _cols(arr, _E3)] = np.einsum("nij,nj->ni", fr, vd - vs) / v_ref
        a[arr] = f.astype(np.float32)

    point_edges("bond")
    point_edges("cracked")
    if len(a["bond_src"]):
        f = a["bond_feat"]
        keys = a["bond_key"].tolist()
        for k, key in enumerate(keys):
            if key in pred.get("bond_damage", {}):
                f[2 * k:2 * k + 2, S.col("bond_feat", "damage")] = pred["bond_damage"][key]
            if key in pred.get("bond_peak", {}):
                f[2 * k:2 * k + 2, S.col("bond_feat", "peak_util_prev")] = pred["bond_peak"][key]

    def contact_edges(t: str, spin: bool):
        if not len(a[f"{t}_src"]):
            return
        arr = f"{t}_feat"
        f = a[arr].copy()
        s_, d_ = a[f"{t}_src"].astype(np.int64), a[f"{t}_dst"].astype(np.int64)
        fr = _row_frames(a[f"{t}_frame"].astype(np.float64))
        rel = np.einsum("nij,nj->ni", fr, v[s_] - v[d_])
        f[:, S.col(arr, "v_n")] = rel[:, 2] / v_ref
        f[:, S.col(arr, "v_t_0")] = rel[:, 0] / v_ref
        f[:, S.col(arr, "v_t_1")] = rel[:, 1] / v_ref
        f[:, S.col(arr, "v_n_over_c")] = rel[:, 2] / cw
        if spin:
            imp = np.where(s_ >= nc, s_, d_)
            radius = np.exp(f[:, S.col(arr, "log_radius")].astype(np.float64)) * L
            f[:, _cols(arr, [f"spin_{i}" for i in range(3)])] = (
                np.einsum("nij,nj->ni", fr, w[imp]) * radius[:, None] / v_ref)
        a[arr] = f.astype(np.float32)

    contact_edges("impactor_contact", True)
    contact_edges("structure_contact", False)
    out = S.Call(**{**c.__dict__, "arrays": a})
    return out


def prediction_from_targets(c: S.Call) -> dict[str, dict[int, np.ndarray]]:
    """The oracle's own outcome of `c`, in `rewrite_carried`'s input form (for the parity
    test and for teacher-forced debugging)."""
    a = c.arrays
    return {
        "chunk_v": dict(zip(a["chunk_key"].tolist(), a["y_chunk_v_end"].astype(np.float64))),
        "chunk_w": dict(zip(a["chunk_key"].tolist(), a["y_chunk_w_end"].astype(np.float64))),
        "imp_v": dict(zip(a["impactor_key"].tolist(), a["y_impactor_v_end"].astype(np.float64))),
        "imp_w": dict(zip(a["impactor_key"].tolist(), a["y_impactor_w_end"].astype(np.float64))),
        "bond_damage": dict(zip(a["bond_key"].tolist(),
                                (a["bond_feat"][0::2, S.col("bond_feat", "damage")]
                                 + a["y_bond_ddamage"]).astype(np.float64))),
        "bond_peak": dict(zip(a["bond_key"].tolist(), a["y_bond_peak_util"].astype(np.float64))),
    }


def add_noise(c: S.Call, sigma: float, rng: np.random.Generator) -> S.Call:
    """GNS-style Gaussian noise on the carried columns (damage kept in [0, 1])."""
    if sigma <= 0:
        return c
    a = dict(c.arrays)
    for arr, names in S.CARRIED_COLUMNS.items():
        if not len(a[arr]):
            continue
        idx = S.carried_indices(arr)
        f = np.array(a[arr])
        f[:, idx] += rng.normal(0.0, sigma, size=(len(f), len(idx))).astype(np.float32)
        if arr == "bond_feat":
            d = S.col("bond_feat", "damage")
            f[:, d] = np.clip(f[:, d], 0.0, 1.0)
            # twins carry the same damage
            f[1::2, d] = f[0::2, d]
        a[arr] = f
    return S.Call(**{**c.__dict__, "arrays": a})
