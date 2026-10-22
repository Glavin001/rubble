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
_DV3 = [f"def_v_body_{i}" for i in range(3)]
_DW3 = [f"def_w_body_{i}" for i in range(3)]


def body_relative_rates_np(v, w, x, m, inertia_world, body, n_bodies):
    """numpy twin of `losses.body_relative_rates` (world frame)."""
    M = np.zeros(n_bodies)
    np.add.at(M, body, m)
    M = np.maximum(M, 1e-12)
    X = np.zeros((n_bodies, 3))
    np.add.at(X, body, m[:, None] * x)
    X /= M[:, None]
    r = x - X[body]
    vb = np.zeros((n_bodies, 3))
    np.add.at(vb, body, m[:, None] * v)
    vb /= M[:, None]
    L = np.zeros((n_bodies, 3))
    np.add.at(L, body, np.cross(r, m[:, None] * v) + np.einsum("nij,nj->ni", inertia_world, w))
    rr = (r * r).sum(-1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :]
    Iw = np.zeros((n_bodies, 3, 3))
    np.add.at(Iw, body, inertia_world + m[:, None, None] * rr)
    wb = np.linalg.solve(Iw + 1e-12 * np.eye(3), L[:, :, None])[..., 0]
    return v - vb[body] - np.cross(wb[body], r), w - wb[body]


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
    # /4 hidden rates: the predicted velocities about each frame-start body's rigid fit, in
    # the body frame. Poses, and so u and θ, stay the recorded ones, like all geometry here.
    if "chunk_body" in a and "chunk_body_q" in a and nc:
        body = a["chunk_body"].astype(np.int64)
        iw = np.einsum("nij,nj,nkj->nik", rb[:nc],
                       a["chunk_inertia_body"].astype(np.float64), rb[:nc])
        dv, dw = body_relative_rates_np(v[:nc], w[:nc], pos[:nc], mass[:nc], iw, body,
                                        int(body.max()) + 1)
        rbody = quat_to_matrix_np(a["chunk_body_q"].astype(np.float64))
        dv_b = np.einsum("nji,nj->ni", rbody, dv)
        dw_b = np.einsum("nji,nj->ni", rbody, dw)
        R = np.linalg.norm(a["chunk_feat"][:, _cols("chunk_feat", [f"half_extent_{i}"
                                                                  for i in range(3)])]
                           .astype(np.float64) * L, axis=1)
        cf[:, _cols("chunk_feat", _DV3)] = dv_b / v_ref
        cf[:, _cols("chunk_feat", _DW3)] = dw_b * R[:, None] / v_ref
        d = a["chunk_def"].astype(np.float64).copy()
        d[:, 6:9], d[:, 9:12] = dv_b, dw_b
        a["chunk_def"] = d.astype(np.float32)
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
                f[2 * k:2 * k + 2, S.col("bond_feat", "log1p_peak_util_prev")] = np.log1p(
                    pred["bond_peak"][key])
            if key in pred.get("bond_margin", {}):
                f[2 * k:2 * k + 2, S.col("bond_feat", "fail_margin")] = pred["bond_margin"][key]

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
        "bond_margin": dict(zip(a["bond_key"].tolist(),
                                a["y_bond_fail_margin"].astype(np.float64))),
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


def _hops(nc: int, src: np.ndarray, dst: np.ndarray, seeds: list[int]) -> np.ndarray:
    """Bond-graph hops from `seeds` (directed twin rows, chunks only); 10⁶ if unreachable."""
    from collections import deque

    adj: list[list[int]] = [[] for _ in range(nc)]
    for a, b in zip(src.tolist(), dst.tolist()):
        if a < nc and b < nc:
            adj[a].append(b)
    d = np.full(nc, 10**6, dtype=np.int64)
    q = deque()
    for s in seeds:
        d[s] = 0
        q.append(s)
    while q:
        a = q.popleft()
        for b in adj[a]:
            if d[b] > d[a] + 1:
                d[b] = d[a] + 1
                q.append(b)
    return d


def apply_topology(parent: S.Call, succ: S.Call, model_broke: set[int]) -> S.Call:
    """Give `succ` (the oracle's next call) the live bond set the *model's* breaks imply.

    - bonds the oracle broke in `parent` but the model kept are added back, with their
      parent rows (static columns are time-invariant; carried columns are then re-derived by
      `rewrite_carried`) and the censored target "already failed" (broke = 1, r ≥ 1, first
      bin), and removed from `succ`'s cracked edges;
    - bonds the model broke but the oracle kept are dropped with their targets;
    - chunk columns that follow from topology are recomputed: damage mean/max,
      live/broken counts, hops to contact and to anchors.

    Approximations, until the Rust call builder can rebuild a call from a given state (asked
    of P1): model-broken bonds do not become cracked-face edges; added-back bonds keep the
    parent's frames and zero variant-B QS columns; wave arrival, geometry and predicted
    contacts stay the oracle's. DAgger (oracle restarts from the decoded state) removes all
    of these.
    """
    pa, sa = parent.arrays, {k: np.array(v) for k, v in succ.arrays.items()}
    nc = succ.counts["chunks"]
    p_keys = pa["bond_key"].tolist()
    oracle_broke = {k for k, b in zip(p_keys, pa["y_bond_broke"].tolist()) if b}
    add_back = sorted(oracle_broke - set(model_broke))
    s_keys = sa["bond_key"].tolist()
    remove = set(model_broke) - oracle_broke
    keep = np.array([k not in remove for k in s_keys], dtype=bool)
    keep_dir = np.repeat(keep, 2)
    und = [n for n in sa if n.startswith("y_bond_") or n.startswith("base_") and "_bond_" in n]
    d_removed = float(sa["y_bond_dissipated"][~keep].astype(np.float64).sum())
    for n in ("bond_src", "bond_dst", "bond_feat", "bond_qs_feat"):
        sa[n] = sa[n][keep_dir]
    for n in ("bond_key", "bond_frame", *und):
        sa[n] = sa[n][keep]
    if add_back:
        p_idx = {k: i for i, k in enumerate(p_keys)}
        rows = np.array([p_idx[k] for k in add_back])
        rows_dir = np.stack([2 * rows, 2 * rows + 1], 1).reshape(-1)
        # bonds join chunks only; chunk and impactor keys are separate namespaces (impactor
        # 0 and chunk (0 << 32) | 0 are both key 0), so map through chunk keys alone
        p_nc = parent.counts["chunks"]
        s_chunk = {k: i for i, k in enumerate(sa["chunk_key"].tolist())}
        remap = np.array([s_chunk.get(k, -1) for k in pa["chunk_key"].tolist()]
                         + [-1] * parent.counts["impactors"])
        ps, pd = pa["bond_src"][rows_dir], pa["bond_dst"][rows_dir]
        if np.any(ps >= p_nc) or np.any(pd >= p_nc):
            raise S.SchemaError("a bond endpoint is not a chunk")
        src, dst = remap[ps], remap[pd]
        ok_dir = (src >= 0) & (dst >= 0)
        ok = ok_dir[0::2] & ok_dir[1::2]  # both ends must still be in the island
        rows, rows_dir = rows[ok], rows_dir[np.repeat(ok, 2)]
        src, dst = src[np.repeat(ok, 2)], dst[np.repeat(ok, 2)]
        n_add = len(rows)
        add = {
            "bond_src": src.astype(np.int32), "bond_dst": dst.astype(np.int32),
            "bond_feat": pa["bond_feat"][rows_dir], "bond_key": pa["bond_key"][rows],
            "bond_frame": pa["bond_frame"][rows],
            "bond_qs_feat": np.zeros((2 * n_add, len(S.BOND_QS_FEAT_COLUMNS)), np.float32),
            "y_bond_peak_util": np.maximum(pa["y_bond_peak_util"][rows], 1.0).astype(np.float32),
            "y_bond_fail_margin": np.maximum(pa["y_bond_fail_margin"][rows], 1.0
                                             ).astype(np.float32),
            "y_bond_broke": np.ones(n_add, np.int32),
            "y_bond_break_bin": np.ones(n_add, np.int32),
            "y_bond_fail_mode": pa["y_bond_fail_mode"][rows],
            "y_bond_ddamage": np.zeros(n_add, np.float32),
            "y_bond_dissipated": np.zeros(n_add, np.float32),
            "y_bond_support_path": np.zeros(n_add, np.int32),
            "y_bond_break_freq": np.ones(n_add, np.float32),
            "base_qs_bond_broke": np.ones(n_add, np.int32),
            "base_punch_bond_broke": np.ones(n_add, np.int32),
            "base_qs_bond_peak_util": np.ones(n_add, np.float32),
            "base_punch_bond_peak_util": np.ones(n_add, np.float32),
        }
        for n, v in add.items():
            if n in sa:
                sa[n] = np.concatenate([sa[n], v.astype(sa[n].dtype)])
        added = set(pa["bond_key"][rows].tolist())
        ck = sa["cracked_key"].tolist()
        ckeep = np.array([k not in added for k in ck], dtype=bool)
        for n in ("cracked_src", "cracked_dst", "cracked_feat"):
            sa[n] = sa[n][np.repeat(ckeep, 2)]
        for n in ("cracked_key", "cracked_frame"):
            sa[n] = sa[n][ckeep]
    # chunk columns that follow from topology
    cf = sa["chunk_feat"].copy()
    c = lambda n: S.col("chunk_feat", n)  # noqa: E731
    total = cf[:, c("n_live_bonds")] + cf[:, c("n_broken_bonds")]
    src, dst = sa["bond_src"].astype(np.int64), sa["bond_dst"].astype(np.int64)
    dmg = sa["bond_feat"][:, S.col("bond_feat", "damage")]
    live = np.bincount(src[src < nc], minlength=nc).astype(np.float32)
    cf[:, c("n_live_bonds")] = live / 6
    cf[:, c("n_broken_bonds")] = total - live / 6
    dsum = np.bincount(src, weights=dmg, minlength=nc)[:nc]
    cf[:, c("bond_damage_mean")] = dsum / np.maximum(live, 1)
    dmax = np.zeros(nc)
    np.maximum.at(dmax, src, dmg)
    cf[:, c("bond_damage_max")] = dmax
    ic_s, ic_d = sa["impactor_contact_src"], sa["impactor_contact_dst"]
    contact = sorted({int(d) for s_, d in zip(ic_s, ic_d) if s_ >= nc and d < nc})
    h = _hops(nc, src, dst, contact)
    cf[:, c("hops_to_contact")] = np.minimum(h / 16, 4)
    anchors = list(np.nonzero(sa["chunk_anchored"])[0])
    ha = _hops(nc, src, dst, anchors) if anchors else np.full(nc, 10**6)
    cf[:, c("hops_to_anchor")] = np.minimum(ha / 16, 4)
    sa["chunk_feat"] = cf.astype(np.float32)
    energy = dict(succ.energy)
    energy["D_bond"] = float(sa["y_bond_dissipated"].astype(np.float64).sum())
    if d_removed and "D_bond_joint" in energy:
        energy["D_bond_joint"] = max(energy["D_bond_joint"] - d_removed, 0.0)
        energy["D_bond_damping"] = energy["D_bond"] - energy["D_bond_joint"]
    counts = dict(succ.counts, bond=len(sa["bond_key"]), cracked=len(sa["cracked_key"]))
    return S.Call(**{**succ.__dict__, "arrays": sa, "counts": counts, "energy": energy})


def push(parent: S.Call, succ: S.Call, pred: dict, model_broke: set[int] | None) -> S.Call:
    """One pushforward step: the model's topology (if given), then its carried state."""
    s = apply_topology(parent, succ, model_broke) if model_broke is not None else succ
    return rewrite_carried(s, pred)
