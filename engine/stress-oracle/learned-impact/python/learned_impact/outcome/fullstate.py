"""The oracle's full dynamic state as model inputs and heads (calls `/5`, FULL_STATE_V5_DRAFT.md
"Frozen field set").

P1's `/5` calls carry every serialised `World` field per entity at the frame's start (`fs_*_in`)
and end (`fs_*_out`), columns named by field path (`manifest.full_state_columns`). This module
picks the carried bond and chunk fields, encodes them nondimensionally, aligns them to the
call's bond and chunk rows by key, and defines each head:
- **monotone** fields (running maxima and accumulators: κ, κ_c, fail_ts, fail_c, crush,
  dissipated): encoded end = encoded start + softplus(raw − 2), so "unchanged" is
  representable and they never decrease, the margin head's form;
- **free** fields: encoded end = encoded start + raw (a residual on the frame-start value);
- the failure **mode**: 3 classes (none, tension, compression).

Encodings (per bond: u_c, θ_c, G_f·A from `fs_static_bond_scale`, f_t; per chunk: m, L, g):
log1p for κ-like maxima and utilisation, ÷ u_c or θ_c for plastic offsets, 1/(1 + ductility)
(∞ at rest → 0), ÷ f_t for stresses, sign·log1p(|ε̇|·dt) for strain rates, ÷ (G_f·A) for
dissipated energy, ÷ (m·g) and ÷ (m·g·L) for support reactions. NaN (None, absent) → 0.

Not carried (classified diagnostic or derived, recomputed every substep): bond force, elastic,
measures, stored, damped; the chunk's `@` world arrays (loads, f_int, contact hits) are inputs
at most, a later step.
"""

from __future__ import annotations

import numpy as np

# (column, encoding, kind) — kind: "mono" (monotone), "free" (residual)
BOND_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("joint.kappa", "log1p", "mono"),
    ("joint.kappa_c", "log1p", "mono"),
    ("fail_ts", "log1p", "mono"),
    ("fail_c", "log1p", "mono"),
    ("joint.crush", "id", "mono"),
    ("joint.dissipated", "per_gf", "mono"),
    ("peak_fail_margin", "log1p", "free"),
    ("joint.ductility", "inv1p", "free"),
    ("joint.ductility_c", "inv1p", "free"),
    ("joint.plastic.lin[0]", "per_uc", "free"),
    ("joint.plastic.lin[1]", "per_uc", "free"),
    ("joint.plastic.ang[2]", "per_thc", "free"),
    ("joint.governing_stress", "per_ft", "free"),
    ("joint.strain_rate", "slog_dt", "free"),
    ("joint.utilization", "log1p", "free"),
)
BOND_MODE = ("joint.mode=tension", "joint.mode=compression")
N_BOND_IN = len(BOND_FIELDS) + len(BOND_MODE)  # encoded inputs per bond (twins share them)
N_BOND_OUT = len(BOND_FIELDS) + 3  # field heads + mode logits (none, tension, compression)
MONO = np.array([k == "mono" for _, _, k in BOND_FIELDS])

CHUNK_FIELDS: tuple[str, ...] = tuple(f"reaction[{i}].{a}" for i in (0, 1) for a in "xyz")
N_CHUNK_IN = len(CHUNK_FIELDS)
N_CHUNK_OUT = len(CHUNK_FIELDS)


def _enc(x: np.ndarray, how: str, sc: dict[str, np.ndarray], dt: float) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=np.inf)
    if how == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if how == "id":
        return x
    if how == "per_gf":
        return x / np.maximum(sc["gf_area_tension"], 1e-12)
    if how == "inv1p":
        return np.where(np.isfinite(x), 1.0 / (1.0 + np.clip(x, 0.0, None)), 0.0)
    if how == "per_uc":
        return x / np.maximum(sc["u_c"], 1e-12)
    if how == "per_thc":
        return x / np.maximum(sc["theta_c"], 1e-12)
    if how == "per_ft":
        return x / np.maximum(sc["f_t"], 1e-12)
    if how == "slog_dt":
        return np.sign(x) * np.log1p(np.abs(x) * dt)
    raise ValueError(how)


def available(c, manifest_cols: dict | None) -> bool:
    return manifest_cols is not None and "fs_bond_in" in c.arrays


def bond_arrays(c, cols: dict, static_scale: dict[int, np.ndarray], scale_names: list[str]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per undirected live bond (the call's `bond_key` order): encoded inputs [E, N_BOND_IN],
    encoded targets [E, len(BOND_FIELDS)], mode class [E] (0 none, 1 tension, 2 compression)."""
    a = c.arrays
    names = list(cols["bond"])
    keys = a["bond_key"].astype(np.int64)
    fk = a["fs_bond_key"].astype(np.int64)
    order = np.argsort(fk, kind="stable")
    pos = order[np.clip(np.searchsorted(fk[order], keys), 0, len(fk) - 1)]
    if len(keys) and not np.array_equal(fk[pos], keys):
        raise ValueError(f"{c.event_id}#{c.call_index}: live bonds missing from fs_bond_key")
    sc_rows = np.stack([static_scale.get(int(k), np.full(len(scale_names), np.nan)) for k in keys]) \
        if len(keys) else np.zeros((0, len(scale_names)))
    sc = {n: sc_rows[:, i] for i, n in enumerate(scale_names)}
    dt = float(c.scales["dt"])

    def take(arr: str, col: str) -> np.ndarray:
        return a[arr][pos, names.index(col)] if col in names else np.zeros(len(keys))
    x_in = np.stack([_enc(take("fs_bond_in", f), how, sc, dt) for f, how, _ in BOND_FIELDS]
                    + [np.nan_to_num(take("fs_bond_in", m)) for m in BOND_MODE], 1) \
        if len(keys) else np.zeros((0, N_BOND_IN))
    if "fs_bond_out" not in a:  # a live (inference) call: inputs only, targets unused
        return x_in.astype(np.float32), np.zeros((len(keys), len(BOND_FIELDS)), np.float32), \
            np.zeros(len(keys), np.int64)
    y = np.stack([_enc(take("fs_bond_out", f), how, sc, dt) for f, how, _ in BOND_FIELDS], 1) \
        if len(keys) else np.zeros((0, len(BOND_FIELDS)))
    t_, c_ = np.nan_to_num(take("fs_bond_out", BOND_MODE[0])), np.nan_to_num(take("fs_bond_out", BOND_MODE[1]))
    mode = np.where(t_ > 0.5, 1, np.where(c_ > 0.5, 2, 0)).astype(np.int64)
    return x_in.astype(np.float32), y.astype(np.float32), mode


def chunk_arrays(c, cols: dict, gravity: float) -> tuple[np.ndarray, np.ndarray]:
    """Per chunk (the call's chunk order = fs_chunk rows): encoded support reactions at the
    frame's start [Nc, 6] and end [Nc, 6] (force ÷ m·g, moment ÷ m·g·L)."""
    a = c.arrays
    names = list(cols["chunk"])
    m = a["chunk_mass"].astype(np.float64)
    g = max(gravity, 1e-6)
    L = float(c.scales["L"])
    sc = np.concatenate([np.repeat((m * g)[:, None], 3, 1), np.repeat((m * g * L)[:, None], 3, 1)], 1)

    def block(arr):
        return np.stack([np.nan_to_num(a[arr][:, names.index(f)]) if f in names else np.zeros(len(m))
                         for f in CHUNK_FIELDS], 1) / np.maximum(sc, 1e-12)
    ci = block("fs_chunk_in").astype(np.float32)
    return ci, (block("fs_chunk_out").astype(np.float32) if "fs_chunk_out" in a else np.zeros_like(ci))


def decode_raw(enc: np.ndarray, how: str, sc: dict[str, np.ndarray], dt: float) -> np.ndarray:
    """The inverse of `_enc`: an encoded prediction back to the field's raw units."""
    x = enc.astype(np.float64)
    if how == "log1p":
        return np.expm1(np.clip(x, 0.0, None))
    if how == "id":
        return x
    if how == "per_gf":
        return x * sc["gf_area_tension"]
    if how == "inv1p":
        return np.where(x > 0, 1.0 / np.clip(x, 1e-12, None) - 1.0, np.inf)
    if how == "per_uc":
        return x * sc["u_c"]
    if how == "per_thc":
        return x * sc["theta_c"]
    if how == "per_ft":
        return x * sc["f_t"]
    if how == "slog_dt":
        return np.sign(x) * np.expm1(np.abs(x)) / dt
    raise ValueError(how)


def predicted_columns() -> dict[str, list[str]]:
    """The full-state columns the model predicts, by entity, in P1's field-path names."""
    return {"bond": [f for f, _, _ in BOND_FIELDS] + list(BOND_MODE) + ["alive", "joint.damage"],
            "chunk": list(CHUNK_FIELDS)}


# ---- clusters (bodies) ------------------------------------------------------------------
# A frame-end body is predicted from its frame-start record: the cluster that held its lowest
# chunk at the frame's start. Targets are changes from that record: position ÷ L, rotation
# as the body-frame rotation vector q_s⁻¹ q_e, velocity ÷ v, angular velocity · L ÷ v, the
# activity (Active / Settled) and Δactive_timer ÷ dt. Keyed by the body's lowest chunk key.
CL_POS = tuple(f"pose.position.{a}" for a in "xyz")
CL_ROT = tuple(f"pose.rotation.{a}" for a in "wxyz")
CL_VEL = tuple(f"velocity.{a}" for a in "xyz")
CL_ANG = tuple(f"angular_velocity.{a}" for a in "xyz")
CL_ACT = ("activity=Active", "activity=Settled")
CL_TIMER = "active_timer"
N_CL_START = 3 + 3 + 2 + 1  # inputs: v ÷ v_ref, ω·L ÷ v_ref, activity one-hot, timer ÷ dt
N_CL_OUT = 3 + 3 + 3 + 3 + 1 + 2  # dpos, drot, dv, dω, dtimer, activity logits


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def _qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _q_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    s = np.linalg.norm(q[:, 1:], axis=1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[:, :1])
    return np.where(s > 1e-12, q[:, 1:] / np.maximum(s, 1e-300) * ang, 2.0 * q[:, 1:])


def _rotvec_to_q(r: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(r, axis=1, keepdims=True)
    half = 0.5 * a
    k = np.where(a > 1e-12, np.sin(half) / np.maximum(a, 1e-300), 0.5)
    return np.concatenate([np.cos(half), r * k], 1)


def _cl(rows: np.ndarray, names: list[str], cols) -> np.ndarray:
    return np.stack([np.nan_to_num(rows[:, names.index(c)]) if c in names else np.zeros(len(rows))
                     for c in cols], 1)


def cluster_groups(c, groups: np.ndarray) -> dict:
    """Frame-end bodies of call `c` given a chunk → group partition (the oracle's
    `fs_chunk_cluster_out` in training, the decoded fragments at inference): per group its
    key (lowest chunk key), the frame-start record (raw) and the encoded start inputs."""
    a = c.arrays
    names = list(c.fs["cols"]["cluster"])
    keys = a["chunk_key"].astype(np.int64)
    _, inv = np.unique(groups, return_inverse=True)
    n = int(inv.max()) + 1 if len(inv) else 0
    lowest = np.full(n, -1, np.int64)
    first_chunk = np.zeros(n, np.int64)
    for i, (gi, k) in enumerate(zip(inv, keys)):
        if lowest[gi] < 0 or k < lowest[gi]:
            lowest[gi], first_chunk[gi] = k, i
    start_rows = a["fs_chunk_cluster_in"].astype(np.int64)[first_chunk]
    st = a["fs_cluster_in"][start_rows] if n else np.zeros((0, len(names)))
    v_ref, L, dt = float(c.scales["v"]), float(c.scales["L"]), float(c.scales["dt"])
    start = {"pos": _cl(st, names, CL_POS), "rot": _cl(st, names, CL_ROT), "vel": _cl(st, names, CL_VEL),
             "ang": _cl(st, names, CL_ANG), "act": _cl(st, names, CL_ACT),
             "timer": _cl(st, names, (CL_TIMER,))[:, 0]}
    start["rot"][np.linalg.norm(start["rot"], axis=1) < 1e-9] = [1.0, 0.0, 0.0, 0.0]
    x = np.concatenate([start["vel"] / v_ref, start["ang"] * L / v_ref, start["act"],
                        (start["timer"] / dt)[:, None]], 1) if n else np.zeros((0, N_CL_START))
    return {"of_chunk": inv, "key": lowest, "start": start, "x": x.astype(np.float32)}


def cluster_targets(c, g: dict) -> tuple[np.ndarray, np.ndarray]:
    """Encoded frame-end targets of the oracle's clusters grouped as `g` (from
    `cluster_groups(c, fs_chunk_cluster_out)`): [Ncl, 13] changes and [Ncl] activity."""
    a = c.arrays
    names = list(c.fs["cols"]["cluster"])
    out_rows = np.zeros(len(g["key"]), np.int64)
    end_of_chunk = a["fs_chunk_cluster_out"].astype(np.int64)
    for i, gi in enumerate(g["of_chunk"]):
        out_rows[gi] = end_of_chunk[i]
    en = a["fs_cluster_out"][out_rows] if len(out_rows) else np.zeros((0, len(names)))
    v_ref, L, dt = float(c.scales["v"]), float(c.scales["L"]), float(c.scales["dt"])
    s = g["start"]
    q_e = _cl(en, names, CL_ROT)
    q_e[np.linalg.norm(q_e, axis=1) < 1e-9] = [1.0, 0.0, 0.0, 0.0]
    y = np.concatenate([(_cl(en, names, CL_POS) - s["pos"]) / L,
                        _q_to_rotvec(_qmul(_qinv(s["rot"]), q_e)),
                        (_cl(en, names, CL_VEL) - s["vel"]) / v_ref,
                        (_cl(en, names, CL_ANG) - s["ang"]) * L / v_ref,
                        ((_cl(en, names, (CL_TIMER,))[:, 0] - s["timer"]) / dt)[:, None]], 1) \
        if len(out_rows) else np.zeros((0, 13))
    act = _cl(en, names, CL_ACT)
    act_cls = np.where(act[:, 1] > 0.5, 1, 0).astype(np.int64) if len(out_rows) else np.zeros(0, np.int64)
    return y.astype(np.float32), act_cls


def cluster_raw(c, g: dict, enc: np.ndarray, act_logits: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Decoded cluster columns in raw units (world): position, rotation (w, x, y, z), velocity,
    angular velocity, activity one-hot, active_timer."""
    v_ref, L, dt = float(c.scales["v"]), float(c.scales["L"]), float(c.scales["dt"])
    s = g["start"]
    pos = s["pos"] + enc[:, 0:3] * L
    rot = _qmul(s["rot"], _rotvec_to_q(enc[:, 3:6]))
    vel = s["vel"] + enc[:, 6:9] * v_ref
    ang = s["ang"] + enc[:, 9:12] * v_ref / L
    timer = s["timer"] + enc[:, 12] * dt
    act = act_logits.argmax(-1)
    cols = [*CL_POS, *CL_ROT, *CL_VEL, *CL_ANG, *CL_ACT, CL_TIMER]
    vals = np.concatenate([pos, rot, vel, ang, np.stack([(act == 0), (act == 1)], 1).astype(float),
                           timer[:, None]], 1)
    return cols, vals


# ---- chunk hidden state (ChunkState u, θ, v, ω in the body frame), the body derivation's input
HIDDEN_COLS = tuple(f"def_{g}_body_{i}" for g in ("u", "rot", "v", "w") for i in range(3))
HIDDEN_NAMES = tuple(f"{g}.{a}" for g in ("u", "th", "v", "w") for a in "xyz")


def hidden_scale(c) -> np.ndarray:
    """[Nc, 12] divisors of (u, θ, v, ω) giving the chunk_feat def_* encodings: u_c, u_c/R, v,
    v/R (R = |half extents|)."""
    from . import schema as S
    a = c.arrays
    L, v = float(c.scales["L"]), float(c.scales["v"])
    he = a["chunk_feat"][:, [S.col("chunk_feat", f"half_extent_{i}") for i in range(3)]].astype(np.float64) * L
    R = np.maximum(np.linalg.norm(he, axis=1), 1e-12)
    uc = np.maximum(a["chunk_def_uc"].astype(np.float64), 1e-12)
    rep = lambda x: np.repeat(x[:, None], 3, 1)  # noqa: E731
    return np.concatenate([rep(uc), rep(uc / R), rep(np.full_like(R, v)), rep(v / R)], 1)
