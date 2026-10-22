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
    return block("fs_chunk_in").astype(np.float32), block("fs_chunk_out").astype(np.float32)
