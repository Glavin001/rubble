"""Reference Python of the energy envelope (`stress_learned::kin_envelope`, the decode's hard
constraint): bit-for-bit with the Rust on `runs/fixtures/kin_envelope_v1.json` (242 calls of
P1's calls-v5). Scalar expressions in a fixed order; no BLAS matrix products (their summation
order is not fixed). See the Rust module's docs for the definition."""
from __future__ import annotations

import math

import numpy as np

TOL = 0.05


def _ke(m, I, q, v, w):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    qw, qx, qy, qz = q[0] / n, q[1] / n, q[2] / n, q[3] / n
    r = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
         [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
         [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]]
    iw = [[r[a][0] * I[0] * r[b][0] + r[a][1] * I[1] * r[b][1] + r[a][2] * I[2] * r[b][2] for b in range(3)] for a in range(3)]
    iww = [iw[a][0] * w[0] + iw[a][1] * w[1] + iw[a][2] * w[2] for a in range(3)]
    return 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 0.5 * (w[0] * iww[0] + w[1] * iww[1] + w[2] * iww[2])


def envelope(manifest: dict, arrays: dict, dt: float):
    """(E, chunk radii [(dx, drot) or None for anchored], impactor radii [(dx, drot)])."""
    g = math.sqrt(sum(float(x) ** 2 for x in manifest.get("gravity", [0, 0, 0])))
    cols = manifest["full_state_columns"]["bond"]
    E = 0.0
    rows = []
    for pre in ("chunk", "impactor"):
        m = np.asarray(arrays[f"{pre}_mass"], np.float64)
        I = np.asarray(arrays[f"{pre}_inertia_body"], np.float64)
        q = np.asarray(arrays[f"{pre}_q_world"], np.float64)
        v = np.asarray(arrays[f"{pre}_v"], np.float64)
        w = np.asarray(arrays[f"{pre}_w"], np.float64)
        anch = np.asarray(arrays["chunk_anchored"]) if pre == "chunk" else np.zeros(len(m))
        for k in range(len(m)):
            E += _ke(float(m[k]), [float(x) for x in I[k]], [float(x) for x in q[k]], [float(x) for x in v[k]], [float(x) for x in w[k]])
            rows.append((pre, float(m[k]), min(float(I[k][0]), float(I[k][1]), float(I[k][2])), bool(anch[k])))
    for x in np.asarray(arrays["fs_bond_in"], np.float64)[:, cols.index("stored")]:
        if math.isfinite(x):
            E += float(x)
    Eb = (1.0 + TOL) * E
    chunk, imp = [], []
    for pre, m, imin, anchored in rows:
        r = (dt * (math.sqrt(2.0 * Eb / m) + g * dt), dt * math.sqrt(2.0 * Eb / imin))
        if pre == "chunk":
            chunk.append(None if anchored else r)
        else:
            imp.append(r)
    return E, chunk, imp


def clamp(d, r):
    """Project the 3-vector `d` radially onto the ball of radius `r`; (d, clamped)."""
    n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if n > r:
        s = r / n
        return [d[0] * s, d[1] * s, d[2] * s], True
    return list(d), False
