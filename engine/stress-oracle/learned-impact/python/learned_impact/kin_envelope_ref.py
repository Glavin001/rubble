"""Reference Python of the energy envelope (`stress_learned::kin_envelope`, agreed with P2's
`decode.energy_envelope`): E = (energy.KE_in + energy.E_elastic_start) (1 + TOL), TOL = 2.0;
per free node r_x = dt (sqrt(2E/m) + |g| dt), r_r = dt sqrt(2E/I_min), I_min the smallest of
the row's *_inertia_body; radial clamp outside r (1 + 1e-9). Bit-for-bit with the Rust on
`runs/fixtures/kin_envelope_v2.json`. A call without E_elastic_start is unbounded (None)."""
from __future__ import annotations

import math

import numpy as np

TOL = 2.0


def envelope(manifest: dict, call):
    """(E or None, chunk radii [(dx, drot) or None], impactor radii [(dx, drot) or None])."""
    dt = call.scales["dt"]
    g = math.sqrt(sum(float(x) ** 2 for x in manifest.get("gravity", [0, 0, 0])))
    if "KE_in" not in call.energy or "E_elastic_start" not in call.energy:
        e = None
    else:
        e = (call.energy["KE_in"] + call.energy["E_elastic_start"]) * (1.0 + TOL)
    def radii(m, imin):
        return None if e is None else (dt * (math.sqrt(2.0 * e / m) + g * dt), dt * math.sqrt(2.0 * e / imin))
    a = call.arrays
    out = []
    for pre in ("chunk", "impactor"):
        m = np.asarray(a[f"{pre}_mass"], np.float64)
        I = np.asarray(a[f"{pre}_inertia_body"], np.float64)
        anch = np.asarray(a["chunk_anchored"]) if pre == "chunk" else np.zeros(len(m))
        out.append([None if anch[k] else radii(float(m[k]), min(float(I[k][0]), float(I[k][1]), float(I[k][2]))) for k in range(len(m))])
    return (None if e is None else e / (1.0 + TOL)), out[0], out[1]


def clamp(d, r):
    """Project the 3-vector `d` radially onto the ball of radius `r` when outside r (1 + 1e-9)."""
    n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    if n > r * (1.0 + 1e-9):
        s = r / n
        return [d[0] * s, d[1] * s, d[2] * s], True
    return list(d), False
