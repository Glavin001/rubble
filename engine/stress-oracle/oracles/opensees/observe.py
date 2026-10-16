"""Raw OpenSees responses -> `stress-observation/1` (probes, bodies, flags, values, notes).

Sign conventions follow CONTRACT.md / the reference solver's `probe_value`:

* `section_force`: for every crossing element, the force the element exerts on the
  node on the `+normal` side is `-R` (R = element resisting force at that node, as
  returned by `ops.eleForce`), its moment about `point` is `-(M + (x - point) x R)`.
  No distributed loads act on the elements (masses are lumped at nodes), so this is
  the internal force at the bond centroid. `normal` = tension positive = `R . n`.
* `chunk_displacement` / `chunk_velocity`: node displacement from the undeformed
  position (the static deflection appears at t = 0), plus the rigid offset
  `theta x r` for refined (level > 0) chunks.
* `reaction`: `ops.nodeReaction` (= force the support exerts on the structure).

Bond failure (joint.rs `stress_measures` + `failure_indices`, strengths x Weibull)
is evaluated for every live element at every analysis step from the generalised bond
force at the bond centroid: `q.lin = (R_j . t1, R_j . t2, R_j . n)`,
`q.ang = m . (t1, t2, n)` with `m = M_j + (x_j - c) x R_j`.
"""

from __future__ import annotations

import numpy as np


def failure_indices(fd, lin, ang):
    """Vectorised mirror of joint.rs `stress_measures` + `failure_indices` (multiplier = Weibull).

    `lin` = (V_t1, V_t2, N) and `ang` = (M_t1, M_t2, T), tension positive, shape (K, 3).
    """
    area = fd["area"]
    axial = lin[:, 2] / area
    bending = np.abs(ang[:, 0]) / fd["s_t1"] + np.abs(ang[:, 1]) / fd["s_t2"]
    shear = np.hypot(lin[:, 0], lin[:, 1]) / area + np.abs(ang[:, 2]) / fd["torsion_modulus"]
    tension = axial + bending
    normal_compression = np.maximum(-axial, 0.0)
    compression = -axial + bending
    compressive_force = np.maximum(-lin[:, 2], 0.0)
    mult = fd["weibull"]
    ft = fd["tensile"] * mult
    fc = fd["compressive"] * mult
    cap = fd["shear_cap"] * mult if fd["shear_cap"] is not None else np.inf
    tau_max = np.minimum(fd["cohesion"] * mult + fd["friction"] * normal_compression, cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        i_t = np.maximum(tension / ft, 0.0)
        i_s = np.where(tau_max > 0, shear / tau_max, np.inf)
        i_c = np.maximum(compression / fc, 0.0)
        i_b = compressive_force / fd["buckling_load"] if fd["buckling_load"] else np.zeros_like(i_t)
    stack = np.stack([i_t, i_s, i_c, i_b], axis=1)
    nan = np.isnan(lin[:, 2])
    mx = np.where(nan, np.nan, np.max(np.where(np.isnan(stack), -1.0, stack), axis=1))
    names = np.array(["tension", "shear", "compression", "buckling"], dtype=object)
    mode = np.where(nan, "", names[np.argmax(np.where(np.isnan(stack), -1.0, stack), axis=1)])
    return {"max": mx, "mode": mode, "tension": i_t, "shear": i_s, "compression": i_c, "buckling": i_b}


def _probe_series(model, raw, p):
    t = raw["t"]
    node_row = {int(tag): i for i, tag in enumerate(raw["node_tags"])}
    ele_row = {int(tag): i for i, tag in enumerate(raw["ele_tags"])}
    coords = {nd["tag"]: np.array(nd["coord"]) for nd in model["nodes"]}
    kind = p["type"]
    if kind in ("chunk_displacement", "chunk_velocity"):
        arr = raw["disp"] if kind == "chunk_displacement" else raw["vel"]
        i = node_row[p["node"]]
        u = arr[:, i, :3] + np.cross(arr[:, i, 3:], np.array(p["offset"]))
        return u @ np.array(p["axis"])
    if kind == "section_force":
        pt = np.array(p["point"])
        n = np.array(p["normal"])
        F = np.zeros((len(t), 3))
        M = np.zeros((len(t), 3))
        for c in p["crossing"]:
            f = raw["ele_force"][:, ele_row[c["element"]], :]
            r = f[:, 0:6] if c["plus_end"] == "i" else f[:, 6:12]
            x = np.array(c["plus_coord"])
            F += -r[:, :3]
            M += -(r[:, 3:] + np.cross(x - pt, r[:, :3]))
        comp = p["component"]
        if comp["kind"] == "normal":
            return -(F @ n)
        if comp["kind"] == "force":
            return F @ np.array(comp["axis"])
        return M @ np.array(comp["axis"])
    if kind == "reaction":
        a = np.array(p["axis"])
        out = np.zeros(len(t))
        for nd in p["nodes"]:
            out += raw["reaction"][:, node_row[nd], :3] @ a
        return out
    raise ValueError(kind)


def _sample(t, v, interval, end):
    """Samples on `k * interval` (exact grid points of `t`) with lo/hi envelopes."""
    n = int(round(end / interval))
    ts, vs, lo, hi = [], [], [], []
    prev = 0
    for k in range(n + 1):
        target = k * interval
        i = int(np.argmin(np.abs(t - target)))
        if abs(t[i] - target) > 1e-9 * max(interval, 1.0):
            raise RuntimeError(f"sample time {target} not on the analysis grid")
        seg = v[prev:i + 1] if k > 0 else v[i:i + 1]
        seg = seg[~np.isnan(seg)]
        if not np.isfinite(v[i]):
            continue
        ts.append(float(t[i]))
        vs.append(float(v[i]))
        lo.append(float(seg.min()) if seg.size else float(v[i]))
        hi.append(float(seg.max()) if seg.size else float(v[i]))
        prev = i
    return {"t": ts, "v": vs, "lo": lo, "hi": hi}


def failure_history(model, raw):
    """Max failure index per step (and its element / mode) over all live elements."""
    ele_row = {int(tag): i for i, tag in enumerate(raw["ele_tags"])}
    node_coord = {nd["tag"]: np.array(nd["coord"]) for nd in model["nodes"]}
    el_by_tag = {e["tag"]: e for e in model["elements"]}
    K = len(raw["t"])
    best = np.zeros(K)
    best_el = np.full(K, -1)
    best_mode = np.full(K, "", dtype=object)
    per_element_max = {}
    for fd in model["failure"]:
        tag = fd["element"]
        el = el_by_tag[tag]
        f = raw["ele_force"][:, ele_row[tag], :]
        Rj, Mj = f[:, 6:9], f[:, 9:12]
        c = np.array(fd["centroid"])
        m = Mj + np.cross(node_coord[el["j"]] - c, Rj)
        t1, t2, n = (np.array(fd[k]) for k in ("t1", "t2", "n"))
        lin = np.stack([Rj @ t1, Rj @ t2, Rj @ n], axis=1)
        ang = np.stack([m @ t1, m @ t2, m @ n], axis=1)
        idx = failure_indices(fd, lin, ang)
        live = np.isfinite(idx["max"])
        per_element_max[tag] = float(np.nanmax(idx["max"])) if live.any() else float("nan")
        val = np.where(live, idx["max"], -1.0)
        upd = val > best
        best = np.where(upd, val, best)
        best_el = np.where(upd, tag, best_el)
        best_mode = np.where(upd, idx["mode"], best_mode)
    return best, best_el, best_mode, per_element_max


def observe(scene, model, raw, version, seed, extra_notes=()):
    a = model["analysis"]
    t = raw["t"]
    obs = {
        "format": "stress-observation/1",
        "scene": scene["name"],
        "solver": "opensees",
        "solver_version": version,
        "seed": seed,
        "end_time": float(t[-1]),
        "probes": {},
        "bodies": {},
        "flags": {},
        "values": {},
        "notes": [],
    }
    for p in model["probes"]:
        v = _probe_series(model, raw, p)
        obs["probes"][p["name"]] = _sample(t, v, a["sample_interval"], a["duration"])

    # Bodies: final node state; elastic, nothing detaches.
    node_row = {int(tag): i for i, tag in enumerate(raw["node_tags"])}
    for b in model["bodies"]:
        out = []
        for ci in range(b["chunks"]):
            nd = b["chunk_node"][ci]
            if b["chunk_removed_by_event"][ci] is not None:
                out.append({"detached": False, "removed": True, "fragment": -1, "velocity": [0.0] * 3, "displacement": [0.0] * 3})
                continue
            i = node_row[nd]
            r = np.array(b["chunk_offset"][ci])
            u = raw["disp"][-1, i]
            v = raw["vel"][-1, i]
            out.append({
                "detached": False, "removed": False, "fragment": 0,
                "velocity": (v[:3] + np.cross(v[3:], r)).tolist(),
                "displacement": (u[:3] + np.cross(u[3:], r)).tolist(),
            })
        obs["bodies"][b["name"]] = out

    best, best_el, best_mode, per_el = failure_history(model, raw)
    kmax = int(np.argmax(best))
    broken = bool(best.max() >= 1.0)
    # Elastic analysis: damage onset (index >= 1) is all we can detect, so both flags
    # mean "some bond reached its failure surface"; nothing ever disconnects here.
    obs["flags"]["any_bond_cracked"] = broken
    obs["flags"]["any_bond_broken"] = broken
    obs["flags"]["collapse"] = False
    obs["values"]["max_failure_index"] = float(best.max())
    obs["values"]["max_failure_index_time"] = float(t[kmax])
    el_by_tag = {e["tag"]: e for e in model["elements"]}
    over = [tag for tag, x in per_el.items() if x >= 1.0]
    obs["values"]["bonds_over_strength"] = float(len(over))
    if broken:
        k1 = int(np.argmax(best >= 1.0))
        obs["values"]["first_failure_time"] = float(t[k1])
        obs["values"]["first_crack_time"] = float(t[k1])
        e1 = el_by_tag[int(best_el[k1])]
        obs["notes"].append(
            f"first failure index >= 1 at t={t[k1]:.6f} s: bond {e1['bond']} (chunks {_chunk_pair(model, e1)}), "
            f"mode {best_mode[k1]}, index {best[k1]:.3f}; analysis continues elastically (no damage)")
    em = el_by_tag[int(best_el[kmax])]
    obs["notes"].append(
        f"max failure index {best.max():.4f} at t={t[kmax]:.6f} s in bond {em['bond']} (chunks {_chunk_pair(model, em)}), mode {best_mode[kmax]}")
    if over:
        obs["notes"].append(f"{len(over)} bonds reach index >= 1 at some time: " +
                            ", ".join(str(el_by_tag[x]['bond']) for x in sorted(over)[:40]) + (" ..." if len(over) > 40 else ""))
    for i, f in enumerate(raw["eigen_hz"], start=1):
        obs["values"][f"eigen_f{i}"] = float(f)
    obs["notes"] += list(model["notes"]) + list(extra_notes)
    return obs


def _chunk_pair(model, el):
    nd = {n["tag"]: n["chunk"] for n in model["nodes"]}
    return f"{nd[el['i']]}-{nd[el['j']]}"
