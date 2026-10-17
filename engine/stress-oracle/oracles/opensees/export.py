"""Scene (`stress-scene/1`) -> OpenSees frame model.

`export_model(scene)` is a pure function: it reads only the scene dict and returns a
JSON-serialisable model description (the "input deck"); `build(model, ops)` turns that
description into an OpenSees domain. Nothing here runs an analysis.

Model (one frame model per scene body, body frame = model frame):

* one node per level-0 chunk at its centre, 6 DOFs, lumped mass `m` and the box's
  rotary inertia about its centre (density of the chunk material);
* one `elasticBeamColumn` per level-0 bond between the two chunk nodes with
  `E`, `G = E / (2 (1 + nu))` of the bond material (times `sim.stiffness_scale`),
  `A` = bond area, `Iz = derived.i_t1`, `Iy = derived.i_t2`, `J = derived.torsion_constant`;
  `geomTransf Linear` with `vecxz = tangent`, so local x = centre a -> centre b
  (= bond normal), local z = t1, local y = -t2;
* `fix` all 6 DOFs of `support: fixed` chunks, translations of `pinned` chunks;
* gravity as nodal loads `m g` (Constant series), point forces as nodal force + moment
  about the chunk centre with a `Path` series reproducing the scene's TimeFunction;
* per-element stiffness-proportional damping `beta_e = 2 zeta / omega_e`, `omega_e =
  sqrt(kn / m_red)` (the reference solver's bond dashpots: `zeta` at each bond's own
  axial frequency), applied with `region ... -rayleigh 0 0 beta_e 0`;
* `remove_chunks` events: the elements touching removed chunks and the removed nodes
  are deleted at `time`; the runner measures the interface forces the deleted
  elements exert on the kept nodes and re-applies them as nodal loads ramping
  linearly to zero over `duration` (none for `duration = 0`).
"""

from __future__ import annotations

import math

MODEL_FORMAT = "opensees-frame-model/1"

# ----------------------------------------------------------------------------- vectors


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return [a[0] / n, a[1] / n, a[2] / n]


def _scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _is_identity(q, tol=1e-12):
    return abs(abs(q[0]) - 1.0) < tol and all(abs(x) < tol for x in q[1:])


class UnsupportedScene(ValueError):
    """The scene uses a feature this oracle cannot represent (never silently dropped)."""


# ----------------------------------------------------------------------------- time functions


def time_function_path(tf, duration, dt):
    """A TimeFunction as `(times, values)` for an OpenSees `Path` series, or `("constant", v)`.

    OpenSees' Path returns 0 outside `[times[0], times[-1]]`, so the path is padded
    explicitly to cover `[-1, duration + 1]` with the scene's clamping rules.
    Curved functions are sampled on the analysis grid (Newmark only evaluates loads at
    step times, so this is exact at every step).
    """
    kind = tf["type"]
    end = duration + 1.0
    if kind == "constant":
        return ("constant", tf["value"])
    if kind == "ramp":
        t0, t1, v = tf["t0"], tf["t1"], tf["value"]
        if t1 <= t0:
            return ([min(-1.0, t0 - 1.0), t0, t0 + 1e-12, max(end, t0 + 1.0)], [0.0, 0.0, v, v])
        return ([min(-1.0, t0 - 1.0), t0, t1, max(end, t1 + 1.0)], [0.0, 0.0, v, v])
    if kind == "table":
        pts = tf["points"]
        if not pts:
            return ("constant", 0.0)
        times = [min(-1.0, pts[0][0] - 1.0)] + [p[0] for p in pts] + [max(end, pts[-1][0] + 1.0)]
        values = [pts[0][1]] + [p[1] for p in pts] + [pts[-1][1]]
        return (times, values)
    if kind in ("half_sine", "friedlander"):
        if kind == "half_sine":
            start, length = tf["start"], tf["duration"]

            def f(t):
                return tf["peak"] * math.sin(math.pi * (t - start) / length) if start <= t <= start + length else 0.0
        else:
            start, length = tf["arrival"], tf["duration"]

            def f(t):
                s = (t - start) / length
                return tf["peak"] * (1.0 - s) * math.exp(-tf["decay"] * s) if 0.0 <= s <= 1.0 else 0.0
        n = max(2, int(math.ceil(length / dt)))
        ts = [start + length * i / n for i in range(n + 1)]
        times = [min(-1.0, start - 1.0), start - 1e-12] + ts + [start + length + 1e-12, max(end, start + length + 1.0)]
        values = [0.0, 0.0] + [f(t) for t in ts] + [0.0, 0.0]
        return (times, values)
    raise UnsupportedScene(f"time function '{kind}'")


def eval_time_function(tf, t):
    """Scene TimeFunction value (mirror of `TimeFunction::eval`)."""
    kind = tf["type"]
    if kind == "constant":
        return tf["value"]
    if kind == "ramp":
        if t <= tf["t0"]:
            return 0.0
        if t >= tf["t1"]:
            return tf["value"]
        return tf["value"] * (t - tf["t0"]) / (tf["t1"] - tf["t0"])
    if kind == "table":
        pts = tf["points"]
        if not pts:
            return 0.0
        if t <= pts[0][0]:
            return pts[0][1]
        for a, b in zip(pts, pts[1:]):
            if t <= b[0]:
                f = (t - a[0]) / max(b[0] - a[0], 1e-300)
                return a[1] + f * (b[1] - a[1])
        return pts[-1][1]
    if kind == "half_sine":
        if t < tf["start"] or t > tf["start"] + tf["duration"]:
            return 0.0
        return tf["peak"] * math.sin(math.pi * (t - tf["start"]) / tf["duration"])
    if kind == "friedlander":
        s = (t - tf["arrival"]) / tf["duration"]
        if not 0.0 <= s <= 1.0:
            return 0.0
        return tf["peak"] * (1.0 - s) * math.exp(-tf["decay"] * s)
    raise UnsupportedScene(f"time function '{kind}'")


# ----------------------------------------------------------------------------- selectors


def select_chunks(selector, body):
    chunks = body["chunks"]
    if selector == "all":
        return list(range(len(chunks)))
    if "group" in selector:
        return [i for i, c in enumerate(chunks) if selector["group"] in c.get("groups", [])]
    if "indices" in selector:
        return list(selector["indices"])
    if "region" in selector:
        r = selector["region"]
        return [i for i, c in enumerate(chunks) if _in_region(c["center"], r)]
    raise UnsupportedScene(f"chunk selector {selector}")


def _in_region(p, r):
    return all(r["min"][k] <= p[k] <= r["max"][k] for k in range(3))


def _with_descendants(sel, body):
    chosen = set(sel)
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(body["chunks"]):
            if i not in chosen and c.get("parent") is not None and c["parent"] in chosen:
                chosen.add(i)
                changed = True
    return sorted(chosen)


def _coarse_ancestor(body, i):
    c = body["chunks"][i]
    while c.get("level", 0) > 0:
        i = c["parent"]
        c = body["chunks"][i]
    return i


# ----------------------------------------------------------------------------- export


def analysis_dt(scene):
    """Newmark step: the probe sample interval divided by `substeps` (default 4), <= frame_dt."""
    sim = scene["sim"]
    hint = scene.get("oracle", {}).get("opensees", {}) or {}
    sample = sim.get("sample_interval") or sim.get("frame_dt", 1.0 / 60.0)
    substeps = int(hint.get("substeps", 4))
    dt = min(sample, sim.get("frame_dt", sample)) / substeps
    return dt, sample, substeps


def export_model(scene):
    """Pure function: scene dict (with `derived` bond data) -> model dict."""
    if scene.get("format") != "stress-scene/1":
        raise UnsupportedScene(f"scene format {scene.get('format')}")
    if scene.get("impactors"):
        raise UnsupportedScene("impactors (contact) are not modelled by the frame oracle")
    if scene.get("ground"):
        raise UnsupportedScene("ground contact is not modelled by the frame oracle")
    sim = scene["sim"]
    mats = scene["materials"]
    scale = sim.get("stiffness_scale", 1.0)
    gravity = scene.get("gravity", [0.0, 0.0, -9.81])
    dt, sample, substeps = analysis_dt(scene)
    duration = sim["duration"]
    notes = []

    nodes, elements, patterns, events, probes, failure = [], [], [], [], [], []
    node_of = {}  # (body index, chunk index) -> node tag
    elem_of = {}  # (body index, bond index) -> element tag

    for bi, body in enumerate(scene["bodies"]):
        if not _is_identity(body.get("orientation", [1, 0, 0, 0])):
            raise UnsupportedScene(f"body '{body['name']}': rotated bodies are not supported")
        if any(abs(x) > 0 for x in body.get("linear_velocity", [0, 0, 0]) + body.get("angular_velocity", [0, 0, 0])):
            raise UnsupportedScene(f"body '{body['name']}': initial velocity is not supported")
        for ci, c in enumerate(body["chunks"]):
            if c.get("level", 0) != 0:
                continue
            if not _is_identity(c.get("orientation", [1, 0, 0, 0])):
                raise UnsupportedScene(f"chunk {ci}: rotated chunks are not supported (diagonal nodal inertia)")
            h = c["half_extents"]
            rho = mats[c["material"]]["density"]
            m = rho * 8.0 * h[0] * h[1] * h[2]
            a, b, cc = 2 * h[0], 2 * h[1], 2 * h[2]
            inertia = [m * (b * b + cc * cc) / 12.0, m * (a * a + cc * cc) / 12.0, m * (a * a + b * b) / 12.0]
            support = c.get("support", "none")
            fix = [1] * 6 if support == "fixed" else ([1, 1, 1, 0, 0, 0] if support == "pinned" else [0] * 6)
            tag = len(nodes) + 1
            node_of[(bi, ci)] = tag
            nodes.append({"tag": tag, "body": bi, "chunk": ci, "coord": list(c["center"]), "mass": [m, m, m] + inertia, "fix": fix})

        for bj, bond in enumerate(body["bonds"]):
            if bond.get("level", 0) != 0:
                continue
            if bond.get("rebar"):
                raise UnsupportedScene("rebar is not modelled by the elastic frame oracle")
            d = bond.get("derived")
            if d is None:
                raise UnsupportedScene("bond without `derived` data (generate scenes with `stress-ref gen-scenes`)")
            mat = mats[bond["material"]]
            e_mod = mat["youngs_modulus"] * scale
            g_mod = e_mod / (2.0 * (1.0 + mat["poisson_ratio"]))
            na, nb = node_of[(bi, bond["a"])], node_of[(bi, bond["b"])]
            xa, xb = body["chunks"][bond["a"]]["center"], body["chunks"][bond["b"]]["center"]
            axis = _sub(xb, xa)
            n = _unit(bond["normal"])
            t1 = _unit(_sub(bond["tangent"], _scale(n, _dot(bond["tangent"], n))))
            t2 = _cross(n, t1)
            misalign = 1.0 - abs(_dot(_unit(axis), n))
            if misalign > 1e-9:
                notes.append(f"bond {bj}: centre line deviates from the bond normal (1-cos = {misalign:.2e}); element axis follows the centre line")
            length = _norm(axis)
            # Reference-solver dashpot: zeta at the bond's axial frequency sqrt(kn / m_red).
            ma = nodes[na - 1]["mass"][0]
            mb = nodes[nb - 1]["mass"][0]
            fa, fb = any(nodes[na - 1]["fix"][:3]), any(nodes[nb - 1]["fix"][:3])
            m_red = min(ma, mb) if (fa and fb) else (mb if fa else (ma if fb else ma * mb / (ma + mb)))
            kn = e_mod * bond["area"] / d["length"]
            zeta = mat.get("damping_ratio", 0.01)
            beta = 2.0 * zeta / math.sqrt(kn / m_red) if zeta > 0 else 0.0
            tag = len(elements) + 1
            elem_of[(bi, bj)] = tag
            elements.append({
                "tag": tag, "body": bi, "bond": bj, "i": na, "j": nb,
                "A": bond["area"], "E": e_mod, "G": g_mod, "J": d["torsion_constant"],
                "Iy": d["i_t2"], "Iz": d["i_t1"], "vecxz": t1, "length": length, "beta": beta,
            })
            # Failure surface of the bond (joint.rs), evaluated at the bond centroid.
            w0, w1 = bond["width"]
            long, short = max(w0, w1), min(w0, w1)
            ratio = short / long
            # bond.rs: max shear stress = T / W, W = long short^2 / (3 + 1.8 short/long).
            torsion_modulus = d.get("torsion_modulus", long * short * short / (3.0 + 1.8 * ratio))
            bl = bond.get("buckling_length")
            i_min = min(d["i_t1"], d["i_t2"])
            failure.append({
                "element": tag, "centroid": list(bond["centroid"]), "n": n, "t1": t1, "t2": t2,
                "area": bond["area"], "s_t1": d["section_modulus_t1"], "s_t2": d["section_modulus_t2"],
                "torsion_modulus": torsion_modulus,
                "tensile": mat["tensile_strength"], "compressive": mat["compressive_strength"],
                "cohesion": mat["cohesion"], "friction": mat["friction"],
                "shear_cap": mat.get("shear_cap"),
                "buckling_load": (math.pi ** 2 * mat["youngs_modulus"] * i_min / bl ** 2) if bl else None,
                "weibull": d["weibull"],
            })
            if mat.get("dif") or mat.get("static_fatigue"):
                notes.append(f"material '{bond['material']}': DIF / static-fatigue strength changes are not applied (static strengths x Weibull)")

    # ---- loads
    def body_index(name):
        for i, b in enumerate(scene["bodies"]):
            if b["name"] == name:
                return i
        raise UnsupportedScene(f"unknown body {name}")

    # Chunks removed by events: their loads go to separate patterns removed with them.
    removed_by = {}  # node tag -> event index
    for ei, ev in enumerate(scene.get("events", [])):
        if ev["type"] != "remove_chunks":
            raise UnsupportedScene(f"event '{ev['type']}'")
        bi = body_index(ev["body"])
        sel = _with_descendants(select_chunks(ev["chunks"], scene["bodies"][bi]), scene["bodies"][bi])
        for ci in sel:
            if (bi, ci) in node_of:
                removed_by[node_of[(bi, ci)]] = ei

    def add_load(key, series, node, vec):
        grp = removed_by.get(node)
        k = (key, grp)
        for p in patterns:
            if p["key"] == k:
                p["loads"].append({"node": node, "values": vec})
                return
        patterns.append({"key": k, "tag": len(patterns) + 1, "series": series, "removed_by_event": grp,
                         "loads": [{"node": node, "values": vec}]})

    gnorm = _norm(gravity)
    if gnorm > 0:
        for nd in nodes:
            m = nd["mass"][0]  # on held nodes it goes to the reaction, as in the reference solver
            add_load("gravity", {"type": "constant", "value": 1.0}, nd["tag"], [m * gravity[0], m * gravity[1], m * gravity[2], 0.0, 0.0, 0.0])
    for li, load in enumerate(scene.get("loads", [])):
        if load["type"] != "point_force":
            raise UnsupportedScene(f"load '{load['type']}' is not supported by the frame oracle")
        bi = body_index(load["body"])
        body = scene["bodies"][bi]
        ci = _coarse_ancestor(body, load["chunk"])
        node = node_of[(bi, ci)]
        dirn = _unit(load["direction"])
        center = body["chunks"][ci]["center"]
        point = load.get("point") or body["chunks"][load["chunk"]]["center"]
        mom = _cross(_sub(point, center), dirn)
        path = time_function_path(load["magnitude"], duration, dt)
        series = {"type": "constant", "value": path[1]} if path[0] == "constant" else {"type": "path", "time": path[0], "values": path[1]}
        add_load(f"load{li}", series, node, dirn + mom)
    for p in patterns:
        p["key"] = list(p["key"])

    # ---- events
    for ei, ev in enumerate(scene.get("events", [])):
        bi = body_index(ev["body"])
        removed_nodes = sorted(t for t, e in removed_by.items() if e == ei)
        rset = set(removed_nodes)
        removed_elems, interface = [], []
        for el in elements:
            ri, rj = el["i"] in rset, el["j"] in rset
            if ri or rj:
                removed_elems.append(el["tag"])
                if ri != rj:
                    interface.append({"element": el["tag"], "kept_node": el["j"] if ri else el["i"], "kept_end": "j" if ri else "i"})
        events.append({"time": ev["time"], "duration": ev["duration"], "removed_nodes": removed_nodes,
                       "removed_elements": removed_elems, "interface": interface})

    # ---- probes
    elem_by_tag = {e["tag"]: e for e in elements}
    for p in scene.get("probes", []):
        kind = p["type"]
        entry = {"name": p["name"], "type": kind}
        if kind in ("chunk_displacement", "chunk_velocity"):
            bi = body_index(p["body"])
            body = scene["bodies"][bi]
            ci = _coarse_ancestor(body, p["chunk"])
            entry.update(node=node_of[(bi, ci)], axis=_unit(p["axis"]),
                         offset=_sub(body["chunks"][p["chunk"]]["center"], body["chunks"][ci]["center"]))
        elif kind == "section_force":
            bi = body_index(p["body"])
            body = scene["bodies"][bi]
            pt, nrm = p["point"], _unit(p["normal"])
            crossing = []
            for (b_, bj), tag in elem_of.items():
                if b_ != bi:
                    continue
                bond = body["bonds"][bj]
                if p.get("region") and not _in_region(bond["centroid"], p["region"]):
                    continue
                sa = _dot(_sub(body["chunks"][bond["a"]]["center"], pt), nrm)
                sb = _dot(_sub(body["chunks"][bond["b"]]["center"], pt), nrm)
                if sa * sb >= 0.0:
                    continue
                el = elem_by_tag[tag]
                plus_end = "i" if sa > 0 else "j"
                plus_node = el["i"] if plus_end == "i" else el["j"]
                crossing.append({"element": tag, "plus_end": plus_end, "plus_coord": nodes[plus_node - 1]["coord"]})
            comp = p["component"]
            entry.update(point=pt, normal=nrm, crossing=crossing)
            if comp == "normal":
                entry["component"] = {"kind": "normal"}
            elif "force" in comp:
                entry["component"] = {"kind": "force", "axis": _unit(comp["force"])}
            else:
                entry["component"] = {"kind": "moment", "axis": _unit(comp["moment"])}
            if not crossing:
                notes.append(f"probe '{p['name']}': no element crosses the section")
        elif kind == "reaction":
            bi = body_index(p["body"])
            body = scene["bodies"][bi]
            sel = select_chunks(p["chunks"], body)
            entry.update(nodes=[node_of[(bi, c)] for c in sel if (bi, c) in node_of and any(nodes[node_of[(bi, c)] - 1]["fix"])],
                         axis=_unit(p["axis"]))
        else:
            notes.append(f"probe '{p['name']}' ({kind}) is not measurable by the frame oracle; omitted")
            continue
        probes.append(entry)

    return {
        "format": MODEL_FORMAT,
        "scene": scene["name"],
        "seed": sim.get("seed", 0),
        "ndm": 3, "ndf": 6,
        "bodies": [{"name": b["name"], "chunks": len(b["chunks"]),
                    "chunk_node": [node_of.get((bi, _coarse_ancestor(b, ci))) for ci in range(len(b["chunks"]))],
                    "chunk_offset": [_sub(b["chunks"][ci]["center"], b["chunks"][_coarse_ancestor(b, ci)]["center"]) for ci in range(len(b["chunks"]))],
                    "chunk_removed_by_event": [removed_by.get(node_of.get((bi, _coarse_ancestor(b, ci)))) for ci in range(len(b["chunks"]))]}
                   for bi, b in enumerate(scene["bodies"])],
        "nodes": nodes,
        "elements": elements,
        "patterns": patterns,
        "events": events,
        "probes": probes,
        "failure": failure,
        "analysis": {
            "static_first": bool(sim.get("gravity_prestress", True)),
            "integrator": ["Newmark", 0.5, 0.25],
            "dt": dt, "substeps": substeps, "sample_interval": sample, "duration": duration,
            "eigen_modes": 6,
        },
        "stiffness_scale": scale,
        "notes": sorted(set(notes)),
    }


# ----------------------------------------------------------------------------- build


def build(model, ops):
    """Create the OpenSees domain for `model` (wipes the current one)."""
    ops.wipe()
    ops.model("basic", "-ndm", model["ndm"], "-ndf", model["ndf"])
    for nd in model["nodes"]:
        ops.node(nd["tag"], *nd["coord"])
        ops.mass(nd["tag"], *nd["mass"])
        if any(nd["fix"]):
            ops.fix(nd["tag"], *nd["fix"])
    for el in model["elements"]:
        ops.geomTransf("Linear", el["tag"], *el["vecxz"])
        ops.element("elasticBeamColumn", el["tag"], el["i"], el["j"], el["A"], el["E"], el["G"], el["J"], el["Iy"], el["Iz"], el["tag"])
    # Stiffness-proportional damping, one region per distinct beta.
    groups = {}
    for el in model["elements"]:
        if el["beta"] > 0:
            groups.setdefault(round(el["beta"], 15), []).append(el["tag"])
    for k, (beta, tags) in enumerate(sorted(groups.items()), start=1):
        ops.region(k, "-ele", *tags, "-rayleigh", 0.0, 0.0, beta, 0.0)
    for p in model["patterns"]:
        s = p["series"]
        if s["type"] == "constant":
            ops.timeSeries("Constant", p["tag"], "-factor", s["value"])
        else:
            ops.timeSeries("Path", p["tag"], "-time", *s["time"], "-values", *s["values"])
        ops.pattern("Plain", p["tag"], p["tag"])
        for ld in p["loads"]:
            ops.load(ld["node"], *ld["values"])
