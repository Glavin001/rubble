#!/usr/bin/env python3
"""Per-part mass, nodal (lumped) masses, CG and inertia of a parsed LS-DYNA deck.

    python3 -I engine/tools/fe-import/fe_mass.py CACHE_DIR

Reads CACHE_DIR/deck.pickle (from lsdyna_deck.py) and writes CACHE_DIR/mass.pickle:
per-part element mass, area/volume, centroid, bounding box, and the nodal mass arrays of
each mass layer (structure, added masses, fuel, dummies/cargo). Standard library only.

Mass rules (what LS-DYNA does for a lumped-mass explicit model):
- shell: A * t * rho (+ NSM per area from the section), A = |d13 x d24| / 2 (exact for a
  planar quad, the triangle area when n3 == n4); lumped equally to its distinct nodes;
- solid: volume of the trilinear hexahedron (24 tetrahedra through face centres and
  the element centre; degenerate tets/pentas handled by repeated nodes) * rho;
- beam: rho * A * L with A from the section (circular, diameter ts1, for the spot-weld
  formulation 9; tubular with inner diameter tt1 when set) + NSM * L;
- *ELEMENT_MASS: a point mass on its node; *ELEMENT_MASS_PART: ADDMASS spread over the
  part's nodes in proportion to their element mass (FINMASS > 0 sets the part total).
"""
import math
import os
import pickle
import sys
from array import array


def main():
    cache = sys.argv[1]
    d = pickle.load(open(os.path.join(cache, "deck.pickle"), "rb"))
    nid = d["node_id"]; X = d["xyz"]
    N = len(X) // 3
    parts, secs, mats = d["parts"], d["sections"], d["mats"]

    def rho(pid):
        p = parts.get(pid)
        if not p:
            return 0.0
        m = mats.get(p["mid"], {})
        return m.get("ro") or 0.0

    def P(n):
        i = 3 * nid[n]
        return X[i], X[i + 1], X[i + 2]

    struct = array("d", bytes(8 * N))   # nodal mass from elements
    # Mass-scaling estimate: *CONTROL_TIMESTEP DT2MS < 0 adds mass only to elements
    # whose step TSSFAC * Lc / c is below TSSFAC * |DT2MS| (LS-DYNA keyword manual).
    TSSFAC = 0.9
    DT_TARGET = TSSFAC * 0.778e-6
    ms = {}   # pid -> [added mass, n elements scaled]

    def wave(pid, kind):
        m = mats.get(parts[pid]["mid"], {})
        if m.get("type") in ("RIGID", "NULL") or not m.get("ro") or not m.get("e"):
            return None
        E, nu, r = m["e"], m.get("pr") or 0.0, m["ro"]
        if kind == "shell":
            return math.sqrt(E / (r * (1 - nu * nu)))
        if nu >= 0.5:
            return None
        return math.sqrt(E * (1 - nu) / (r * (1 + nu) * (1 - 2 * nu)))

    def ms_note(pid, m, dt):
        if dt < DT_TARGET:
            e = ms.setdefault(pid, [0.0, 0])
            e[0] += m * ((DT_TARGET / dt) ** 2 - 1); e[1] += 1
    pm = {}                              # pid -> stats

    def stat(pid):
        s = pm.get(pid)
        if s is None:
            s = pm[pid] = dict(n_shell=0, n_solid=0, n_beam=0, area=0.0, volume=0.0, length=0.0,
                               mass=0.0, mx=0.0, my=0.0, mz=0.0,
                               lo=[1e30] * 3, hi=[-1e30] * 3, nodes=set())
        return s

    # ---- shells
    sp, sn = d["shell_pid"], d["shell_n"]
    cache_t = {}
    for e in range(len(sp)):
        pid = sp[e]
        n1, n2, n3, n4 = sn[4 * e], sn[4 * e + 1], sn[4 * e + 2], sn[4 * e + 3]
        a, b, c, q = P(n1), P(n2), P(n3), P(n4)
        ux, uy, uz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        vx, vy, vz = q[0] - b[0], q[1] - b[1], q[2] - b[2]
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        tm = cache_t.get(pid)
        if tm is None:
            sec = secs.get(parts[pid]["secid"], {})
            tm = cache_t[pid] = (sec.get("t", 0.0), rho(pid), sec.get("marea", 0.0) or 0.0,
                                 wave(pid, "shell"))
        m = area * (tm[0] * tm[1] + tm[2])
        if tm[3] and area > 0:
            lens = [math.dist(a, b), math.dist(b, c)]
            if n3 == n4:
                lens.append(math.dist(c, a)); ls = 2 * area / max(lens)
            else:
                lens += [math.dist(c, q), math.dist(q, a)]; ls = area / max(lens)
            ms_note(pid, m, TSSFAC * ls / tm[3])
        s = stat(pid)
        s["n_shell"] += 1; s["area"] += area; s["mass"] += m
        ns = (n1, n2, n3) if n3 == n4 else (n1, n2, n3, n4)
        share = m / len(ns)
        for n in ns:
            i = nid[n]; struct[i] += share
            p = P(n)
            s["mx"] += share * p[0]; s["my"] += share * p[1]; s["mz"] += share * p[2]
            for k in range(3):
                if p[k] < s["lo"][k]: s["lo"][k] = p[k]
                if p[k] > s["hi"][k]: s["hi"][k] = p[k]
            s["nodes"].add(n)

    # ---- solids
    FACES = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    so, sn8 = d["solid_pid"], d["solid_n"]
    for e in range(len(so)):
        pid = so[e]
        ids = sn8[8 * e:8 * e + 8]
        pts = [P(n) for n in ids]
        cx = sum(p[0] for p in pts) / 8; cy = sum(p[1] for p in pts) / 8; cz = sum(p[2] for p in pts) / 8
        vol = 0.0
        for f in FACES:
            q = [pts[k] for k in f]
            fx = sum(p[0] for p in q) / 4; fy = sum(p[1] for p in q) / 4; fz = sum(p[2] for p in q) / 4
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                # tet (centre, face centre, a, b), outward-ordered faces give positive volume
                ax, ay, az = a[0] - cx, a[1] - cy, a[2] - cz
                bx, by, bz = b[0] - cx, b[1] - cy, b[2] - cz
                gx, gy, gz = fx - cx, fy - cy, fz - cz
                vol += (ax * (by * gz - bz * gy) - ay * (bx * gz - bz * gx) + az * (bx * gy - by * gx)) / 6.0
        vol = abs(vol)
        m = vol * rho(pid)
        cw = wave(pid, "solid")
        if cw and vol > 0:
            amax = 0.0
            for f in FACES:
                q = [pts[k] for k in f]
                ux, uy, uz = q[2][0] - q[0][0], q[2][1] - q[0][1], q[2][2] - q[0][2]
                vx, vy, vz = q[3][0] - q[1][0], q[3][1] - q[1][1], q[3][2] - q[1][2]
                amax = max(amax, 0.5 * math.sqrt((uy * vz - uz * vy) ** 2 + (uz * vx - ux * vz) ** 2 + (ux * vy - uy * vx) ** 2))
            if amax > 0:
                ms_note(pid, m, TSSFAC * vol / amax / cw)
        s = stat(pid)
        s["n_solid"] += 1; s["volume"] += vol; s["mass"] += m
        uniq = list(dict.fromkeys(ids))
        share = m / len(uniq)
        for n in uniq:
            i = nid[n]; struct[i] += share
            p = P(n)
            s["mx"] += share * p[0]; s["my"] += share * p[1]; s["mz"] += share * p[2]
            for k in range(3):
                if p[k] < s["lo"][k]: s["lo"][k] = p[k]
                if p[k] > s["hi"][k]: s["hi"][k] = p[k]
            s["nodes"].add(n)

    # ---- beams
    bp, bn = d["beam_pid"], d["beam_n"]
    for e in range(len(bp)):
        pid = bp[e]
        a, b = P(bn[2 * e]), P(bn[2 * e + 1])
        L = math.dist(a, b)
        sec = secs.get(parts[pid]["secid"], {})
        c2 = sec.get("card2") or []
        do = c2[0] if c2 and isinstance(c2[0], float) else 0.0
        di = c2[2] if len(c2) > 2 and isinstance(c2[2], float) else 0.0
        A = math.pi / 4 * (do * do - di * di)
        m = rho(pid) * A * L + (sec.get("nsm") or 0.0) * L
        s = stat(pid)
        s["n_beam"] += 1; s["length"] += L; s["mass"] += m
        for n, p in ((bn[2 * e], a), (bn[2 * e + 1], b)):
            struct[nid[n]] += m / 2
            s["mx"] += m / 2 * p[0]; s["my"] += m / 2 * p[1]; s["mz"] += m / 2 * p[2]
            s["nodes"].add(n)

    # ---- added masses, by layer
    layers = {k: array("d", bytes(8 * N)) for k in ("added_parts", "point_main", "fuel", "dummies_cargo")}
    layer_items = {k: [] for k in layers}
    set_file_payload_pids = set()
    for (eid, n, m, pid, src) in d["elmass"]:
        # main deck point masses (instrument/bookkeeping nodes) vs the set file's dummy masses
        key = "dummies_cargo" if eid >= 9200000 else "point_main"
        layers[key][nid[n]] += m
        layer_items[key].append((eid, n, m))
    FUEL = {2000609, 2000610}
    for (pid, add, fin, src) in d["mass_part"]:
        s = pm.get(pid)
        if not s:
            continue
        key = "fuel" if pid in FUEL else "added_parts"
        if abs(add - d["params"].get("M3_1", -1)) < 1e-12 and pid == 2000204:
            key = "dummies_cargo"
        target = fin - s["mass"] if fin > 0 else add
        tot = sum(struct[nid[n]] for n in s["nodes"])
        for n in s["nodes"]:
            i = nid[n]
            layers[key][i] += target * struct[i] / tot
        layer_items[key].append((pid, parts[pid]["title"], target))

    for s in pm.values():
        s["n_nodes"] = len(s["nodes"]); s["nodes"] = None
    out = dict(parts=pm, struct=struct, layers=layers, layer_items=layer_items, mass_scaling=ms)
    pickle.dump(out, open(os.path.join(cache, "mass.pickle"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    tot = sum(s["mass"] for p, s in pm.items() if parts[p]["source"] == "vehicle")
    print("vehicle element mass %.1f kg" % (tot * 1000))
    print("mass-scaling estimate %.2f kg over %d elements" % (sum(v[0] for v in ms.values()) * 1000, sum(v[1] for v in ms.values())))
    for k, a in layers.items():
        print(k, "%.1f kg" % (sum(a) * 1000))


if __name__ == "__main__":
    main()
