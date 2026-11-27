#!/usr/bin/env python3
"""External oracle for the thin-walled section calculator (engine_vehiclegen::thin_wall).

Local only, never a dependency of a build or a test. Reads the calculator's JSON (the
`sections` example of engine-softbody-lab) and computes the same sections with Python's
`sectionproperties` (a 2D finite-element section analysis on a triangle mesh of the solid
region), installed into a throwaway virtualenv under the gitignored engine/out/:

    python3 -m venv engine/out/sections/venv
    engine/out/sections/venv/bin/pip install sectionproperties
    cargo run --release -p engine-softbody-lab --example sections > engine/out/sections/ours.json
    engine/out/sections/venv/bin/python engine/tools/sections/oracle.py engine/out/sections/ours.json

Two references per section:
  * `sectionproperties` on the union of our wall pieces (the same region: the differences are
    its mesh and its own plastic-axis search, not geometry);
  * the classic thin-wall midline approximation (each wall a line of length L carrying t·L,
    corners counted twice or not at all, no t³ terms), to show the size of the O(t/b) terms
    that an exact region avoids.
Prints a table and writes engine/out/sections/oracle.json.
"""
import json
import sys

from shapely import set_precision
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import CompoundGeometry, Geometry


def sp_props(pieces):
    # in millimetres (sectionproperties' mesher is happier at unit-scale sizes), back to SI
    region = unary_union([Polygon([(1e3 * x, 1e3 * y) for x, y in p]) for p in pieces])
    # the pieces' shared edges leave float-level duplicate and collinear vertices, which
    # crash the mesher: snap to a 1e-7 mm grid and drop collinear points (exact otherwise)
    region = set_precision(region, 1e-7).simplify(0)
    polys = [region] if region.geom_type == "Polygon" else list(region.geoms)
    geoms = [Geometry(geom=p) for p in polys]
    geom = geoms[0] if len(geoms) == 1 else CompoundGeometry(geoms)
    area = region.area
    # mesh: a few elements through the wall (the smallest wall is ~1.2 mm; element area in m²)
    geom.create_mesh(mesh_sizes=[area / 4000.0])
    sec = Section(geometry=geom)
    sec.calculate_geometric_properties()
    sec.calculate_plastic_properties()
    ixx, iyy, ixy = sec.get_ic()
    sxx, syy = sec.get_s()
    zxxp, zxxm, zyyp, zyym = sec.get_z()
    cx, cy = sec.get_c()
    return {
        "area": sec.get_area() * 1e-6,
        "centroid": [cx * 1e-3, cy * 1e-3],
        "ixx": ixx * 1e-12,
        "iyy": iyy * 1e-12,
        "ixy": ixy * 1e-12,
        "wx": min(zxxp, zxxm) * 1e-9,
        "wy": min(zyyp, zyym) * 1e-9,
        "zx": sxx * 1e-9,
        "zy": syy * 1e-9,
        "elements": len(sec.elements),
    }


def midline(walls):
    """Thin-wall midline sums: area Σ t L, ∫y² and ∫x² along the lines (no corner or t³ terms),
    and the plastic modulus of the same lines (equal-length-weighted axis by bisection)."""
    segs = []
    for w in walls:
        pts = w["points"]
        n = len(pts)
        m = n if w["closed"] else n - 1
        for i in range(m):
            segs.append((pts[i], pts[(i + 1) % n], w["t"]))
    a = sum(t * ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5 for p, q, t in segs)

    def mom(f):
        tot = 0.0
        for p, q, t in segs:
            L = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
            tot += t * L * f(p, q)
        return tot

    cx = mom(lambda p, q: 0.5 * (p[0] + q[0])) / a
    cy = mom(lambda p, q: 0.5 * (p[1] + q[1])) / a
    sq = lambda u0, u1: (u0 * u0 + u0 * u1 + u1 * u1) / 3.0
    ixx = mom(lambda p, q: sq(p[1] - cy, q[1] - cy))
    iyy = mom(lambda p, q: sq(p[0] - cx, q[0] - cx))

    def zpl(axis):
        def above(s):
            tot = 0.0
            for p, q, t in segs:
                L = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
                u0, u1 = p[axis], q[axis]
                lo, hi = min(u0, u1), max(u0, u1)
                if hi - lo < 1e-15:
                    frac = 1.0 if lo > s else 0.0
                else:
                    frac = min(max((hi - s) / (hi - lo), 0.0), 1.0)
                tot += t * L * frac
            return tot
        lo = min(min(p[axis], q[axis]) for p, q, _ in segs)
        hi = max(max(p[axis], q[axis]) for p, q, _ in segs)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if above(mid) > 0.5 * a:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
        # ∫|u − s| t ds, each segment split at s, exact for a straight line
        tot = 0.0
        for p, q, t in segs:
            L = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
            u0, u1 = p[axis] - s, q[axis] - s
            if u0 * u1 >= 0:
                tot += t * L * abs(u0 + u1) / 2
            else:
                f = abs(u0) / (abs(u0) + abs(u1))
                tot += t * L * (f * abs(u0) / 2 + (1 - f) * abs(u1) / 2)
        return tot

    return {"area": a, "ixx": ixx, "iyy": iyy, "zx": zpl(1), "zy": zpl(0)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "engine/out/sections/ours.json"
    ours = json.load(open(path))
    keys = ["area", "ixx", "iyy", "wx", "wy", "zx", "zy"]
    out = []
    print(f"{'section':32} {'quantity':5} {'ours':>13} {'sectionprop':>13} {'diff':>10} {'midline':>13} {'diff':>9}")
    for s in ours:
        sp = sp_props(s["pieces"])
        ml = midline(s["walls"])
        row = {"name": s["name"], "ours": {k: s["properties"][k] for k in keys}, "sectionproperties": sp, "midline": ml}
        for k in keys:
            a, b = s["properties"][k], sp[k]
            m = ml.get(k)
            d_sp = (a - b) / b if b else float("nan")
            d_ml = "" if m is None else f"{(m - a) / a:+9.2e}"
            print(f"{s['name']:32} {k:5} {a:13.6e} {b:13.6e} {d_sp:+10.2e} {'' if m is None else f'{m:13.6e}':>13} {d_ml:>9}")
        out.append(row)
    dst = path.rsplit("/", 1)[0] + "/oracle.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
