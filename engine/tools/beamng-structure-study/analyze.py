#!/usr/bin/env python3
"""Structure measurements of BeamNG cars for engine/docs/beamng-structure-study.md.

Input: the JSON files written by the `structure_dump` test of
`engine/crates/engine-softbody/tests/authoring_census.rs` (one per car, outside the
repository). Output: Markdown tables on stdout. Standard library only.

    python3 -I engine/tools/beamng-structure-study/analyze.py <dump dir> [car ...] [--only SECTION]

Sections: parts, attach, body, grade, brace, mech, wheelpath, glass, flex, stab, types,
panels, height, counts;
`--only summary` prints the cross-car tables instead.
Frame: JBeam (+X left, +Y rear, +Z up), metres, kg, N, N/m.
The output holds per-car detail (part and node-group names); the committed
document carries aggregates only.
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict

H = 1.0 / 2000.0
INF = 1e30

# ---------------------------------------------------------------- helpers


def q(v, p):
    v = sorted(x for x in v if x is not None and not math.isnan(x))
    if not v:
        return float("nan")
    i = (len(v) - 1) * p
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float) and x >= INF:
        return "∞"
    if isinstance(x, (int,)):
        return str(x)
    ax = abs(x)
    if ax >= 1e5:
        return f"{x:.2g}".replace("e+0", "e")
    if ax >= 100:
        return f"{x:.0f}"
    return f"{x:.{d}f}"


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def dist(a, b):
    return norm(sub(a, b))


def pt_tri(p, a, b, c):
    """Distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5)."""
    ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return norm(ap)
    bp = sub(p, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return dist(p, (a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]))
    cp = sub(p, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return dist(p, (a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return dist(p, (b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])))
    den = 1.0 / (va + vb + vc)
    v, w = vb * den, vc * den
    return dist(p, (a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w))


class TriGrid:
    """Triangles bucketed by the cells their box touches, for nearest-surface queries."""

    def __init__(self, pos, tris, cell=0.08):
        self.pos, self.tris, self.cell = pos, tris, cell
        self.g = defaultdict(list)
        for ti, t in enumerate(tris):
            ps = [pos[i] for i in t]
            lo = [int(math.floor(min(p[k] for p in ps) / cell)) for k in range(3)]
            hi = [int(math.floor(max(p[k] for p in ps) / cell)) for k in range(3)]
            if (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1) > 4000:
                continue
            for x in range(lo[0], hi[0] + 1):
                for y in range(lo[1], hi[1] + 1):
                    for z in range(lo[2], hi[2] + 1):
                        self.g[(x, y, z)].append(ti)

    def nearest(self, p, reach=6):
        c = [int(math.floor(p[k] / self.cell)) for k in range(3)]
        best = float("inf")
        for r in range(reach + 1):
            if best <= (r - 1) * self.cell:
                break
            seen = set()
            for x in range(c[0] - r, c[0] + r + 1):
                for y in range(c[1] - r, c[1] + r + 1):
                    for z in range(c[2] - r, c[2] + r + 1):
                        if max(abs(x - c[0]), abs(y - c[1]), abs(z - c[2])) != r:
                            continue
                        for ti in self.g.get((x, y, z), ()):
                            if ti in seen:
                                continue
                            seen.add(ti)
                            t = self.tris[ti]
                            try:
                                d = pt_tri(p, self.pos[t[0]], self.pos[t[1]], self.pos[t[2]])
                            except ZeroDivisionError:  # a degenerate triangle
                                continue
                            if d < best:
                                best = d
        return best


# ---------------------------------------------------------------- the model

CLASSES = [
    # (class, role, keywords): first match on the part name wins.
    ("glass", "glass", ["glass", "windshield", "window", "backlight"]),
    ("lights", "bolt-on", ["headlight", "taillight", "foglight", "lightbar", "light", "lamp"]),
    ("wheel", "mechanical", ["wheeldata", "wheel_", "tire", "hubcap", "wheels"]),
    ("door", "closure", ["doorpanel", "door"]),
    ("hood", "closure", ["hood", "bonnet"]),
    ("trunk", "closure", ["trunk", "tailgate", "liftgate", "boot", "hatch"]),
    ("fender", "bolt-on", ["fender", "quarterpanel", "wing"]),
    ("bumper", "bolt-on", ["bumper", "fascia", "bullbar", "pushbar", "crashbar"]),
    ("trim", "bolt-on", ["sideskirt", "spoiler", "grille", "mirror", "lettering", "licenseplate", "roofrack", "flare", "mudflap", "skirt", "lip", "badge", "wiper", "antenna", "trim", "roofbar", "runningboard"]),
    ("roof", "structural", ["softtop", "hardtop", "tops", "roof", "top_"]),
    ("interior", "interior", ["seat", "dash", "steeringwheel", "steer", "interior", "console", "shifter", "pedal", "gauge", "rollcage", "cage", "navi", "radio"]),
    ("subframe", "mechanical", ["subframe", "crossmember"]),
    ("suspension", "mechanical", ["suspension", "hub", "strut", "shock", "coilover", "spring", "swaybar", "sway", "arm", "steering", "brake", "axle", "leaf", "link", "knuckle", "halfshaft"]),
    ("engine", "mechanical", ["enginemount", "engine", "intake", "turbo", "supercharger", "header", "oilpan", "ecu", "flywheel", "n2o", "nitrous"]),
    ("driveline", "mechanical", ["transmission", "transfer", "driveshaft", "differential", "diff", "clutch", "torqueconverter", "finaldrive"]),
    ("exhaust", "mechanical", ["exhaust", "muffler"]),
    ("fuel", "mechanical", ["fueltank", "fuel", "battery"]),
    ("cooling", "mechanical", ["radiator", "intercooler", "radsupport"]),
    ("frame", "structural", ["frame", "chassis"]),
    ("body", "structural", ["body", "unibody", "cab", "bed"]),
]


def classify(part):
    p = part.lower()
    for cls, role, keys in CLASSES:
        if any(k in p for k in keys):
            return cls, role
    return "other", "other"


def num(v, d=None):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if v == "FLT_MAX":
            return float("inf")
        try:
            return float(v)
        except ValueError:
            return d
    return d


class Car:
    def __init__(self, path):
        d = json.load(open(path))
        self.label = d["label"]
        self.parts = [p for p, _ in d["parts"]]
        self.slot_of = dict((p, s) for p, s in d["parts"])
        self.rows = d["rows"]
        self.flex = d["flex"]
        self.groups = dict((g, v) for g, v in d["groups"])
        self.nid = {n["id"]: i for i, n in enumerate(d["nodes"])}
        self.pos = [tuple(n["pos"]) for n in d["nodes"]]
        self.mass = [n["mass"] for n in d["nodes"]]
        self.ids = [n["id"] for n in d["nodes"]]
        self.coll = [n["collision"] for n in d["nodes"]]
        self.selfc = [n["selfCollision"] for n in d["nodes"]]
        n = len(self.pos)
        self.npart = [None] * n
        self.ngroups = [[] for _ in range(n)]
        self.nrow = [{} for _ in range(n)]
        for r in self.rows["nodes"]:
            i = self.nid.get(str(r["row"].get("id")))
            if i is None:
                continue
            self.npart[i] = r["part"]
            self.nrow[i] = r["row"]
            g = r["row"].get("group")
            self.ngroups[i] = [g] if isinstance(g, str) and g else (g if isinstance(g, list) else [])
        self.beams = []
        for r in self.rows["beams"]:
            f = r["row"]
            a, b = self.nid.get(str(f.get("id1"))), self.nid.get(str(f.get("id2")))
            if a is None or b is None:
                continue
            ty = (f.get("beamType") or "|NORMAL").lstrip("|").upper() or "NORMAL"
            if ty in ("LBEAM", "BROKEN"):
                continue
            bg = f.get("breakGroup")
            bg = [bg] if isinstance(bg, str) and bg else ([x for x in bg if x] if isinstance(bg, list) else [])
            dg = f.get("deformGroup")
            dg = [dg] if isinstance(dg, str) and dg else ([x for x in dg if x] if isinstance(dg, list) else [])
            strength = num(f.get("beamStrength"), float("inf"))
            deform = num(f.get("beamDeform"), 220000.0)
            self.beams.append(
                dict(
                    a=a, b=b, part=r["part"], ty=ty,
                    k=num(f.get("beamSpring"), 4.3e6), c=num(f.get("beamDamp"), 580.0),
                    deform=deform if deform < INF else float("inf"),
                    strength=strength if strength < INF else float("inf"),
                    L=dist(self.pos[a], self.pos[b]), bg=bg, dg=dg,
                    dle=num(f.get("deformLimitExpansion")), dl=num(f.get("deformLimit")),
                    pre=num(f.get("beamPrecompression"), 1.0),
                    prer=num(f.get("precompressionRange")),
                    trig=num(f.get("deformationTriggerRatio")),
                    bgt=num(f.get("breakGroupType"), 0.0),
                    lb=num(f.get("beamLongBound"), 1.0), sb=num(f.get("beamShortBound"), 1.0),
                    ls=num(f.get("beamLimitSpring")), optional=bool(f.get("optional")),
                )
            )
        self.tris = []
        for r in self.rows["triangles"]:
            f = r["row"]
            t = [self.nid.get(str(f.get(k))) for k in ("id1", "id2", "id3")]
            if None in t:
                continue
            self.tris.append((t, r["part"], "NONCOLLIDABLE" not in (f.get("triangleType") or "")))
        # Parent of each part: the part whose slot row names this part's slot type.
        self.parent = {}
        slot_owner = {}
        for sec in ("slots", "slots2"):
            for r in self.rows[sec]:
                f = r["row"]
                for key in ("type", "name"):
                    if isinstance(f.get(key), str):
                        slot_owner.setdefault(f[key], r["part"])
                at = f.get("allowTypes")
                for t in at if isinstance(at, list) else []:
                    slot_owner.setdefault(t, r["part"])
        for p in self.parts:
            self.parent[p] = slot_owner.get(self.slot_of[p])
        # Wheels.
        self.wheels = []
        for r in self.rows["pressureWheels"] + self.rows["hubWheels"]:
            f = r["row"]
            n1, n2 = self.nid.get(str(f.get("node1"))), self.nid.get(str(f.get("node2")))
            if n1 is None or n2 is None:
                continue
            self.wheels.append(dict(name=f.get("name"), n1=n1, n2=n2, row=f, part=r["part"]))
        ys = sorted(set(round(0.5 * (self.pos[w["n1"]][1] + self.pos[w["n2"]][1]), 2) for w in self.wheels))
        self.axle_y = ys
        self.total = sum(self.mass)

    def depth(self, p):
        d = 0
        while self.parent.get(p) and d < 20:
            p = self.parent[p]
            d += 1
        return d

    def body_parts(self):
        """The structural shell: parts classed body or frame that own nodes."""
        own = Counter(self.npart)
        return [p for p in self.parts if classify(p)[0] in ("body", "frame") and own.get(p, 0) >= 20]

    def s_of(self, y):
        """Longitudinal station: 0 at the front axle, 1 at the rearmost axle."""
        if len(self.axle_y) < 2:
            return float("nan")
        return (y - self.axle_y[0]) / (self.axle_y[-1] - self.axle_y[0])


def spring(b):
    return b["ty"] in ("NORMAL", "SUPPORT", "ANISOTROPIC", "BOUNDED", "PRESSURED", "HYDRO") and b["k"] > 0


# ---------------------------------------------------------------- sections


def sec_parts(car):
    print(f"## {car.label}: parts and slot tree\n")
    print(f"{len(car.pos)} nodes, {car.total:.0f} kg (tyres excluded), {len(car.beams)} beams, {len(car.tris)} triangles, "
          f"{len(car.flex)} bound meshes, {len(car.rows['props'])} props, {len(car.rows['torsionbars'])} torsion bars, "
          f"axles at y = {car.axle_y}\n")
    nn, mm, bb, tt, ff, pp = Counter(), Counter(), Counter(), Counter(), defaultdict(list), Counter()
    for i, p in enumerate(car.npart):
        nn[p] += 1
        mm[p] += car.mass[i]
    for b in car.beams:
        bb[b["part"]] += 1
    for t in car.tris:
        tt[t[1]] += 1
    for f in car.flex:
        ff[f["part"]].append(f["mesh"])
    allflex = defaultdict(list)
    for r in car.rows["flexbodies"]:
        if isinstance(r["row"].get("mesh"), str):
            allflex[r["part"]].append(r["row"]["mesh"])
    for r in car.rows["props"]:
        pp[r["part"]] += 1
    rows = []
    for p in car.parts:
        cls, role = classify(p)
        ms = [car.mass[i] for i, x in enumerate(car.npart) if x == p]
        rows.append([
            "· " * car.depth(p) + p, car.parent.get(p) or "–", cls, role, nn[p], fmt(mm[p], 1),
            f"{fmt(q(ms, 0.1))}/{fmt(q(ms, 0.5))}/{fmt(q(ms, 0.9))}" if ms else "–",
            bb[p], tt[p], len(allflex[p]), pp[p], ", ".join(allflex[p][:6]) + (" …" if len(allflex[p]) > 6 else ""),
        ])
    table(["part (indented by slot depth)", "parent", "class", "role", "nodes", "kg", "node kg p10/50/90", "beams", "tris", "meshes", "props", "mesh names"], rows)
    by = defaultdict(lambda: [0, 0.0, 0, 0, 0, 0])
    for p in car.parts:
        cls, _ = classify(p)
        e = by[cls]
        e[0] += nn[p]
        e[1] += mm[p]
        e[2] += bb[p]
        e[3] += tt[p]
        e[4] += len(allflex[p])
        e[5] += 1
    print("By class:\n")
    table(["class", "parts", "nodes", "kg", "% of mass", "beams", "tris", "meshes"],
          [[c, e[5], e[0], fmt(e[1], 1), fmt(100 * e[1] / car.total, 1), e[2], e[3], e[4]] for c, e in sorted(by.items(), key=lambda x: -x[1][1])])


def beam_stats(bs):
    fin = [b["strength"] for b in bs if b["strength"] < INF]
    kl = [b["k"] * b["L"] for b in bs if b["k"] > 0]
    ey = [100 * b["deform"] / (b["k"] * b["L"]) for b in bs if b["k"] > 0 and b["L"] > 0 and b["deform"] < INF]
    return dict(
        n=len(bs), k=q([b["k"] for b in bs], 0.5), c=q([b["c"] for b in bs], 0.5), L=q([b["L"] for b in bs], 0.5),
        kl=q(kl, 0.5), deform=q([b["deform"] for b in bs if b["deform"] < INF], 0.5), ey=q(ey, 0.5),
        fin=len(fin), smin=min(fin) if fin else float("nan"), s50=q(fin, 0.5), ssum=sum(fin),
        types=Counter(b["ty"] for b in bs),
    )


def sec_attach(car):
    """Every pair of parts joined by beams: the mount patterns."""
    print(f"## {car.label}: what attaches to what\n")
    pair = defaultdict(list)
    for b in car.beams:
        pa, pb = car.npart[b["a"]], car.npart[b["b"]]
        if pa != pb and pa and pb:
            pair[tuple(sorted((pa, pb)))].append(b)
    rows = []
    for (pa, pb), bs in sorted(pair.items(), key=lambda x: (classify(x[0][0])[0], x[0])):
        s = beam_stats(bs)
        groups = Counter(g for b in bs for g in b["bg"])
        na = len(set(b["a"] if car.npart[b["a"]] == pa else b["b"] for b in bs))
        nb = len(set(b["a"] if car.npart[b["a"]] == pb else b["b"] for b in bs))
        rows.append([
            pa, pb, s["n"], f"{na}↔{nb}", " ".join(f"{t[:3]}{c}" for t, c in s["types"].most_common()),
            fmt(s["k"]), fmt(s["c"], 0), fmt(s["deform"]), f"{s['fin']}/{s['n']}", fmt(s["smin"]), fmt(s["s50"]), fmt(s["ssum"]),
            "; ".join(f"{g}×{c}" for g, c in groups.most_common(5)) or "–",
        ])
    table(["part A", "part B", "beams", "nodes A↔B", "types", "k p50", "c p50", "deform p50", "finite strength", "min", "p50", "Σ finite", "break groups"], rows)
    # Couplers.
    cp = [(r["part"], r["row"]) for r in car.rows["nodes"] if any(k in r["row"] for k in ("couplerStrength", "couplerTag", "tag"))]
    if cp:
        print("Coupler and tag nodes:\n")
        table(["part", "node", "couplerTag", "tag", "couplerStrength", "couplerRadius", "start radius", "lock", "weld", "breakGroup"],
              [[p, f.get("id"), f.get("couplerTag", ""), f.get("tag", ""), fmt(num(f.get("couplerStrength"))), f.get("couplerRadius", ""),
                f.get("couplerStartRadius", ""), f.get("couplerLock", ""), f.get("couplerWeld", ""), f.get("breakGroup", "")] for p, f in cp])


def body_nodes(car):
    bp = set(car.body_parts())
    return [i for i, p in enumerate(car.npart) if p in bp]


STATIONS = [(-9, -0.18, "nose (ahead of −0.18)"), (-0.18, 0.08, "front axle / towers"), (0.08, 0.30, "firewall / A-pillar"),
            (0.30, 0.62, "cabin front"), (0.62, 0.90, "cabin rear"), (0.90, 1.12, "rear axle / towers"), (1.12, 9, "tail (behind 1.12)")]


def station(s):
    for lo, hi, name in STATIONS:
        if lo <= s < hi:
            return name
    return "?"


def sec_body(car):
    print(f"## {car.label}: the body shell's nodes\n")
    ids = body_nodes(car)
    if not ids:
        print("(no body part found)\n")
        return
    print(f"Shell parts: {', '.join(car.body_parts())}; {len(ids)} nodes, {sum(car.mass[i] for i in ids):.0f} kg\n")
    zs = [car.pos[i][2] for i in ids]
    z0, z1 = min(zs), max(zs)
    xs = max(abs(car.pos[i][0]) for i in ids)
    ys = [car.pos[i][1] for i in ids]
    print(f"Box: x ±{xs:.2f}, y {min(ys):.2f}…{max(ys):.2f}, z {z0:.2f}…{z1:.2f}; wheelbase {car.axle_y[-1] - car.axle_y[0]:.2f}\n")

    def band(z):
        t = (z - z0) / (z1 - z0)
        return "floor (<0.2)" if t < 0.2 else ("sill-belt (0.2–0.55)" if t < 0.55 else ("belt-roof (0.55–0.85)" if t < 0.85 else "roof (>0.85)"))

    cell = defaultdict(list)
    for i in ids:
        cell[(station(car.s_of(car.pos[i][1])), band(car.pos[i][2]))].append(i)
    bands = ["floor (<0.2)", "sill-belt (0.2–0.55)", "belt-roof (0.55–0.85)", "roof (>0.85)"]
    print("Nodes, mass (kg) and median node mass by station (s: 0 front axle, 1 rear axle) and height band (share of shell height):\n")
    rows = []
    for _, _, st in STATIONS:
        r = [st]
        tot = []
        for b in bands:
            v = cell.get((st, b), [])
            tot += v
            r.append(f"{len(v)} / {sum(car.mass[i] for i in v):.0f} / {fmt(q([car.mass[i] for i in v], 0.5))}" if v else "–")
        r.append(f"{len(tot)} / {sum(car.mass[i] for i in tot):.0f}")
        on_c = sum(1 for i in tot if abs(car.pos[i][0]) < 0.02)
        r.append(on_c)
        rows.append(r)
    table(["station"] + [b + " n / kg / p50" for b in bands] + ["all n / kg", "on centreline"], rows)
    # Lateral lanes.
    lanes = Counter()
    for i in ids:
        t = abs(car.pos[i][0]) / xs
        lanes["centre (<0.05)" if t < 0.05 else ("inner (0.05–0.45)" if t < 0.45 else ("rail (0.45–0.8)" if t < 0.8 else "side (>0.8)"))] += 1
    print("Lateral lanes (|x| as a share of half width): " + ", ".join(f"{k} {v}" for k, v in lanes.items()) + "\n")
    # Name prefixes: the author's own structural naming.
    pre = defaultdict(list)
    for i in ids:
        m = re.match(r"^([A-Za-z_]+?)(\d+)(l+|r+)?$", car.ids[i])
        pre[(m.group(1) + m.group(2)) if m else car.ids[i]].append(i)
    print("Node-name families (the author's rows; side suffixes stripped):\n")
    rows = []
    for k, v in sorted(pre.items(), key=lambda x: min(car.pos[i][1] for i in x[1])):
        P = [car.pos[i] for i in v]
        rows.append([k, len(v), fmt(sum(car.mass[i] for i in v), 1), fmt(q([car.mass[i] for i in v], 0.5)),
                     f"{min(p[1] for p in P):.2f}…{max(p[1] for p in P):.2f}", f"{car.s_of(min(p[1] for p in P)):.2f}…{car.s_of(max(p[1] for p in P)):.2f}",
                     f"{min(p[2] for p in P):.2f}…{max(p[2] for p in P):.2f}", f"{max(abs(p[0]) for p in P):.2f}",
                     ", ".join(sorted(set(g for i in v for g in car.ngroups[i]))[:5])])
    table(["family", "nodes", "kg", "node kg p50", "y range", "s range", "z range", "|x| max", "groups"], rows)
    # Flags and groups.
    nc = sum(1 for i in ids if not car.coll[i])
    ns = sum(1 for i in ids if car.selfc[i])
    print(f"Collision off: {nc} nodes; selfCollision on: {ns}; distinct masses: {sorted(Counter(round(car.mass[i], 2) for i in ids).items())}\n")
    # Spacing.
    nn = []
    for i in ids:
        d = sorted(dist(car.pos[i], car.pos[j]) for j in ids if j != i)
        nn.append(d[0])
    print(f"Nearest-neighbour spacing within the shell: p10 {q(nn, 0.1):.3f}, p50 {q(nn, 0.5):.3f}, p90 {q(nn, 0.9):.3f}, min {min(nn):.3f} m\n")
    # Distance to the shell's visual meshes.
    S = set(ids)
    meshes = [f for f in car.flex if len(f["pos"]) >= 90 and sum(1 for i in f["pool"] if i in S) >= 0.5 * len(f["pool"])]
    if meshes:
        P, T = [], []
        for f in meshes:
            o = len(P)
            P += [tuple(p) for p in f["pos"]]
            T += [(t[0] + o, t[1] + o, t[2] + o) for t in f["tris"]]
        g = TriGrid(P, T)
        ds = [g.nearest(car.pos[i]) for i in ids]
        fin = [d for d in ds if d < 1e9]
        print(f"Meshes bound mostly to shell nodes: {', '.join(f['mesh'] + ' (' + str(len(f['tris'])) + ')' for f in meshes)}\n")
        print(f"Distance from shell nodes to those meshes' surface ({len(T)} triangles): "
              f"p10 {q(fin, 0.1) * 100:.1f}, p50 {q(fin, 0.5) * 100:.1f}, p90 {q(fin, 0.9) * 100:.1f}, max {max(fin) * 100:.1f} cm; "
              f"within 2 cm {sum(d < 0.02 for d in fin)}, 2–5 cm {sum(0.02 <= d < 0.05 for d in fin)}, 5–10 cm {sum(0.05 <= d < 0.1 for d in fin)}, "
              f"over 10 cm {sum(d >= 0.1 for d in fin) + len(ds) - len(fin)} of {len(ds)}\n")
        far = sorted(((d, car.ids[i]) for d, i in zip(ds, ids)), reverse=True)[:12]
        print("Farthest: " + ", ".join(f"{n} {d * 100:.0f} cm" for d, n in far) + "\n")


def sec_grade(car):
    """Crush-path grading: beam values against longitudinal station."""
    print(f"## {car.label}: grading along the car\n")
    ids = set(body_nodes(car))
    if not ids:
        return
    bs = [b for b in car.beams if b["a"] in ids and b["b"] in ids and b["ty"] in ("NORMAL", "SUPPORT")]
    cell = defaultdict(list)
    for b in bs:
        ym = 0.5 * (car.pos[b["a"]][1] + car.pos[b["b"]][1])
        cell[station(car.s_of(ym))].append(b)
    rows = []
    for _, _, st in STATIONS:
        v = cell.get(st, [])
        if not v:
            continue
        s = beam_stats(v)
        lon = [b for b in v if abs(car.pos[b["a"]][1] - car.pos[b["b"]][1]) > 0.7 * b["L"]]
        sl = beam_stats(lon) if lon else None
        rows.append([
            st, s["n"], sum(1 for b in v if b["ty"] == "SUPPORT"), fmt(s["k"]), fmt(q([b["k"] for b in v], 0.9)), fmt(s["c"], 0),
            fmt(s["deform"]), fmt(q([b["deform"] for b in v if b["deform"] < INF], 0.1)), fmt(q([b["deform"] for b in v if b["deform"] < INF], 0.9)),
            fmt(s["ey"]), f"{s['fin']}", fmt(s["s50"]),
            fmt(100 * sum(1 for b in v if b["dle"]) / len(v), 0), fmt(q([b["dle"] for b in v if b["dle"]], 0.5)),
            len(lon), fmt(sl["k"]) if sl else "–", fmt(sl["deform"]) if sl else "–",
            fmt(sum(b["deform"] for b in lon if b["deform"] < INF)) if lon else "–",
        ])
    print("Shell-internal NORMAL and SUPPORT beams by the station of their midpoint. "
          "'Longitudinal' beams run within 45° of the car's axis; their Σ deform is the plastic force the station offers along the crush path.\n")
    table(["station", "beams", "SUPPORT", "k p50", "k p90", "c p50", "deform p50", "deform p10", "deform p90", "ε_y p50 %", "finite strength", "strength p50",
           "DLE %", "DLE p50", "longitudinal beams", "their k p50", "their deform p50", "their Σ deform"], rows)
    # The same for the car's front and rear attachments (bumper, crash structure).
    # Distinct (spring, damp, deform, strength) settings: the author's beam classes.
    cls = Counter((b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], b["dle"]) for b in bs)
    print("The shell's beam classes (one row per distinct setting, by count):\n")
    table(["type", "k", "c", "deform", "strength", "deformLimitExpansion", "beams", "L p50", "stations"],
          [[t, fmt(float(k)), c, fmt(float(d)), fmt(float(s)), e or "–", n,
            fmt(q([b["L"] for b in bs if (b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], b["dle"]) == (t, k, c, d, s, e)], 0.5)),
            " ".join(f"{st.split(' ')[0]}{m}" for st, m in Counter(station(car.s_of(0.5 * (car.pos[b['a']][1] + car.pos[b['b']][1]))) for b in bs
                                                             if (b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], b["dle"]) == (t, k, c, d, s, e)).most_common(4))]
           for (t, k, c, d, s, e), n in cls.most_common(24)])


def rank3(dirs):
    """Smallest eigenvalue share of the direction covariance: 0 when beams are coplanar."""
    m = [[sum(d[i] * d[j] for d in dirs) for j in range(3)] for i in range(3)]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr <= 0:
        return 0.0
    # Eigenvalues of a symmetric 3x3 (trigonometric form).
    a, b, c = m[0][0], m[1][1], m[2][2]
    d, e, f = m[0][1], m[1][2], m[0][2]
    qq = tr / 3
    p2 = (a - qq) ** 2 + (b - qq) ** 2 + (c - qq) ** 2 + 2 * (d * d + e * e + f * f)
    p = math.sqrt(p2 / 6)
    if p < 1e-12:
        return 1 / 3
    B = [[(m[i][j] - (qq if i == j else 0)) / p for j in range(3)] for i in range(3)]
    det = (B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
           + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]))
    r = max(-1.0, min(1.0, det / 2))
    phi = math.acos(r) / 3
    lmin = qq + 2 * p * math.cos(phi + 2 * math.pi / 3)
    return max(0.0, lmin / tr)


def sec_brace(car):
    print(f"## {car.label}: bracing of the shell\n")
    ids = body_nodes(car)
    S = set(ids)
    if not ids:
        return
    nb = defaultdict(set)
    allnb = defaultdict(list)
    for b in car.beams:
        if not spring(b) or b["ty"] not in ("NORMAL", "SUPPORT"):
            continue
        if b["a"] in S and b["b"] in S and b["ty"] == "NORMAL":
            nb[b["a"]].add(b["b"])
            nb[b["b"]].add(b["a"])
        allnb[b["a"]].append(b)
        allnb[b["b"]].append(b)
    deg = [len(nb[i]) for i in ids]
    print(f"Shell-internal NORMAL beams per node: p10 {q(deg, 0.1):.0f}, p50 {q(deg, 0.5):.0f}, p90 {q(deg, 0.9):.0f}, min {min(deg)}, max {max(deg)}\n")
    # Rank of each node's beam directions.
    rk = []
    for i in ids:
        dirs = []
        for j in nb[i]:
            d = sub(car.pos[j], car.pos[i])
            n = norm(d)
            if n > 0:
                dirs.append((d[0] / n, d[1] / n, d[2] / n))
        rk.append(rank3(dirs) if len(dirs) >= 3 else 0.0)
    print(f"Spread of each node's beam directions (smallest eigenvalue share; 0 = all beams in one plane, 0.33 = isotropic): "
          f"p10 {q(rk, 0.1):.3f}, p50 {q(rk, 0.5):.3f}; nodes under 0.02 (near-planar bracing): {sum(r < 0.02 for r in rk)} of {len(ids)}\n")
    # Triangles and tetrahedra of the beam graph.
    tri = 0
    tet = 0
    in_tet = set()
    idl = sorted(ids)
    for a in idl:
        for b in nb[a]:
            if b <= a:
                continue
            common = [c for c in nb[a] & nb[b] if c > b]
            tri += len(common)
            for x in range(len(common)):
                for y in range(x + 1, len(common)):
                    c, d = common[x], common[y]
                    if d in nb[c]:
                        # Skip flat ones.
                        v = abs(dot(sub(car.pos[b], car.pos[a]), cross(sub(car.pos[c], car.pos[a]), sub(car.pos[d], car.pos[a])))) / 6
                        L = max(dist(car.pos[a], car.pos[b]), dist(car.pos[a], car.pos[c]), dist(car.pos[a], car.pos[d]))
                        if v > 0.002 * L ** 3:
                            tet += 1
                            in_tet.update((a, b, c, d))
    nbeams = sum(len(v) for v in nb.values()) // 2
    print(f"Beam graph: {len(ids)} nodes, {nbeams} beams ({nbeams / len(ids):.1f} per node; a rigid 3D truss needs ≥ 3 − 6/n = {3 - 6 / len(ids):.2f}), "
          f"{tri} beam triangles, {tet} non-flat tetrahedra; {len(in_tet)} of {len(ids)} nodes are a corner of at least one tetrahedron\n")
    # Beam length against the local spacing: edges, diagonals, skip beams.
    nn = {}
    for i in ids:
        d = sorted(dist(car.pos[i], car.pos[j]) for j in ids if j != i)
        nn[i] = d[min(2, len(d) - 1)]
    kinds = defaultdict(list)
    for b in car.beams:
        if b["a"] in S and b["b"] in S and b["ty"] == "NORMAL":
            r = b["L"] / (0.5 * (nn[b["a"]] + nn[b["b"]]))
            kinds["edge (≤1.3× local spacing)" if r <= 1.3 else ("diagonal (1.3–2.2×)" if r <= 2.2 else "long / skip (>2.2×)")].append(b)
    rows = []
    for k in ["edge (≤1.3× local spacing)", "diagonal (1.3–2.2×)", "long / skip (>2.2×)"]:
        v = kinds.get(k, [])
        if v:
            s = beam_stats(v)
            rows.append([k, len(v), fmt(100 * len(v) / nbeams, 0), fmt(s["L"]), fmt(s["k"]), fmt(s["deform"]), fmt(s["ey"]), fmt(q([b["L"] for b in v], 0.9))])
    print("Shell NORMAL beams by length over the local spacing (the mean of both ends' third-nearest-neighbour distance):\n")
    table(["kind", "beams", "%", "L p50", "k p50", "deform p50", "ε_y p50 %", "L p90"], rows)
    # Cross-car beams: left to right.
    cross_lr = [b for b in car.beams if b["a"] in S and b["b"] in S and b["ty"] == "NORMAL"
                and car.pos[b["a"]][0] * car.pos[b["b"]][0] < -0.01]
    print(f"Beams crossing the centre plane (left node to right node): {len(cross_lr)} ({100 * len(cross_lr) / nbeams:.0f} %), L p50 {q([b['L'] for b in cross_lr], 0.5):.2f} m; "
          f"through the cabin above the floor band: see 'cabin spans' below\n")
    zs = [car.pos[i][2] for i in ids]
    z0, z1 = min(zs), max(zs)
    span = []
    for b in car.beams:
        if b["a"] in S and b["b"] in S and b["ty"] == "NORMAL" and b["L"] > 0.9:
            pa, pb = car.pos[b["a"]], car.pos[b["b"]]
            sm = car.s_of(0.5 * (pa[1] + pb[1]))
            zm = (0.5 * (pa[2] + pb[2]) - z0) / (z1 - z0)
            if 0.3 < sm < 0.9 and 0.25 < zm < 0.8:
                span.append(b)
    print(f"Cabin spans (shell beams over 0.9 m whose midpoint is inside the cabin volume, s 0.3–0.9 and 25–80 % of height): {len(span)}; "
          f"k p50 {fmt(q([b['k'] for b in span], 0.5))}, deform p50 {fmt(q([b['deform'] for b in span], 0.5))}, L p50 {fmt(q([b['L'] for b in span], 0.5))}\n")
    # SUPPORT beams inside the shell and from the shell.
    sup = [b for b in car.beams if b["ty"] == "SUPPORT" and (b["a"] in S or b["b"] in S)]
    inside = [b for b in sup if b["a"] in S and b["b"] in S]
    to = Counter()
    for b in sup:
        if not (b["a"] in S and b["b"] in S):
            o = b["b"] if b["a"] in S else b["a"]
            to[classify(car.npart[o] or "?")[0]] += 1
    print(f"SUPPORT beams touching the shell: {len(sup)}; inside the shell {len(inside)} (L p50 {fmt(q([b['L'] for b in inside], 0.5))}, k p50 {fmt(q([b['k'] for b in inside], 0.5))}, "
          f"precompressed {sum(1 for b in inside if abs(b['pre'] - 1) > 1e-6 or b['prer'])}); to other classes: {dict(to.most_common())}\n")
    pre = [b["pre"] for b in sup if abs(b["pre"] - 1) > 1e-6]
    if pre:
        print(f"Their precompression where set: {sorted(Counter(round(p, 2) for p in pre).items())}\n")
    tb = [r for r in car.rows["torsionbars"] if r["part"] in car.body_parts()]
    if tb:
        print(f"Torsion bars defined by the shell: {len(tb)}; spring {sorted(Counter(num(r['row'].get('spring')) for r in tb).items())}, "
              f"deform {sorted(Counter(num(r['row'].get('deform')) for r in tb).items())}, strength {sorted(Counter(num(r['row'].get('strength')) for r in tb).items())}\n")


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def sec_mech(car):
    print(f"## {car.label}: mechanical parts\n")
    rows = []
    own = defaultdict(list)
    for i, p in enumerate(car.npart):
        own[p].append(i)
    for p in car.parts:
        cls, role = classify(p)
        if role != "mechanical" and cls not in ("interior",):
            continue
        ids = own.get(p, [])
        S = set(ids)
        internal = [b for b in car.beams if b["a"] in S and b["b"] in S]
        ext = [b for b in car.beams if (b["a"] in S) != (b["b"] in S)]
        defined = [b for b in car.beams if b["part"] == p]
        if not ids and not defined:
            continue
        to = Counter(classify(car.npart[b["b"] if b["a"] in S else b["a"]] or "?")[0] for b in ext)
        si, se = beam_stats(internal), beam_stats(ext)
        P = [car.pos[i] for i in ids]
        rows.append([
            p, cls, len(ids), fmt(sum(car.mass[i] for i in ids), 1), fmt(q([car.mass[i] for i in ids], 0.5)), fmt(max([car.mass[i] for i in ids], default=float("nan"))),
            f"{car.s_of(min(x[1] for x in P)):.2f}…{car.s_of(max(x[1] for x in P)):.2f}" if P else "–",
            len(internal), fmt(si["k"]), len(defined), len(ext), " ".join(f"{k}{v}" for k, v in to.most_common(4)),
            fmt(se["k"]), fmt(se["deform"]), f"{se['fin']}/{se['n']}", fmt(se["smin"]), fmt(se["s50"]),
            "; ".join(f"{g}×{c}" for g, c in Counter(g for b in ext + defined for g in b["bg"]).most_common(4)) or "–",
        ])
    table(["part", "class", "nodes", "kg", "node kg p50", "max", "s range", "internal beams", "their k p50", "beams defined", "beams to other parts", "to classes",
           "k p50", "deform p50", "finite strength", "min", "p50", "break groups"], rows)
    # Engine and driveline mounts in detail: every beam from an engine or driveline node to a structural node.
    eng = set(i for i, p in enumerate(car.npart) if p and classify(p)[0] in ("engine", "driveline"))
    mounts = [b for b in car.beams if (b["a"] in eng) != (b["b"] in eng)]
    mounts = [b for b in mounts if classify(car.npart[b["b"] if b["a"] in eng else b["a"]] or "?")[0] in ("body", "frame", "subframe", "suspension")]
    if mounts:
        cl = Counter((b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], tuple(b["bg"])) for b in mounts)
        print("Engine and driveline to structure (body, frame, subframe), one row per distinct setting:\n")
        table(["type", "k", "c", "deform", "strength", "break group", "beams", "L p50"],
              [[t, fmt(float(k)), c, fmt(float(d)), fmt(float(s)), " ".join(g) or "–", n,
                fmt(q([b["L"] for b in mounts if (b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], tuple(b["bg"])) == (t, k, c, d, s, g)], 0.5))]
               for (t, k, c, d, s, g), n in cl.most_common(20)])
    if car.rows["powertrain"]:
        print("Powertrain devices: " + ", ".join(f"{r['row'].get('name')}({r['row'].get('type')})" for r in car.rows["powertrain"]) + "\n")
    eg = Counter()
    for r in car.rows["nodes"]:
        g = r["row"].get("engineGroup")
        for x in [g] if isinstance(g, str) else (g or []):
            eg[x] += 1
    if eg:
        print(f"engineGroup nodes: {dict(eg)}\n")


def sec_wheelpath(car):
    """The load path from each wheel's axle nodes to the shell."""
    print(f"## {car.label}: wheel to body load path\n")
    S = set(body_nodes(car))
    adj = defaultdict(list)
    for b in car.beams:
        if spring(b):
            adj[b["a"]].append((b["b"], b))
            adj[b["b"]].append((b["a"], b))
    rows = []
    for w in car.wheels:
        start = {w["n1"], w["n2"]}
        layers = [start]
        seen = set(start)
        reached = None
        for hop in range(1, 7):
            nxt = set()
            for i in layers[-1]:
                for j, b in adj[i]:
                    if j not in seen:
                        nxt.add(j)
            seen |= nxt
            layers.append(nxt)
            if reached is None and nxt & S:
                reached = hop
        # The hub's rigid cluster: nodes one hop from the axle nodes that are not shell nodes.
        hub = set(start) | set(j for i in start for j, _ in adj[i] if j not in S)
        hubparts = Counter(car.npart[i] for i in hub)
        out = [b for i in hub for j, b in adj[i] if j not in hub]
        to_body = [b for b in out if (b["a"] in S or b["b"] in S)]
        to = Counter(classify(car.npart[b["b"] if b["a"] in hub else b["a"]] or "?")[0] for b in out)
        fin = [b["strength"] for b in out if b["strength"] < INF]
        fr = w["row"]
        rows.append([
            w["name"], f"{car.pos[w['n1']][0]:.2f}/{car.pos[w['n2']][0]:.2f}", fmt(car.mass[w["n1"]]) + "/" + fmt(car.mass[w["n2"]]),
            len(hub), fmt(sum(car.mass[i] for i in hub), 1), " ".join(f"{p}:{c}" for p, c in hubparts.most_common(4)),
            reached, len(out), " ".join(f"{k}{v}" for k, v in to.most_common(5)), len(to_body),
            " ".join(f"{t[:3]}{c}" for t, c in Counter(b["ty"] for b in to_body).most_common()),
            f"{len(fin)}/{len(out)}", fmt(min(fin) if fin else float("nan")), fmt(q(fin, 0.5)),
            fmt(num(fr.get("hubRadius"))), fmt(num(fr.get("radius"))), fmt(num(fr.get("hubWidth"))), fmt(num(fr.get("tireWidth"))),
            fmt(num(fr.get("hubNodeWeight"))), fmt(num(fr.get("nodeWeight"))), fmt(num(fr.get("numRays")), 0),
            fr.get("hubcapBreakGroup", "") and "yes", " ".join(fr.get("axleBeams") or []) if isinstance(fr.get("axleBeams"), list) else "",
        ])
    table(["wheel", "axle nodes x", "axle node kg", "hub cluster nodes", "kg", "cluster parts", "hops to shell", "beams leaving cluster", "to classes",
           "direct to shell", "types", "finite strength", "min", "p50", "hub r", "tyre r", "hub w", "tyre w", "hub node kg", "tyre node kg", "rays", "hubcap", "axleBeams"], rows)
    # Suspension to shell: every beam joining a suspension/subframe node to a shell node.
    sus = set(i for i, p in enumerate(car.npart) if p and classify(p)[0] in ("suspension", "subframe"))
    for name, sel in (("front", lambda y: car.s_of(y) < 0.5), ("rear", lambda y: car.s_of(y) >= 0.5)):
        bs = [b for b in car.beams if ((b["a"] in sus and b["b"] in S) or (b["b"] in sus and b["a"] in S)) and sel(0.5 * (car.pos[b["a"]][1] + car.pos[b["b"]][1]))]
        if not bs:
            continue
        shell_nodes = set(b["a"] if b["a"] in S else b["b"] for b in bs)
        sus_nodes = set(b["a"] if b["a"] in sus else b["b"] for b in bs)
        cl = Counter((b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], tuple(b["bg"]), round(b["pre"], 2)) for b in bs)
        print(f"{name} suspension/subframe to shell: {len(bs)} beams, {len(sus_nodes)} suspension nodes onto {len(shell_nodes)} shell nodes "
              f"(shell node mass p50 {fmt(q([car.mass[i] for i in shell_nodes], 0.5))} kg, their z {min(car.pos[i][2] for i in shell_nodes):.2f}…{max(car.pos[i][2] for i in shell_nodes):.2f})\n")
        table(["type", "k", "c", "deform", "strength", "break group", "precompression", "beams", "L p50"],
              [[t, fmt(float(k)), c, fmt(float(d)), fmt(float(s)), " ".join(g) or "–", pre, n,
                fmt(q([b["L"] for b in bs if (b["ty"], round(b["k"]), round(b["c"]), b["deform"], b["strength"], tuple(b["bg"]), round(b["pre"], 2)) == (t, k, c, d, s, g, pre)], 0.5))]
               for (t, k, c, d, s, g, pre), n in cl.most_common(14)])
    # Weakest interfaces in the whole chassis: break groups by total strength.
    grp = defaultdict(list)
    for b in car.beams:
        for g in b["bg"]:
            grp[g].append(b)
    rows = []
    for g, bs in grp.items():
        cls = Counter(classify(car.npart[b["a"]] or "?")[0] for b in bs) + Counter(classify(car.npart[b["b"]] or "?")[0] for b in bs)
        fin = [b["strength"] for b in bs if b["strength"] < INF]
        rows.append((min(fin) if fin else float("inf"), g, len(bs), " ".join(f"{k}" for k, _ in cls.most_common(2)), len(fin), q(fin, 0.5), sum(fin),
                     sum(1 for b in bs if b["bgt"] == 1)))
    rows.sort()
    print(f"Break groups: {len(rows)}. By the weakest member's strength (a group breaks whole when one type-0 member breaks):\n")
    table(["break group", "beams", "between classes", "finite", "min strength", "p50", "Σ", "passive (type 1)"],
          [[g, n, c, f, fmt(m), fmt(p), fmt(s), pv] for m, g, n, c, f, p, s, pv in rows])


def sec_glass(car):
    print(f"## {car.label}: glass, lamps and deform groups\n")
    dg = defaultdict(list)
    for b in car.beams:
        for g in b["dg"]:
            dg[g].append(b)
    fx = defaultdict(list)
    for r in car.rows["flexbodies"]:
        g = r["row"].get("deformGroup")
        if isinstance(g, str) and g:
            fx[g].append(r["row"])
    pr = Counter(str(r["row"].get("deformGroup")) for r in car.rows["props"] if r["row"].get("deformGroup"))
    rows = []
    for g in sorted(set(dg) | set(fx)):
        bs = dg.get(g, [])
        nodes = set(x for b in bs for x in (b["a"], b["b"]))
        parts = Counter(car.npart[i] for i in nodes)
        fin = [b["strength"] for b in bs if b["strength"] < INF]
        rows.append([
            g, len(bs), len(nodes), " ".join(f"{p}:{c}" for p, c in parts.most_common(3)),
            " ".join(f"{t[:3]}{c}" for t, c in Counter(b["ty"] for b in bs).most_common()),
            fmt(q([b["k"] for b in bs], 0.5)), fmt(q([b["deform"] for b in bs if b["deform"] < INF], 0.5)), f"{len(fin)}/{len(bs)}", fmt(q(fin, 0.5)),
            " ".join(f"{t}×{c}" for t, c in Counter(b["trig"] for b in bs).most_common(3)),
            sum(1 for b in bs if b["bg"]), fmt(q([b["L"] for b in bs], 0.5)),
            len(fx.get(g, [])), " ".join(sorted(set(str(f.get("deformMaterialDamaged", "")) for f in fx.get(g, [])))), pr.get(g, 0),
        ])
    table(["deform group", "beams", "nodes", "node parts", "types", "k p50", "deform p50", "finite strength", "strength p50", "trigger ratio × beams",
           "also in a break group", "L p50", "meshes swapped", "damaged material", "props disabled"], rows)
    # Glass parts' own nodes.
    gl = [p for p in car.parts if classify(p)[0] == "glass"]
    own = Counter(car.npart)
    print("Glass parts: " + ", ".join(f"{p} ({own.get(p, 0)} nodes, {sum(1 for b in car.beams if b['part'] == p)} beams)" for p in gl) + "\n")


def sec_flex(car):
    print(f"## {car.label}: flexbody binding\n")
    print(f"flexbody rows {len(car.rows['flexbodies'])}, bound to a mesh found in the Collada files {len(car.flex)}; props {len(car.rows['props'])}\n")
    ng = Counter(len(r["row"].get("[group]") or r["row"].get("group") or []) if isinstance(r["row"].get("[group]") or r["row"].get("group"), list) else 1
                 for r in car.rows["flexbodies"])
    print(f"Node groups named per flexbody row: {dict(sorted(ng.items()))}\n")
    rows = []
    agg = defaultdict(lambda: defaultdict(list))
    for f in car.flex:
        P = [tuple(p) for p in f["pos"]]
        if len(P) < 30:
            continue
        pool = [car.pos[i] for i in f["pool"]]
        step = max(1, len(P) // 1500)
        d1, d3 = [], []
        for p in P[::step]:
            d = sorted(dist(p, n) for n in pool)
            d1.append(d[0])
            d3.append(d[min(2, len(d) - 1)])
        # Mesh thickness proxy: nodes of the pool to the surface.
        lo = [min(p[k] for p in P) for k in range(3)]
        hi = [max(p[k] for p in P) for k in range(3)]
        near = [n for n in pool if all(lo[k] - 0.1 <= n[k] <= hi[k] + 0.1 for k in range(3))]
        own_parts = Counter(car.npart[i] for i in f["pool"])
        cls = classify(f["part"])[0]
        row = [f["part"], f["mesh"], cls, len(P), len(f["tris"]), len(f["groups"]), len(pool), len(near), len(own_parts),
               fmt(q(d1, 0.5), 3), fmt(q(d1, 0.9), 3), fmt(max(d1), 3), fmt(q(d3, 0.5), 3), fmt(q(d3, 0.9), 3), fmt(max(d3), 3)]
        rows.append(row)
        a = agg[cls]
        a["verts"].append(len(P))
        a["pool"].append(len(pool))
        a["near"].append(len(near))
        a["d1"].append(q(d1, 0.5))
        a["d1x"].append(max(d1))
        a["d3"].append(q(d3, 0.5))
        a["d3x"].append(max(d3))
        a["groups"].append(len(f["groups"]))
        a["parts"].append(len(own_parts))
    table(["part", "mesh", "class", "verts", "tris", "groups", "pool nodes", "pool nodes within 10 cm of the mesh box", "parts in pool",
           "vertex→nearest node p50", "p90", "max", "→3rd nearest p50", "p90", "max"], rows)
    print("By class (median over the class's meshes; max over them for the max columns):\n")
    table(["class", "meshes", "verts p50", "groups p50", "parts in pool p50", "pool p50", "near pool p50", "nearest p50", "nearest max", "3rd nearest p50", "3rd nearest max"],
          [[c, len(a["verts"]), fmt(q(a["verts"], 0.5), 0), fmt(q(a["groups"], 0.5), 0), fmt(q(a["parts"], 0.5), 0), fmt(q(a["pool"], 0.5), 0), fmt(q(a["near"], 0.5), 0),
            fmt(q(a["d1"], 0.5), 3), fmt(max(a["d1x"]), 3), fmt(q(a["d3"], 0.5), 3), fmt(max(a["d3x"]), 3)] for c, a in sorted(agg.items())])
    # Parts with several meshes: outer/inner pairs.
    per = defaultdict(list)
    for r in car.rows["flexbodies"]:
        per[r["part"]].append(r["row"])
    multi = {p: v for p, v in per.items() if len(v) > 1 and classify(p)[0] in ("door", "hood", "trunk", "fender", "bumper", "body")}
    print("Meshes per panel part and whether they share one group list:\n")
    table(["part", "meshes", "distinct group lists", "with a deformGroup"],
          [[p, len(v), len(set(json.dumps(f.get("[group]") or f.get("group")) for f in v)), sum(1 for f in v if f.get("deformGroup"))] for p, v in multi.items()])


def sec_stab(car):
    print(f"## {car.label}: stiffness against mass\n")
    n = len(car.pos)
    sk, sc, row = [0.0] * n, [0.0] * n, [0.0] * n
    ends = [0] * n
    for b in car.beams:
        if b["ty"] not in ("NORMAL", "SUPPORT", "ANISOTROPIC", "BOUNDED"):
            continue
        a, c = b["a"], b["b"]
        ma, mb = car.mass[a], car.mass[c]
        if ma <= 0 or mb <= 0:
            continue
        g = H * H * b["k"] + 2 * H * b["c"]
        sk[a] += b["k"]
        sk[c] += b["k"]
        sc[a] += b["c"]
        sc[c] += b["c"]
        row[a] += g * (1 / ma + 1 / math.sqrt(ma * mb))
        row[c] += g * (1 / mb + 1 / math.sqrt(ma * mb))
        if b["k"] > 0:
            ends[a] += 1
            ends[c] += 1
    by = defaultdict(list)
    for i in range(n):
        by[classify(car.npart[i] or "?")[0]].append(i)
    rows = []
    for cls, ids in sorted(by.items(), key=lambda x: -len(x[1])):
        m = [car.mass[i] for i in ids]
        wh = [math.sqrt(sk[i] / car.mass[i]) * H for i in ids if car.mass[i] > 0]
        r = [row[i] for i in ids]
        dn = [H * sc[i] / car.mass[i] for i in ids if car.mass[i] > 0]
        rows.append([cls, len(ids), fmt(q(m, 0.5)), fmt(q([sk[i] for i in ids], 0.5)), fmt(q([sk[i] / car.mass[i] for i in ids], 0.5)), fmt(q(ends and [ends[i] for i in ids], 0.5), 0),
                     fmt(q(wh, 0.5)), fmt(q(wh, 0.9)), fmt(max(wh) if wh else float("nan")),
                     fmt(q(r, 0.5)), fmt(q(r, 0.9)), fmt(max(r)), fmt(100 * sum(x > 4 for x in r) / len(r), 0), fmt(q(dn, 0.5), 3), fmt(q(dn, 0.9), 3)])
    print("Per node: Σk over its spring beams, ω·h = h·√(Σk/m) at h = 0.5 ms, and the Gershgorin row "
          "Σ(h²k + 2hc)(1/mᵢ + 1/√(mᵢmⱼ)) (an explicit step is stable below 4; the census's 'row' is this over 4.8).\n")
    table(["class", "nodes", "kg p50", "Σk p50", "Σk/m p50", "spring ends p50", "ω·h p50", "p90", "max", "row p50", "p90", "max", "% over 4", "h·Σc/m p50", "p90"], rows)


def sec_types(car):
    print(f"## {car.label}: field use\n")
    fields = Counter()
    vals = defaultdict(Counter)
    for r in car.rows["beams"]:
        for k, v in r["row"].items():
            fields[k] += 1
            if isinstance(v, (int, float, str, bool)):
                vals[k][v if not isinstance(v, float) else round(v, 4)] += 1
    n = len(car.rows["beams"])
    print(f"Beam rows {n}. Fields present (rows; most common values):\n")
    table(["field", "rows", "%", "distinct", "most common"],
          [[k, c, fmt(100 * c / n, 0), len(vals[k]), ", ".join(f"{fmt(v) if isinstance(v, float) else v}×{m}" for v, m in vals[k].most_common(5))]
           for k, c in fields.most_common() if k not in ("id1", "id2")])
    nf = Counter()
    nv = defaultdict(Counter)
    for r in car.rows["nodes"]:
        for k, v in r["row"].items():
            nf[k] += 1
            if isinstance(v, (int, float, str, bool)):
                nv[k][v] += 1
    print(f"Node rows {len(car.rows['nodes'])}:\n")
    table(["field", "rows", "distinct", "most common"],
          [[k, c, len(nv[k]), ", ".join(f"{v}×{m}" for v, m in nv[k].most_common(6))] for k, c in nf.most_common() if k not in ("id", "posX", "posY", "posZ")])
    for sec in ("triangles", "flexbodies", "torsionbars", "slidenodes", "hydros", "pressureWheels"):
        tf = Counter()
        tv = defaultdict(Counter)
        for r in car.rows[sec]:
            for k, v in r["row"].items():
                tf[k] += 1
                if isinstance(v, (int, float, str, bool)):
                    tv[k][v] += 1
        if tf:
            print(f"{sec} rows {len(car.rows[sec])}: " + "; ".join(
                f"{k} ({c}: {', '.join(f'{v}×{m}' for v, m in tv[k].most_common(4))})" for k, c in tf.most_common()
                if not k.startswith("id") and k not in ("mesh", "name", "node1", "node2", "pos", "rot", "scale")) + "\n")



def summary(cars):
    """Cross-car aggregates: the tables of engine/docs/beamng-structure-study.md."""
    L = [c.label for c in cars]
    print("# Cross-car summary\n")
    # 1. Mass and nodes by class.
    classes = ["body", "frame", "roof", "door", "hood", "trunk", "fender", "bumper", "trim", "lights", "glass", "interior",
               "subframe", "suspension", "wheel", "engine", "driveline", "exhaust", "fuel", "cooling", "other"]
    print("## Nodes / kg / share of mass by class\n")
    rows = []
    for cl in classes:
        r = [cl]
        for c in cars:
            ids = [i for i, p in enumerate(c.npart) if classify(p or "?")[0] == cl]
            m = sum(c.mass[i] for i in ids)
            r.append(f"{len(ids)} / {m:.0f} / {100 * m / c.total:.0f} %" if ids else "–")
        rows.append(r)
    rows.append(["total"] + [f"{len(c.pos)} / {c.total:.0f}" for c in cars])
    rows.append(["beams / triangles / parts with nodes"] + [f"{len(c.beams)} / {len(c.tris)} / {len(set(c.npart))}" for c in cars])
    table(["class"] + L, rows)
    # 2. Shell.
    print("## Shell lattice\n")
    rows = defaultdict(list)
    for c in cars:
        ids = body_nodes(c)
        S = set(ids)
        if not ids:
            for k in ("nodes / kg",):
                rows[k].append("–")
            continue
        m = [c.mass[i] for i in ids]
        rows["nodes / kg"].append(f"{len(ids)} / {sum(m):.0f}")
        rows["node kg p10 / p50 / p90 / max"].append(f"{fmt(q(m, .1))} / {fmt(q(m, .5))} / {fmt(q(m, .9))} / {fmt(max(m))}")
        rows["distinct node masses"].append(len(set(round(x, 2) for x in m)))
        nn = [sorted(dist(c.pos[i], c.pos[j]) for j in ids if j != i)[0] for i in ids]
        rows["nearest-neighbour spacing p10 / p50 / p90 (m)"].append(f"{q(nn, .1):.2f} / {q(nn, .5):.2f} / {q(nn, .9):.2f}")
        rows["on centreline / collision off / selfCollision on"].append(
            f"{sum(1 for i in ids if abs(c.pos[i][0]) < 0.02)} / {sum(1 for i in ids if not c.coll[i])} / {sum(1 for i in ids if c.selfc[i])}")
        nb = defaultdict(set)
        bs = [b for b in c.beams if b["a"] in S and b["b"] in S and b["ty"] == "NORMAL"]
        for b in bs:
            nb[b["a"]].add(b["b"])
            nb[b["b"]].add(b["a"])
        deg = [len(nb[i]) for i in ids]
        rows["shell beams / per node"].append(f"{len(bs)} / {len(bs) / len(ids):.1f}")
        rows["beams per node p10 / p50 / p90 / min"].append(f"{q(deg, .1):.0f} / {q(deg, .5):.0f} / {q(deg, .9):.0f} / {min(deg)}")
        Ls = [b["L"] for b in bs]
        rows["beam length p10 / p50 / p90 (m)"].append(f"{q(Ls, .1):.2f} / {q(Ls, .5):.2f} / {q(Ls, .9):.2f}")
        n3 = {i: sorted(dist(c.pos[i], c.pos[j]) for j in ids if j != i)[2] for i in ids}
        kinds = Counter()
        for b in bs:
            r = b["L"] / (0.5 * (n3[b["a"]] + n3[b["b"]]))
            kinds["e" if r <= 1.3 else ("d" if r <= 2.2 else "s")] += 1
        rows["edge / diagonal / skip beams (%)"].append(f"{100 * kinds['e'] / len(bs):.0f} / {100 * kinds['d'] / len(bs):.0f} / {100 * kinds['s'] / len(bs):.0f}")
        x = [b for b in bs if c.pos[b["a"]][0] * c.pos[b["b"]][0] < -0.01]
        rows["beams crossing the centre plane (%)"].append(f"{100 * len(x) / len(bs):.0f}")
        rows["FLT_MAX strength (%)"].append(f"{100 * sum(1 for b in bs if b['strength'] >= INF) / len(bs):.0f}")
        rows["deformLimitExpansion set (%) / values"].append(
            f"{100 * sum(1 for b in bs if b['dle']) / len(bs):.0f} / {' '.join(str(k) for k, _ in Counter(b['dle'] for b in bs if b['dle']).most_common(3))}")
        rows["distinct (k, c, deform, strength) settings"].append(len(set((b["k"], b["c"], b["deform"], b["strength"]) for b in bs)))
        rows["k p10 / p50 / p90"].append(f"{fmt(q([b['k'] for b in bs], .1))} / {fmt(q([b['k'] for b in bs], .5))} / {fmt(q([b['k'] for b in bs], .9))}")
        rows["c p10 / p50 / p90"].append(f"{fmt(q([b['c'] for b in bs], .1), 0)} / {fmt(q([b['c'] for b in bs], .5), 0)} / {fmt(q([b['c'] for b in bs], .9), 0)}")
        rows["deform p10 / p50 / p90"].append(f"{fmt(q([b['deform'] for b in bs], .1))} / {fmt(q([b['deform'] for b in bs], .5))} / {fmt(q([b['deform'] for b in bs], .9))}")
        sup = [b for b in c.beams if b["ty"] == "SUPPORT" and (b["a"] in S or b["b"] in S)]
        rows["SUPPORT beams touching the shell / inside it"].append(f"{len(sup)} / {sum(1 for b in sup if b['a'] in S and b['b'] in S)}")
        rows["torsion bars defined by the shell"].append(sum(1 for r in c.rows["torsionbars"] if r["part"] in c.body_parts()))
        tr = [t for t in c.tris if all(i in S for i in t[0])]
        on = set(i for t in tr for i in t[0])
        rows["collision triangles on shell nodes / shell nodes on one (%)"].append(f"{len(tr)} / {100 * len(on) / len(ids):.0f}")
        meshes = [f for f in c.flex if len(f["pos"]) >= 90 and sum(1 for i in f["pool"] if i in S) >= 0.5 * len(f["pool"])]
        if meshes:
            P, T = [], []
            for f in meshes:
                o = len(P)
                P += [tuple(p) for p in f["pos"]]
                T += [(t[0] + o, t[1] + o, t[2] + o) for t in f["tris"]]
            g = TriGrid(P, T)
            ds = [g.nearest(c.pos[i]) for i in ids]
            ds = [d if d < 1e9 else 1.0 for d in ds]
            rows["node to shell mesh surface p50 / p90 / max (cm)"].append(f"{100 * q(ds, .5):.1f} / {100 * q(ds, .9):.1f} / {100 * max(ds):.0f}")
            rows["nodes within 2 cm / 5 cm of the surface (%)"].append(f"{100 * sum(d < .02 for d in ds) / len(ds):.0f} / {100 * sum(d < .05 for d in ds) / len(ds):.0f}")
        else:
            rows["node to shell mesh surface p50 / p90 / max (cm)"].append("–")
            rows["nodes within 2 cm / 5 cm of the surface (%)"].append("–")
    table(["shell"] + L, [[k] + v for k, v in rows.items()])
    # 3. Grading.
    print("## Grading: shell beams by station (k p50 / deform p50 / ε_y p50 % / Σ deform of longitudinal beams)\n")
    rows = []
    for _, _, st in STATIONS:
        r = [st]
        for c in cars:
            S = set(body_nodes(c))
            v = [b for b in c.beams if b["a"] in S and b["b"] in S and b["ty"] == "NORMAL"
                 and station(c.s_of(0.5 * (c.pos[b["a"]][1] + c.pos[b["b"]][1]))) == st]
            if not v:
                r.append("–")
                continue
            s = beam_stats(v)
            lon = [b for b in v if abs(c.pos[b["a"]][1] - c.pos[b["b"]][1]) > 0.7 * b["L"]]
            r.append(f"{fmt(s['k'])} / {fmt(s['deform'])} / {fmt(s['ey'])} / {fmt(sum(b['deform'] for b in lon if b['deform'] < INF))}")
        rows.append(r)
    table(["station"] + L, rows)
    # 4. Stiffness against mass.
    print("## Σk/m per node by class: p50 (1e6 s⁻²) / ω·h p50 / p90 / max\n")
    stat = {}
    for c in cars:
        n = len(c.pos)
        sk = [0.0] * n
        for b in c.beams:
            if b["ty"] in ("NORMAL", "SUPPORT", "ANISOTROPIC", "BOUNDED"):
                sk[b["a"]] += b["k"]
                sk[b["b"]] += b["k"]
        stat[c.label] = sk
    rows = []
    for cl in classes:
        r = [cl]
        any_ = False
        for c in cars:
            ids = [i for i, p in enumerate(c.npart) if classify(p or "?")[0] == cl and c.mass[i] > 0]
            if len(ids) < 2:
                r.append("–")
                continue
            any_ = True
            sk = stat[c.label]
            wh = [math.sqrt(sk[i] / c.mass[i]) * H for i in ids]
            r.append(f"{q([sk[i] / c.mass[i] for i in ids], .5) / 1e6:.1f} / {q(wh, .5):.2f} / {q(wh, .9):.2f} / {max(wh):.2f}")
        if any_:
            rows.append(r)
    table(["class"] + L, rows)
    # 5. Interfaces: break groups by the pair of classes they join.
    print("## Break groups by the classes they join: groups / beams per group p50 / weakest member p50 (N) / Σ strength p50 (N)\n")
    pairs = defaultdict(lambda: defaultdict(list))
    for c in cars:
        grp = defaultdict(list)
        for b in c.beams:
            for g in b["bg"]:
                grp[g].append(b)
        for g, bs in grp.items():
            cc = Counter()
            for b in bs:
                ca, cb = classify(c.npart[b["a"]] or "?")[0], classify(c.npart[b["b"]] or "?")[0]
                if ca != cb:
                    cc[tuple(sorted((ca, cb)))] += 1
            if not cc:
                ca = classify(c.npart[bs[0]["a"]] or "?")[0]
                key = (ca, ca)
            else:
                key = cc.most_common(1)[0][0]
            fin = [b["strength"] for b in bs if b["strength"] < INF]
            pairs[key][c.label].append((len(bs), min(fin) if fin else float("nan"), sum(fin)))
    rows = []
    for key in sorted(pairs, key=lambda k: -sum(len(v) for v in pairs[k].values())):
        if sum(len(v) for v in pairs[key].values()) < 2:
            continue
        r = [" ↔ ".join(key)]
        for c in cars:
            v = pairs[key].get(c.label)
            r.append(f"{len(v)} / {q([x[0] for x in v], .5):.0f} / {fmt(q([x[1] for x in v], .5))} / {fmt(q([x[2] for x in v], .5))}" if v else "–")
        rows.append(r)
    table(["joins"] + L, rows)
    # 6. Wheel path.
    print("## Wheel to shell\n")
    rows = defaultdict(list)
    for c in cars:
        S = set(body_nodes(c))
        adj = defaultdict(list)
        for b in c.beams:
            if spring(b) or b["ty"] == "BOUNDED":
                adj[b["a"]].append((b["b"], b))
                adj[b["b"]].append((b["a"], b))
        hops, hubn, hubm, leave, direct = [], [], [], [], []
        for w in c.wheels:
            seen = {w["n1"], w["n2"]}
            layer = set(seen)
            reached = None
            for hop in range(1, 8):
                nxt = set(j for i in layer for j, _ in adj[i] if j not in seen)
                seen |= nxt
                layer = nxt
                if nxt & S:
                    reached = hop
                    break
            hops.append(reached if reached is not None else 99)
            hub = {w["n1"], w["n2"]} | set(j for i in (w["n1"], w["n2"]) for j, _ in adj[i] if j not in S)
            hubn.append(len(hub))
            hubm.append(sum(c.mass[i] for i in hub))
            out = [b for i in hub for j, b in adj[i] if j not in hub]
            leave.append(len(out))
            direct.append(sum(1 for b in out if b["a"] in S or b["b"] in S))
        rows["wheels"].append(len(c.wheels))
        rows["axle-node mass (kg)"].append(" ".join(sorted(set(fmt(c.mass[w["n1"]]) for w in c.wheels))))
        rows["hops from an axle node to a shell node (min–max)"].append(f"{min(hops)}–{max(hops)}" if hops else "–")
        rows["nodes one hop from the axle nodes (min–max) / their kg"].append(f"{min(hubn)}–{max(hubn)} / {min(hubm):.0f}–{max(hubm):.0f}" if hubn else "–")
        rows["beams leaving that cluster / of them to the shell"].append(f"{min(leave)}–{max(leave)} / {min(direct)}–{max(direct)}" if leave else "–")
        sus = set(i for i, p in enumerate(c.npart) if p and classify(p)[0] in ("suspension", "subframe"))
        bs = [b for b in c.beams if (b["a"] in sus and b["b"] in S) or (b["b"] in sus and b["a"] in S)]
        fin = [b["strength"] for b in bs if b["strength"] < INF]
        rows["suspension/subframe ↔ shell beams / finite strength / strength p10 / p50 / p90"].append(
            f"{len(bs)} / {len(fin)} / {fmt(q(fin, .1))} / {fmt(q(fin, .5))} / {fmt(q(fin, .9))}")
        rows["  their types"].append(" ".join(f"{t[:3]}{n}" for t, n in Counter(b["ty"] for b in bs).most_common()))
        rows["  shell nodes they land on / those nodes' kg p50 (shell p50)"].append(
            f"{len(set(b['a'] if b['a'] in S else b['b'] for b in bs))} / {fmt(q([c.mass[b['a'] if b['a'] in S else b['b']] for b in bs], .5))} "
            f"({fmt(q([c.mass[i] for i in S], .5))})" if bs else "–")
    table(["wheel path"] + L, [[k] + v for k, v in rows.items()])
    # 7. Flexbody binding.
    print("## Flexbody binding by class: meshes / pool nodes p50 / vertex→nearest pool node p50 (m) / →3rd nearest p50 / worst vertex→nearest\n")
    agg = defaultdict(lambda: defaultdict(list))
    for c in cars:
        for f in c.flex:
            P = [tuple(p) for p in f["pos"]]
            if len(P) < 90:
                continue
            pool = [c.pos[i] for i in f["pool"]]
            step = max(1, len(P) // 600)
            d1, d3 = [], []
            for p in P[::step]:
                d = sorted(dist(p, n) for n in pool)
                d1.append(d[0])
                d3.append(d[min(2, len(d) - 1)])
            agg[classify(f["part"])[0]][c.label].append((len(pool), q(d1, .5), q(d3, .5), max(d1)))
    rows = []
    for cl in classes:
        if cl not in agg:
            continue
        r = [cl]
        for c in cars:
            v = agg[cl].get(c.label)
            r.append(f"{len(v)} / {q([x[0] for x in v], .5):.0f} / {q([x[1] for x in v], .5):.2f} / {q([x[2] for x in v], .5):.2f} / {max(x[3] for x in v):.2f}" if v else "–")
        rows.append(r)
    table(["class"] + L, rows)
    rows = defaultdict(list)
    for c in cars:
        fr = c.rows["flexbodies"]
        ng = Counter()
        for r in fr:
            g = r["row"].get("[group]") or r["row"].get("group")
            ng[len(g) if isinstance(g, list) else 1] += 1
        rows["flexbody rows / bound meshes / props"].append(f"{len(fr)} / {len(c.flex)} / {len(c.rows['props'])}")
        rows["rows naming 1 / 2 / 3+ node groups"].append(f"{ng[1]} / {ng[2]} / {sum(v for k, v in ng.items() if k >= 3)}")
        rows["rows with a deformGroup (material swap)"].append(sum(1 for r in fr if r["row"].get("deformGroup")))
        rows["node groups / nodes in ≥ 2 groups (%)"].append(f"{len(c.groups)} / {100 * sum(1 for g in c.ngroups if len(g) >= 2) / len(c.pos):.0f}")
    table(["flexbodies"] + L, [[k] + v for k, v in rows.items()])
    # 8. Deform groups (glass, lamps, damage triggers).
    print("## Deform groups\n")
    rows = defaultdict(list)
    for c in cars:
        dg = defaultdict(list)
        for b in c.beams:
            for g in b["dg"]:
                dg[g].append(b)
        glass = {g: v for g, v in dg.items() if any(k in g.lower() for k in ("glass", "windshield", "window"))}
        rows["deform groups / on glass"].append(f"{len(dg)} / {len(glass)}")
        rows["beams in a deform group (% of all)"].append(f"{100 * sum(len(v) for v in dg.values()) / len(c.beams):.0f}")
        rows["glass group: beams p50 (min–max)"].append(f"{q([len(v) for v in glass.values()], .5):.0f} ({min(map(len, glass.values()))}–{max(map(len, glass.values()))})" if glass else "–")
        tr = Counter(b["trig"] for v in glass.values() for b in v)
        rows["glass deformationTriggerRatio values × beams"].append(" ".join(f"{k}×{n}" for k, n in tr.most_common(5)))
        gb = [b for v in glass.values() for b in v]
        rows["glass beams with finite strength (%) / strength p50"].append(
            f"{100 * sum(1 for b in gb if b['strength'] < INF) / len(gb):.0f} / {fmt(q([b['strength'] for b in gb if b['strength'] < INF], .5))}" if gb else "–")
        rows["glass beams also in a break group (%)"].append(f"{100 * sum(1 for b in gb if b['bg']) / len(gb):.0f}" if gb else "–")
        rows["glass beams k p50 / deform p50"].append(f"{fmt(q([b['k'] for b in gb], .5))} / {fmt(q([b['deform'] for b in gb if b['deform'] < INF], .5))}" if gb else "–")
        gn = [i for i, p in enumerate(c.npart) if classify(p or '?')[0] == "glass"]
        rows["nodes owned by glass parts / kg"].append(f"{len(gn)} / {sum(c.mass[i] for i in gn):.0f}")
    table(["deform groups"] + L, [[k] + v for k, v in rows.items()])
    # 9. Beam types and whole-car field use.
    print("## Beam types (% of rows) and options\n")
    rows = defaultdict(list)
    for c in cars:
        n = len(c.beams)
        t = Counter(b["ty"] for b in c.beams)
        for ty in ("NORMAL", "SUPPORT", "BOUNDED", "ANISOTROPIC", "PRESSURED", "HYDRO"):
            rows[ty].append(f"{100 * t[ty] / n:.1f}")
        rows["torsion bars / slidenodes / rails / hydros"].append(f"{len(c.rows['torsionbars'])} / {len(c.rows['slidenodes'])} / {len(c.rows['rails'])} / {len(c.rows['hydros'])}")
        rows["in a break group (%)"].append(f"{100 * sum(1 for b in c.beams if b['bg']) / n:.0f}")
        rows["breakGroupType 1 (% of grouped)"].append(f"{100 * sum(1 for b in c.beams if b['bg'] and b['bgt'] == 1) / max(1, sum(1 for b in c.beams if b['bg'])):.0f}")
        rows["precompression ≠ 1 (%) / of SUPPORT (%)"].append(
            f"{100 * sum(1 for b in c.beams if abs(b['pre'] - 1) > 1e-6) / n:.0f} / "
            f"{100 * sum(1 for b in c.beams if b['ty'] == 'SUPPORT' and abs(b['pre'] - 1) > 1e-6) / max(1, t['SUPPORT']):.0f}")
        rows["deformLimitExpansion (%) / deformLimit (%)"].append(f"{100 * sum(1 for b in c.beams if b['dle']) / n:.0f} / {100 * sum(1 for b in c.beams if b['dl']) / n:.0f}")
        rows["k = 0 beams (damping only)"].append(sum(1 for b in c.beams if b["k"] == 0))
        rows["FLT_MAX strength (%)"].append(f"{100 * sum(1 for b in c.beams if b['strength'] >= INF) / n:.0f}")
        rows["strength < deform (finite both)"].append(sum(1 for b in c.beams if b["strength"] < b["deform"] < INF))
        rows["collision triangles / NONCOLLIDABLE"].append(f"{sum(1 for x in c.tris if x[2])} / {sum(1 for x in c.tris if not x[2])}")
        rows["nodes: collision on (%) / selfCollision on (%)"].append(f"{100 * sum(c.coll) / len(c.pos):.0f} / {100 * sum(c.selfc) / len(c.pos):.0f}")
        rows["coupler nodes (couplerTag) / couplerStrength values"].append(
            f"{sum(1 for r in c.rows['nodes'] if r['row'].get('couplerTag'))} / "
            f"{' '.join(fmt(float(k)) for k in sorted(set(num(r['row'].get('couplerStrength')) for r in c.rows['nodes'] if num(r['row'].get('couplerStrength')))))}")
        fr = Counter(round(num(r["row"].get("frictionCoef"), 1.0), 2) for r in c.rows["nodes"])
        rows["frictionCoef values × nodes"].append(" ".join(f"{k}×{v}" for k, v in fr.most_common(4)))
        nm = Counter(str(r["row"].get("nodeMaterial", "")).lstrip("|") for r in c.rows["nodes"])
        rows["nodeMaterial × nodes"].append(" ".join(f"{k}×{v}" for k, v in nm.most_common(4)))
    table(["whole car"] + L, [[k] + v for k, v in rows.items()])
    # 10. Mechanical classes: nodes, kg, mounts.
    print("## Mechanical and interior parts: nodes / kg / heaviest node; beams to other classes / finite strength p50\n")
    rows = []
    for cl in ("engine", "driveline", "exhaust", "fuel", "cooling", "subframe", "suspension", "wheel", "interior", "frame"):
        r = [cl]
        for c in cars:
            ids = set(i for i, p in enumerate(c.npart) if classify(p or "?")[0] == cl)
            if not ids:
                r.append("–")
                continue
            ext = [b for b in c.beams if (b["a"] in ids) != (b["b"] in ids)]
            fin = [b["strength"] for b in ext if b["strength"] < INF]
            r.append(f"{len(ids)} / {sum(c.mass[i] for i in ids):.0f} / {max(c.mass[i] for i in ids):.0f}; {len(ext)} / {fmt(q(fin, .5))}")
        rows.append(r)
    table(["class"] + L, rows)



def sec_panels(car):
    """Closures and bolt-on panels as node clouds: depth and rigidifier nodes."""
    print(f"## {car.label}: panels (depth and non-colliding nodes)\n")
    inst = defaultdict(list)
    for i, p in enumerate(car.npart):
        cl = classify(p or "?")[0]
        if cl in ("door", "hood", "trunk", "fender", "bumper", "roof"):
            root = p
            while car.parent.get(root) and classify(car.parent[root])[0] == cl:
                root = car.parent[root]
            inst[(cl, root)].append(i)
    deg = Counter()
    for b in car.beams:
        deg[b["a"]] += 1
        deg[b["b"]] += 1
    rows = []
    for (cl, root), ids in sorted(inst.items()):
        P = [car.pos[i] for i in ids]
        if len(P) < 4:
            continue
        best = None
        for t in range(0, 180, 6):
            for u in range(0, 180, 6):
                a, b = math.radians(t), math.radians(u)
                d = (math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a))
                pr = [dot(x, d) for x in P]
                r = max(pr) - min(pr)
                if best is None or r < best:
                    best = r
        off = [i for i in ids if not car.coll[i]]
        S = set(ids)
        internal = sum(1 for b in car.beams if b["a"] in S and b["b"] in S)
        ext = sum(1 for b in car.beams if (b["a"] in S) != (b["b"] in S))
        rows.append([cl, root, len(ids), fmt(sum(car.mass[i] for i in ids), 1), fmt(best), len(off), " ".join(str(deg[i]) for i in off),
                     internal, fmt(internal / len(ids), 1), ext, sum(1 for t in car.tris if all(i in S for i in t[0]))])
    print("Depth is the node cloud's smallest extent over all directions (6° steps): for a single-layer panel, the offset of its rigidifier nodes.\n")
    table(["class", "part (with same-class children)", "nodes", "kg", "depth (m)", "non-colliding nodes", "their beam counts", "internal beams", "per node", "beams to other parts", "triangles"], rows)


def sec_height(car):
    """Longitudinal yield ahead of s 0.2 and behind s 0.95, by the height of each beam's midpoint."""
    print(f"## {car.label}: longitudinal load path by height\n")
    if len(car.axle_y) < 2 or not car.wheels:
        return
    print(f"Wheel centre height {sum(car.pos[w['n1']][2] for w in car.wheels) / len(car.wheels):.2f} m. "
          "NORMAL beams of any part within 45° of the car's axis; bands by midpoint height in the model's own z.\n")
    rows = []
    for name, lo, hi in (("front (s < 0.2)", -9, 0.2), ("rear (s > 0.95)", 0.95, 9)):
        bands = defaultdict(list)
        for b in car.beams:
            if b["ty"] != "NORMAL":
                continue
            pa, pb = car.pos[b["a"]], car.pos[b["b"]]
            if not (lo <= car.s_of(0.5 * (pa[1] + pb[1])) < hi) or abs(pa[1] - pb[1]) < 0.7 * b["L"]:
                continue
            zm = 0.5 * (pa[2] + pb[2])
            bands["low (z < 0.45)" if zm < 0.45 else ("middle (0.45–0.75)" if zm < 0.75 else "high (> 0.75)")].append(b)
        tot = sum(b["deform"] for v in bands.values() for b in v if b["deform"] < INF)
        for band in ("low (z < 0.45)", "middle (0.45–0.75)", "high (> 0.75)"):
            v = bands.get(band, [])
            if not v:
                continue
            fin = [b["deform"] for b in v if b["deform"] < INF]
            cl = Counter(classify(car.npart[b["a"]] or "?")[0] + "–" + classify(car.npart[b["b"]] or "?")[0] for b in v)
            rows.append([name, band, len(v), fmt(sum(fin)), fmt(100 * sum(fin) / tot, 1), fmt(q([b["k"] for b in v], 0.5)), fmt(q(fin, 0.5)),
                         " ".join(f"{k}:{n}" for k, n in cl.most_common(4))])
    table(["end", "band", "longitudinal beams", "Σ deform (N)", "% of the end", "k p50", "deform p50", "between classes"], rows)
    m = defaultdict(float)
    for i in range(len(car.pos)):
        if car.s_of(car.pos[i][1]) < 0.2:
            m[classify(car.npart[i] or "?")[0]] += car.mass[i]
    print("Mass ahead of s 0.2 by class (kg): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(m.items(), key=lambda x: -x[1])) + "\n")
    com = [sum(car.mass[i] * car.pos[i][k] for i in range(len(car.pos))) / car.total for k in range(3)]
    print(f"Centre of mass (tyres excluded): s {car.s_of(com[1]):.2f}, z {com[2]:.2f} m\n")


def sec_counts(car):
    """Part and group counts; the engine's internals and stops; SUPPORT and damping-only beams."""
    print(f"## {car.label}: counts\n")
    own = set(p for p in car.npart if p)
    bg = set(g for b in car.beams for g in b["bg"])
    print(f"Parts {len(car.parts)}; that own nodes {len(own)} ({dict(Counter(classify(p)[1] for p in own))}); "
          f"break groups {len(bg)}; deepest slot level below the root {max(car.depth(p) for p in car.parts)}\n")
    eng = set(i for i, p in enumerate(car.npart) if p and classify(p)[0] == "engine")
    if eng:
        ib = [b for b in car.beams if b["a"] in eng and b["b"] in eng]
        heavy = [i for i in eng if car.mass[i] >= 10]
        print(f"Engine: {len(eng)} nodes, {len(heavy)} of 10 kg or more ({sorted(set(round(car.mass[i], 1) for i in heavy))}); "
              f"{len(ib)} beams among them, k p50 {fmt(q([b['k'] for b in ib], 0.5))}, k max {fmt(max(b['k'] for b in ib))}\n")
        sup = [b for b in car.beams if b["ty"] == "SUPPORT" and ((b["a"] in eng) != (b["b"] in eng))]
        print(f"Engine stops (SUPPORT from an engine node to any other part): {len(sup)}; precompressed {sum(1 for b in sup if abs(b['pre'] - 1) > 1e-6)} "
              f"({sorted(Counter(round(b['pre'], 2) for b in sup).items())}); k p50 {fmt(q([b['k'] for b in sup], 0.5))}, L p50 {fmt(q([b['L'] for b in sup], 0.5))}\n")
    sup = [b for b in car.beams if b["ty"] == "SUPPORT"]
    if sup:
        print(f"SUPPORT: {len(sup)}; precompression ≠ 1 on {sum(1 for b in sup if abs(b['pre'] - 1) > 1e-6)}; finite strength on {sum(1 for b in sup if b['strength'] < INF)}; "
              f"k p50 {fmt(q([b['k'] for b in sup], 0.5))}; beamLongBound {Counter(b['lb'] for b in sup).most_common(4)}\n")
    z = [b for b in car.beams if b["k"] == 0]
    cl = Counter((b["ty"], " – ".join(sorted((classify(car.npart[b["a"]] or "?")[0], classify(car.npart[b["b"]] or "?")[0])))) for b in z)
    print(f"Beams with no spring (k = 0): {len(z)}: " + "; ".join(f"{t} {c} ×{n}" for (t, c), n in cl.most_common(12)) + "\n")


SECTIONS = dict(parts=sec_parts, attach=sec_attach, body=sec_body, grade=sec_grade, brace=sec_brace, mech=sec_mech,
                wheelpath=sec_wheelpath, glass=sec_glass, flex=sec_flex, stab=sec_stab, types=sec_types,
                panels=sec_panels, height=sec_height, counts=sec_counts)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
        args = [a for a in args if a not in only and ",".join(only) != a]
    if not args:
        print(__doc__)
        return
    d = args[0]
    import os
    cars = args[1:] or sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))
    if only == ["summary"]:
        summary([Car(os.path.join(d, c + ".json")) for c in cars])
        return
    for c in cars:
        car = Car(os.path.join(d, c + ".json"))
        print(f"# {car.label}\n")
        for name, fn in SECTIONS.items():
            if only is None or name in only:
                fn(car)


if __name__ == "__main__":
    main()
