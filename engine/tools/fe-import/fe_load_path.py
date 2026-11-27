#!/usr/bin/env python3
"""The front load path of a parsed LS-DYNA vehicle deck: parts, how they join, sections along
each member, and where the powertrain sits.

    python3 -I engine/tools/fe-import/fe_load_path.py CACHE_DIR OUT_DIR

Reads CACHE_DIR/deck.pickle (lsdyna_deck.py) and writes OUT_DIR/load_path.json:

- "parts": every part ahead of x = X_REAR (the car lies at negative x, the bumper near 0) with
  its element mass, gauge, yield, flow stress (mean of its curve over plastic strain 0-0.2, as
  fe_sections.py), Cowper-Symonds constants and the bounding box of its nodes;
- "joins": for each pair of those parts, how many connections join them, by kind:
  `spotweld` (*CONSTRAINED_SPOTWELD node pairs), `weld_beam` (the 5 mm spot-weld beams, tied to
  the shells by contact: each end is given to the part whose shell node is nearest, within
  TIE_MM; an approximation of the tied contact's search, stated as such), `rigid` (one
  *CONSTRAINED_NODAL_RIGID_BODY spanning both parts: a bolt or a clinch), `shared` (nodes common
  to both parts' elements: merged meshes);
- "cuts": plane cuts x = c through the members named in MEMBERS (left side), as fe_sections.py:
  per part the wall length in the section, the section's box, equivalent gauge and flow stress,
  and the steel area;
- "solids": the powertrain's and other heavy solids' boxes.

Standard library only. The deck is untrusted data: this script only reads it. Nothing in a test
or a build reads its output (AGENTS.md: inspiration/ is local only).
"""
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict

X_REAR = -2300.0    # mm: parts whose nodes all lie behind this are left out
TIE_MM = 4.0        # a weld beam's end belongs to the part with a shell node this near
# member -> part-name pattern (names lower-cased, leading "<n>_" removed), left side windows
MEMBERS = {
    "midrail": (r"(midrail|midreail|midraril|midrearil|bumpermidrail|midraril)", {"y": (330, 700), "z": (150, 700)}),
    "upper_rail": (r"(upperrail|uppertaril|upperreail|uppertiebar|apillarupperreailcnt)", {"y": (480, 900), "z": (550, 1000)}),
    "subframe": (r"(subframe|subfame|subrame|su bramebrkt)", {"y": (-900, 900), "z": (100, 450)}),
    "shotgun": (r"(shotgun|rocker|apillarinnerlower|apillarlower)", {"y": (330, 900), "z": (100, 600)}),
    "bumper_beam": (r"(bumperfront$)", {"y": (-900, 900), "z": (300, 700)}),
}
CUT_X = [-80, -120, -160, -200, -240, -280, -320, -360, -400, -450, -500, -550, -600, -650, -700, -750, -800,
         -850, -900, -950, -1000, -1050, -1100, -1150, -1200, -1250, -1300, -1350, -1400, -1500, -1600, -1700, -1800]


def flow_stress(curve, lo=0.0, hi=0.2, n=41):
    if not curve or len(curve) < 2:
        return None

    def at(e):
        (a0, s0), (a1, s1) = curve[0], curve[1]
        if e < a0:
            return s0 + (s1 - s0) * (e - a0) / (a1 - a0)
        for (a0, s0), (a1, s1) in zip(curve, curve[1:]):
            if a0 <= e <= a1:
                return s0 + (s1 - s0) * (e - a0) / (a1 - a0) if a1 > a0 else s0
        return curve[-1][1]
    return sum(at(lo + (hi - lo) * i / (n - 1)) for i in range(n)) / n


def clean(title):
    return re.sub(r"^\d+_", "", (title or "").strip()).lower()


def main():
    cache, out = sys.argv[1], sys.argv[2]
    d = pickle.load(open(os.path.join(cache, "deck.pickle"), "rb"))
    parts, mats, secs, curves = d["parts"], d["mats"], d["sections"], d["curves"]
    nid, X = d["node_id"], d["xyz"]

    def xyz(n):
        i = 3 * nid[n]
        return (X[i], X[i + 1], X[i + 2])

    # element node lists per part (shells, solids, beams other than the weld beams)
    weld_pid = {p for p in parts if (mats.get(parts[p]["mid"], {}).get("type") or "").startswith("SPOTWELD")}
    elems = defaultdict(list)          # pid -> list of node-id tuples
    for (pids, ns, k) in ((d["shell_pid"], d["shell_n"], 4), (d["solid_pid"], d["solid_n"], 8), (d["beam_pid"], d["beam_n"], 2)):
        for e in range(len(pids)):
            elems[pids[e]].append(tuple(ns[k * e:k * e + k]))
    box = {}
    for p, es in elems.items():
        if p in weld_pid:
            continue
        lo, hi = [1e30] * 3, [-1e30] * 3
        for t in es:
            for n in set(t):
                if n not in nid:
                    continue
                q = xyz(n)
                for j in range(3):
                    lo[j] = min(lo[j], q[j]); hi[j] = max(hi[j], q[j])
        box[p] = (lo, hi)
    front = {p for p, (lo, hi) in box.items() if hi[0] > X_REAR and parts[p].get("source") == "vehicle"}

    node_parts = defaultdict(set)
    for p in front:
        for t in elems[p]:
            for n in t:
                node_parts[n].add(p)

    joins = defaultdict(lambda: defaultdict(int))

    def join(a, b, kind):
        if a != b and a in front and b in front:
            joins[(min(a, b), max(a, b))][kind] += 1

    # shared nodes (merged meshes)
    for n, ps in node_parts.items():
        if len(ps) > 1:
            ps = sorted(ps)
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    join(ps[i], ps[j], "shared_nodes")
    for (a, b) in d["spotwelds"]:
        for pa in node_parts.get(a, ()):
            for pb in node_parts.get(b, ()):
                join(pa, pb, "spotweld")
    for (pid, nsid) in d["nrb"]:
        ns = d["node_sets"].get(nsid, [])
        ps = sorted({p for n in ns for p in node_parts.get(n, ())})
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                join(ps[i], ps[j], "rigid")
    # weld beams: each end to the nearest shell node's part within TIE_MM
    grid = defaultdict(list)
    cell = TIE_MM
    for n, ps in node_parts.items():
        if n not in nid:
            continue
        q = xyz(n)
        grid[(int(q[0] // cell), int(q[1] // cell), int(q[2] // cell))].append(n)

    def nearest_part(q, exclude=()):
        best, bp = TIE_MM, None
        c = (int(q[0] // cell), int(q[1] // cell), int(q[2] // cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for n in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        r = math.dist(q, xyz(n))
                        if r < best:
                            for p in node_parts[n]:
                                if p not in exclude:
                                    best, bp = r, p
        return bp
    bp_, bn_ = d["beam_pid"], d["beam_n"]
    weld_beams = 0
    for e in range(len(bp_)):
        if bp_[e] not in weld_pid:
            continue
        a, b = bn_[2 * e], bn_[2 * e + 1]
        if a not in nid or b not in nid:
            continue
        qa, qb = xyz(a), xyz(b)
        if max(qa[0], qb[0]) < X_REAR:
            continue
        pa = nearest_part(qa)
        pb = nearest_part(qb, exclude=(pa,))
        if pa is not None and pb is not None:
            join(pa, pb, "weld_beam")
            weld_beams += 1

    def part_row(p):
        m = mats.get(parts[p]["mid"], {})
        s = secs.get(parts[p]["secid"], {})
        lo, hi = box[p]
        return dict(pid=p, name=parts[p]["title"], mat=m.get("type"), t_mm=s.get("t"), sigy_MPa=m.get("sigy"),
                    flow_0_0p2_MPa=(lambda f: f and round(f, 1))(flow_stress(curves.get(m.get("lcss")) or [])),
                    cowper_symonds=(m.get("c"), m.get("p")), elements=len(elems[p]),
                    box_lo=[round(v, 1) for v in lo], box_hi=[round(v, 1) for v in hi])

    # plane cuts through the members (left side: y > 0 windows; the subframe and the beam whole)
    sp, sn = d["shell_pid"], d["shell_n"]
    cuts = []
    for member, (pat, win) in MEMBERS.items():
        rx = re.compile(pat)
        mp = {p for p in front if rx.search(clean(parts[p]["title"])) and mats.get(parts[p]["mid"], {}).get("type") == "PIECEWISE_LINEAR_PLASTICITY"}
        for c in CUT_X:
            per = defaultdict(float)
            lo, hi = [1e30, 1e30], [-1e30, -1e30]
            for p in mp:
                if not (box[p][0][0] <= c <= box[p][1][0]):
                    continue
                for t in elems[p]:
                    if len(t) != 4:
                        continue
                    ids = t[:3] if t[2] == t[3] else t
                    P = [xyz(n) for n in ids]
                    vals = [q[0] - c for q in P]
                    if min(vals) > 0 or max(vals) < 0:
                        continue
                    hits = []
                    for k in range(len(P)):
                        q0, q1 = P[k], P[(k + 1) % len(P)]
                        v0, v1 = vals[k], vals[(k + 1) % len(P)]
                        if (v0 <= 0 < v1) or (v1 <= 0 < v0):
                            s = v0 / (v0 - v1)
                            hits.append(tuple(q0[j] + s * (q1[j] - q0[j]) for j in range(3)))
                    if len(hits) < 2:
                        continue
                    mid = [(hits[0][j] + hits[1][j]) / 2 for j in range(3)]
                    if not (win["y"][0] <= mid[1] <= win["y"][1] and win["z"][0] <= mid[2] <= win["z"][1]):
                        continue
                    per[p] += math.dist(hits[0], hits[1])
                    for h in hits[:2]:
                        lo[0] = min(lo[0], h[1]); hi[0] = max(hi[0], h[1])
                        lo[1] = min(lo[1], h[2]); hi[1] = max(hi[1], h[2])
            if not per:
                continue
            rows = []
            for p, L in sorted(per.items(), key=lambda kv: -kv[1]):
                r = part_row(p)
                rows.append(dict(pid=p, name=r["name"], t_mm=r["t_mm"], sigy_MPa=r["sigy_MPa"], flow_MPa=r["flow_0_0p2_MPa"], wall_mm=round(L, 1)))
            A = sum(r["wall_mm"] * r["t_mm"] for r in rows)
            Lw = sum(r["wall_mm"] for r in rows)
            cuts.append(dict(member=member, x=c, y_lo=round(lo[0], 1), y_hi=round(hi[0], 1), z_lo=round(lo[1], 1), z_hi=round(hi[1], 1),
                             width_y=round(hi[0] - lo[0], 1), height_z=round(hi[1] - lo[1], 1),
                             wall_mm=round(Lw, 1), area_mm2=round(A, 1), t_eq_mm=round(A / Lw, 3),
                             sigy_eq_MPa=round(sum(r["wall_mm"] * r["t_mm"] * r["sigy_MPa"] for r in rows) / A, 1),
                             flow_eq_MPa=round(sum(r["wall_mm"] * r["t_mm"] * (r["flow_MPa"] or r["sigy_MPa"]) for r in rows) / A, 1),
                             parts=rows))

    solids = []
    for p in front:
        m = mats.get(parts[p]["mid"], {})
        if m.get("type") in ("ELASTIC", "RIGID") and len(elems[p]) > 50:
            r = part_row(p)
            solids.append(dict(pid=p, name=r["name"], mat=r["mat"], box_lo=r["box_lo"], box_hi=r["box_hi"]))
    jl = []
    for (a, b), k in joins.items():
        jl.append(dict(a=a, b=b, a_name=parts[a]["title"], b_name=parts[b]["title"], **k))
    res = dict(parts=[part_row(p) for p in sorted(front)], joins=jl, cuts=cuts, solids=solids,
               notes=dict(tie_mm=TIE_MM, weld_beams_tied=weld_beams, x_rear=X_REAR))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "load_path.json"), "w"), indent=0)
    print("parts", len(res["parts"]), "joins", len(jl), "cuts", len(cuts), "weld beams tied", weld_beams)


if __name__ == "__main__":
    main()
