#!/usr/bin/env python3
"""Cut a parsed shell model with planes and measure load-path member sections.

    python3 -I engine/tools/fe-import/fe_sections.py CACHE_DIR OUT_DIR

For each cut (a plane x = c or z = c and a rectangular window in the other two
coordinates), every shell of a structural steel part is intersected with the plane;
the segments inside the window give, per part, the wall length in the section, and
together the section's outer bounding box. Writes OUT_DIR/members.json. The mean axial
crush force of the rail sections is estimated with the Abramowicz-Jones form of the
Wierzbicki-Abramowicz superfolding element for square tubes (see members.json "formula").
Standard library only.
"""
import json
import math
import os
import pickle
import re
import sys

# (member, axis, value, window: {coord: (lo, hi)})   left side of the car (y > 0)
CUTS = [
    ("front_rail_left", "x", -300.0, {"y": (380, 640), "z": (330, 640)}),
    ("front_rail_left", "x", -600.0, {"y": (380, 640), "z": (330, 680)}),
    ("front_rail_left", "x", -900.0, {"y": (350, 640), "z": (330, 680)}),
    ("front_rail_left", "x", -1100.0, {"y": (350, 640), "z": (330, 680)}),
    ("b_pillar_left", "z", 600.0, {"x": (-2800, -2250), "y": (550, 900)}),
    ("b_pillar_left", "z", 900.0, {"x": (-2800, -2250), "y": (550, 900)}),
    ("b_pillar_left", "z", 1200.0, {"x": (-2800, -2250), "y": (500, 900)}),
    ("sill_left", "x", -1800.0, {"y": (600, 900), "z": (150, 450)}),
    ("sill_left", "x", -2200.0, {"y": (600, 900), "z": (150, 450)}),
    ("sill_left", "x", -2700.0, {"y": (600, 900), "z": (150, 450)}),
]
AX = {"x": 0, "y": 1, "z": 2}
# parts that make each member, by name (the window alone also catches doors, seats, trim)
MEMBER_PARTS = {
    "front_rail_left": r"(midrail|midreail|midraril|midrearil|bumpermidrail)",
    "b_pillar_left": r"(bpillar|bpilar|b-pillar|sidepanel$)",
    "sill_left": r"(rocker|sidepanel$)",
}


def flow_stress(curve, lo=0.0, hi=0.2, n=41):
    """Mean of the stress-plastic strain curve over [lo, hi] (linear interpolation)."""
    def at(e):
        # left of the first point: linear extrapolation of the first segment (it meets
        # SIGY at zero plastic strain for these curves); right of the last point: flat
        (a0, s0), (a1, s1) = curve[0], curve[1]
        if e < a0:
            return s0 + (s1 - s0) * (e - a0) / (a1 - a0)
        for (a0, s0), (a1, s1) in zip(curve, curve[1:]):
            if a0 <= e <= a1:
                return s0 + (s1 - s0) * (e - a0) / (a1 - a0) if a1 > a0 else s0
        return curve[-1][1]
    return sum(at(lo + (hi - lo) * i / (n - 1)) for i in range(n)) / n


def main():
    cache, out = sys.argv[1], sys.argv[2]
    d = pickle.load(open(os.path.join(cache, "deck.pickle"), "rb"))
    parts, mats, secs = d["parts"], d["mats"], d["sections"]
    nid, X = d["node_id"], d["xyz"]
    sp, sn = d["shell_pid"], d["shell_n"]

    def structural(pid):
        m = mats.get(parts[pid]["mid"], {})
        return m.get("type") in ("PIECEWISE_LINEAR_PLASTICITY",) and (m.get("ro") or 0) > 7e-9 \
            and (m.get("sigy") or 0) >= 150 and parts[pid]["source"] == "vehicle"
    ok = {p: structural(p) for p in parts}

    res = []
    for (member, axn, c, win) in CUTS:
        a = AX[axn]
        others = [k for k in "xyz" if k != axn]
        rx = re.compile(MEMBER_PARTS[member])
        member_ok = {p for p in parts if ok.get(p) and rx.search(re.sub(r"^\d+_", "", parts[p]["title"]).lower())}
        per = {}
        lo = [1e30, 1e30]; hi = [-1e30, -1e30]
        for e in range(len(sp)):
            pid = sp[e]
            if pid not in member_ok:
                continue
            ids = sn[4 * e:4 * e + 4]
            if ids[2] == ids[3]:
                ids = ids[:3]
            P = []
            for n in ids:
                i = 3 * nid[n]
                P.append((X[i], X[i + 1], X[i + 2]))
            vals = [p[a] - c for p in P]
            if min(vals) > 0 or max(vals) < 0:
                continue
            hits = []
            for k in range(len(P)):
                p, q = P[k], P[(k + 1) % len(P)]
                vp, vq = vals[k], vals[(k + 1) % len(P)]
                if (vp <= 0 < vq) or (vq <= 0 < vp):
                    t = vp / (vp - vq)
                    hits.append(tuple(p[j] + t * (q[j] - p[j]) for j in range(3)))
            if len(hits) < 2:
                continue
            h0, h1 = hits[0], hits[1]
            mid = [(h0[j] + h1[j]) / 2 for j in range(3)]
            if not all(win[k][0] <= mid[AX[k]] <= win[k][1] for k in others):
                continue
            L = math.dist(h0, h1)
            r = per.setdefault(pid, dict(length=0.0))
            r["length"] += L
            for h in (h0, h1):
                for j, k in enumerate(others):
                    v = h[AX[k]]
                    lo[j] = min(lo[j], v); hi[j] = max(hi[j], v)
        rows = []
        for pid, r in sorted(per.items(), key=lambda kv: -kv[1]["length"]):
            m = mats[parts[pid]["mid"]]; s = secs[parts[pid]["secid"]]
            curve = d["curves"].get(m.get("lcss")) or []
            rows.append(dict(pid=pid, name=parts[pid]["title"], t_mm=s.get("t"), sigy_MPa=m.get("sigy"),
                             flow_0_0p2_MPa=round(flow_stress(curve), 1) if curve else None,
                             cowper_symonds=(m.get("c"), m.get("p")), wall_length_mm=round(r["length"], 1)))
        Lsum = sum(r["wall_length_mm"] for r in rows)
        teq = sum(r["wall_length_mm"] * r["t_mm"] for r in rows) / Lsum if Lsum else None
        seq = sum(r["wall_length_mm"] * r["t_mm"] * (r["flow_0_0p2_MPa"] or r["sigy_MPa"]) for r in rows) / \
            sum(r["wall_length_mm"] * r["t_mm"] for r in rows) if Lsum else None
        area = sum(r["wall_length_mm"] * r["t_mm"] for r in rows)
        dims = {others[0]: round(hi[0] - lo[0], 1), others[1]: round(hi[1] - lo[1], 1)}
        res.append(dict(member=member, plane="%s = %g" % (axn, c), window=win, parts=rows,
                        bbox_mm=dims, bbox_lo=lo, bbox_hi=hi, steel_area_mm2=round(area, 1),
                        t_equiv_mm=teq and round(teq, 3), flow_equiv_MPa=seq and round(seq, 1),
                        total_wall_length_mm=round(Lsum, 1)))

    # Mean crush force of the rail sections
    formula = ("Abramowicz & Jones (1984), square tube, static: Pm = 13.06 * sigma0 * t^2 * (C/t)^(1/3), "
               "C = (b + h)/2 for a rectangle (Wierzbicki-Abramowicz superfolding element with the effective "
               "crushing distance); dynamic: x [1 + (eps_rate/D)^(1/q)], eps_rate = 0.33 V / C")
    V = 15646.4  # mm/s, the deck's initial velocity (35 mph)
    for r in res:
        if r["member"] != "front_rail_left" or not r["t_equiv_mm"]:
            continue
        b, h = list(r["bbox_mm"].values())
        C = (b + h) / 2
        t = r["t_equiv_mm"]
        sy = sum(p["wall_length_mm"] * p["t_mm"] * p["sigy_MPa"] for p in r["parts"]) / \
            sum(p["wall_length_mm"] * p["t_mm"] for p in r["parts"])
        rate = 0.33 * V / C
        cs = r["parts"][0]["cowper_symonds"]
        dyn_deck = 1 + (rate / cs[0]) ** (1 / cs[1]) if cs[0] else 1.0
        dyn_mild = 1 + (rate / 40.4) ** (1 / 5)
        Ps_y = 13.06 * sy * t * t * (C / t) ** (1 / 3)
        Ps_f = 13.06 * r["flow_equiv_MPa"] * t * t * (C / t) ** (1 / 3)
        r["crush"] = dict(C_mm=round(C, 1), t_mm=t, sigma_y_equiv_MPa=round(sy, 1),
                          Pm_static_sigy_kN=round(Ps_y / 1000, 1), Pm_static_flow_kN=round(Ps_f / 1000, 1),
                          strain_rate_per_s=round(rate, 1),
                          dyn_factor_deck_cowper_symonds=round(dyn_deck, 3),
                          dyn_factor_mild_steel_D40p4_q5=round(dyn_mild, 3),
                          Pm_dynamic_flow_deck_CS_kN=round(Ps_f * dyn_deck / 1000, 1),
                          Pm_dynamic_flow_mild_CS_kN=round(Ps_f * dyn_mild / 1000, 1))
    # spot welds joining the parts of each member (both nodes on member parts)
    node_part = {}
    want = set(p["pid"] for r in res for p in r["parts"])
    for e in range(len(sp)):
        if sp[e] in want:
            for n in sn[4 * e:4 * e + 4]:
                node_part.setdefault(n, set()).add(sp[e])
    welds = {}
    for r in res:
        ps = set(p["pid"] for p in r["parts"])
        welds.setdefault(r["member"], set()).update(ps)
    wc = {}
    for m, ps in welds.items():
        k = 0
        for (a, b) in d["spotwelds"]:
            if node_part.get(a, set()) & ps and node_part.get(b, set()) & ps:
                k += 1
        wc[m] = k
    json.dump(dict(formula=formula, cuts=res, constrained_spotwelds_between_member_parts=wc),
              open(os.path.join(out, "members.json"), "w"), indent=1)
    for r in res:
        print(r["member"], r["plane"], r["bbox_mm"], "t_eq", r["t_equiv_mm"], "A", r["steel_area_mm2"], r.get("crush", ""))
        for p in r["parts"]:
            print("    ", p["name"], p["t_mm"], p["sigy_MPa"], p["flow_0_0p2_MPa"], p["wall_length_mm"])
    print("spot welds", wc)


if __name__ == "__main__":
    main()
