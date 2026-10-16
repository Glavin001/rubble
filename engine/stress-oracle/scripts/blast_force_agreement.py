#!/usr/bin/env python3
"""Force-level agreement of two quasi-static answers on the same scene pack under gravity:
Blast's ExtStressSolver (`blast_ladder_bench forces`, PhysX repository) and ours
(`examples/blast_forces`, the f64 static solve the Blast-equivalent configuration runs).

Blast reports per bond (compression, tension, shear) in Pa from its bond force F and
moment M with its own formula (NvBlastExtStressFormula.h, bendGainMax = 3):
    n = F.n / A, shear = |F_perp| / A + |M.n| / A * min(4.81 / sqrt(A), 3),
    bend = |M_perp| / A * min(6 / sqrt(A), 3), tension = max(0, n + bend),
    compression = max(0, bend - n),
with A the asset's bond area. The same formula is applied to our bond forces (the asset's
area and normal), so the two are compared stress for stress, bond by bond (matched by
their two nodes; ours keeps the asset's node order). Blast reports bonds in its asset's
order, which `NvBlastCreateAsset` sorts by (lower node, higher node), not in the order
they were given (checked on a column with known forces: `blast_ladder_bench column`).

    blast_force_agreement.py DIR PACK [PACK ...]    (DIR holds ours_PACK.json, blast_PACK.json)
"""
import json
import math
import statistics as st
import sys


def zup_to_yup(v):
    # Packs are Y-up, ours Z-up: ours (x, y, z) = pack (x, -z, y).
    return (v[0], v[2], -v[1])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def perp(v, n):
    d = dot(v, n)
    return math.sqrt(max(0.0, sum((x - d * y) ** 2 for x, y in zip(v, n))))


def blast_formula(f, m, n, area):
    nn = dot(f, n) / area
    shear = perp(f, n) / area + abs(dot(m, n)) / area * min(4.81 / math.sqrt(max(area, 1e-6)), 3.0)
    bend = perp(m, n) / area * min(6.0 / math.sqrt(max(area, 1e-6)), 3.0)
    return (max(0.0, bend - nn), max(0.0, nn + bend), shear)


def asset_order(bonds):
    """(stress index, bond) in Blast's asset order: bonds sorted by (lower, higher) node."""
    order = sorted(range(len(bonds)), key=lambda i: (min(bonds[i][0], bonds[i][1]), max(bonds[i][0], bonds[i][1])))
    return [(k, bonds[i]) for k, i in enumerate(order)]


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def compare(ours, blast, which):
    ob = {}
    for a, b, area, normal, f, m in ours["bonds"]:
        ob[(min(a, b), max(a, b))] = (zup_to_yup(f), zup_to_yup(m), area)
    rows = []
    for i, (n0, n1, area, normal, _c) in asset_order(blast["bonds"]):
        k = (min(n0, n1), max(n0, n1))
        if k not in ob or area <= 0:
            continue
        f, m, our_area = ob[k]
        mine = blast_formula(f, m, normal, area)
        theirs = blast[which][i]
        rows.append((mine, theirs))
    # Peak fibre stress (the larger of compression and tension) and shear, over bonds
    # Blast loads above 1 % of its largest.
    out = {}
    for label, pick in [("peak", lambda s: max(s[0], s[1])), ("shear", lambda s: s[2])]:
        mx = max(pick(t) for _, t in rows) or 1.0
        sel = [(pick(o), pick(t)) for o, t in rows if pick(t) > 0.01 * mx]
        ratios = sorted(o / t for o, t in sel if t > 0)
        out[label] = {
            "bonds": len(sel),
            "pearson": pearson([o for o, _ in sel], [t for _, t in sel]),
            "median_ratio": st.median(ratios) if ratios else float("nan"),
            "within_2x": sum(1 for r in ratios if 0.5 <= r <= 2.0) / max(1, len(ratios)),
            "sum_ratio": sum(o for o, _ in sel) / max(1e-30, sum(t for _, t in sel)),
        }
    out["matched"] = len(rows)
    return out


def main():
    d, packs = sys.argv[1], sys.argv[2:]
    print("| pack | Blast frames to converge | matched bonds | peak fibre: Pearson / median ours/Blast / within 2x | shear: Pearson / median / within 2x | first Blast frame vs converged Blast: peak Pearson / median |")
    print("|---|---|---|---|---|---|")
    for p in packs:
        ours = json.load(open(f"{d}/ours_{p}.json"))
        blast = json.load(open(f"{d}/blast_{p}.json"))
        c = compare(ours, blast, "last")
        # Blast's own first frame against its converged answer (what one capped frame returns).
        first = [((x[0], x[1], x[2]), y) for x, y in zip(blast["first"], blast["last"])]
        mx = max(max(y[0], y[1]) for _, y in first) or 1.0
        sel = [(max(x[0], x[1]), max(y[0], y[1])) for x, y in first if max(y[0], y[1]) > 0.01 * mx]
        fr = sorted(a / b for a, b in sel if b > 0)
        pk, sh = c["peak"], c["shear"]
        print(
            f"| {p} | {blast['converged_at']} | {c['matched']} | {pk['pearson']:.2f} / {pk['median_ratio']:.2f} / {100 * pk['within_2x']:.0f} % "
            f"| {sh['pearson']:.2f} / {sh['median_ratio']:.2f} / {100 * sh['within_2x']:.0f} % | {pearson([a for a, _ in sel], [b for _, b in sel]):.2f} / {st.median(fr):.2f} |"
        )


main()
