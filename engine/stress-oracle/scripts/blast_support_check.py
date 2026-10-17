#!/usr/bin/env python3
"""Equilibrium check of the two quasi-static answers under gravity (see
blast_force_agreement.py): the vertical load the bonds touching the supports carry,
against the structure's weight (the pack's node masses, g = 9.81).

Ours: the vertical force each support bond transmits (its force and the support's side).
Blast reports only (compression, tension, shear) per bond, from n = F.n / A and the
bending fibre stress b: tension = max(0, n + b), compression = max(0, b - n). So
|n| <= max(compression, tension) bounds each support bond's axial force, and for a
vertical bond |N| <= A max(c, t): an upper bound on the load Blast delivers to its supports
(shear is bounded likewise by A * shear for horizontal bonds).

    blast_support_check.py DIR PACK [PACK ...]   (DIR as for blast_force_agreement.py; ours from --blast-inputs)
"""
import json
import sys

d, packs = sys.argv[1], sys.argv[2:]
root = __file__.rsplit("/", 2)[0]
print("| pack | weight (kN) | ours: vertical load into supports (kN) | Blast: upper bound (kN) | Blast bound / weight |")
print("|---|---|---|---|---|")
for p in packs:
    pack = json.load(open(f"{root}/packs/{p}.json"))["scenario"]
    sup = {i for i, n in enumerate(pack["nodes"]) if n["mass"] <= 0}
    weight = sum(n["mass"] for n in pack["nodes"]) * 9.81
    ours = json.load(open(f"{d}/ours_{p}.json"))
    blast = json.load(open(f"{d}/blast_{p}.json"))
    into = 0.0
    for a, b, area, normal, f, m in ours["bonds"]:
        # f acts on chunk a (Z-up): a support at a receives f, at b receives -f.
        if a in sup and b not in sup:
            into += -f[2]
        elif b in sup and a not in sup:
            into += f[2]
    bonds = blast["bonds"]
    order = sorted(range(len(bonds)), key=lambda i: (min(bonds[i][0], bonds[i][1]), max(bonds[i][0], bonds[i][1])))
    bound = 0.0
    for k, i in enumerate(order):
        n0, n1, area, normal, _ = bonds[i]
        if (n0 in sup) == (n1 in sup):
            continue
        c, t, s = blast["last"][k]
        ny = abs(normal[1])
        # Vertical share: axial along the normal's vertical part, shear across it.
        bound += area * (max(c, t) * ny + s * (1.0 - ny * ny) ** 0.5)
    print(f"| {p} | {weight / 1e3:.0f} | {into / 1e3:.0f} | {bound / 1e3:.0f} | {bound / weight:.2f} |")
