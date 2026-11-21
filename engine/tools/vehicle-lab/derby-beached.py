#!/usr/bin/env python3
"""Summarise a derby probe's log (vehicle-lab-record.mjs --probe --eval derby-probe.js):
per car the seconds it was beached (in the round, upright, its AI or player asking to move,
under 2 km/h, at most one wheel touching anything) and how it ended.

    python3 tools/vehicle-lab/derby-beached.py out/derby4-on.log
"""
import json
import sys

rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("{") and '"cars"' in line:
        rows.append(json.loads(line))
if not rows:
    sys.exit("no derby rows")
n = len(rows[0]["cars"])
beached = [0.0] * n
spells = [[] for _ in range(n)]
still = ["hunting", "taking the ramp", "reversing out", "wriggling free", "backing off", "player", "charging"]
for a, b in zip(rows, rows[1:]):
    dt = b["t"] - a["t"]
    for v, c in enumerate(a["cars"]):
        plan, kmh, grounded = c[0], c[1], c[5] if len(c) > 5 else None
        if plan.startswith("OUT") or grounded is None:
            continue
        if abs(kmh) < 2 and grounded <= 1:
            beached[v] += dt
            spells[v].append(a["t"])
end = rows[-1]
print(f"{rows[0]['t']:.1f}..{end['t']:.1f} s, {len(rows)} readings")
for v in range(n):
    c = end["cars"][v]
    first = f", from {spells[v][0]:.1f} s" if spells[v] else ""
    print(f"car {v + 1}: beached {beached[v]:.1f} s{first}; end: {c[0]}, {c[1]} km/h, damage {c[2]} %, wheels touching {c[5] if len(c) > 5 else '?'}")
print(f"total beached {sum(beached):.1f} car-seconds")
