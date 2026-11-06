#!/usr/bin/env python3
"""SVG plots of vehicle_scale_bench CSV lines (no dependencies).

    python3 plot.py out_dir a.csv [b.csv ...]

CSV columns (engine-world examples/vehicle_scale_bench.rs): label, scenario, x, cars,
bodies, p50 ms, p95 ms, p99 ms, explicit steps. One SVG per scenario: the p50 of each
label as a line (p95 dotted), against x.
"""
import csv
import sys
from collections import defaultdict

COLORS = ["#2a6fdb", "#d9480f", "#2b8a3e", "#862e9c", "#495057"]
TITLES = {
    "fleet": "Cars driving at fixed density",
    "density": "128 cars: spread, traffic, pile-up",
    "parked": "1,000 parked + 10 driving",
    "bodies": "Rigid bodies at 1 % awake, 10 driving cars",
    "map": "Map side (m), nothing active",
}
XLABEL = {
    "fleet": "cars",
    "density": "case",
    "parked": "parked + driving",
    "bodies": "rigid bodies",
    "map": "map side (m)",
}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.reader(f):
                if len(r) < 8 or r[0].startswith("#"):
                    continue
                rows.append(r)
    return rows


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def plot(scenario, rows, out):
    by = defaultdict(list)
    xs = []
    for r in rows:
        by[r[0]].append(r)
        if r[2] not in xs:
            xs.append(r[2])
    numeric = all(x.replace(".", "").isdigit() for x in xs)
    if numeric:
        xs.sort(key=float)
    W, H, L, R, T, B = 640, 380, 70, 150, 40, 50
    ymax = max(float(r[6]) for r in rows) * 1.1 or 1.0
    pos = {x: L + (W - L - R) * (i + 0.5) / len(xs) for i, x in enumerate(xs)}

    def py(v):
        return T + (H - T - B) * (1 - v / ymax)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="22" font-size="14" font-weight="bold">{esc(TITLES.get(scenario, scenario))}</text>']
    for k in range(6):
        v = ymax * k / 5
        y = py(v)
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e9ecef"/>')
        s.append(f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    s.append(f'<text x="16" y="{(T + H - B) / 2}" transform="rotate(-90 16 {(T + H - B) / 2})" text-anchor="middle">ms per World step (120 Hz)</text>')
    for x in xs:
        s.append(f'<text x="{pos[x]:.1f}" y="{H - B + 18}" text-anchor="middle">{esc(x)}</text>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 10}" text-anchor="middle">{esc(XLABEL.get(scenario, "x"))}</text>')
    for i, (label, rs) in enumerate(by.items()):
        c = COLORS[i % len(COLORS)]
        rs = sorted(rs, key=lambda r: xs.index(r[2]))
        for col, dash in ((5, ""), (6, ' stroke-dasharray="3 3"')):
            pts = " ".join(f"{pos[r[2]]:.1f},{py(float(r[col])):.1f}" for r in rs)
            s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        for r in rs:
            s.append(f'<circle cx="{pos[r[2]]:.1f}" cy="{py(float(r[5])):.1f}" r="3" fill="{c}"/>')
            s.append(f'<text x="{pos[r[2]] + 5:.1f}" y="{py(float(r[5])) - 5:.1f}" fill="{c}" font-size="10">{float(r[5]):.2f}</text>')
        y = T + 16 * i
        s.append(f'<line x1="{W - R + 12}" x2="{W - R + 32}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>')
        s.append(f'<text x="{W - R + 36}" y="{y + 4}">{esc(label)} p50</text>')
    y = T + 16 * len(by)
    s.append(f'<line x1="{W - R + 12}" x2="{W - R + 32}" y1="{y}" y2="{y}" stroke="#495057" stroke-width="2" stroke-dasharray="3 3"/>')
    s.append(f'<text x="{W - R + 36}" y="{y + 4}">p95</text>')
    s.append("</svg>")
    with open(f"{out}/{scenario}.svg", "w") as f:
        f.write("\n".join(s))


def main():
    out = sys.argv[1]
    rows = load(sys.argv[2:])
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    for sc, rs in by.items():
        plot(sc, rs, out)


if __name__ == "__main__":
    main()
