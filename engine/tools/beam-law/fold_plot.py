#!/usr/bin/env python3
"""SVG plots of the fold measurements (`tools/beam-law/fold.sh`; no dependencies). Per variant and
scenario, the left bays over time: (1) the span's shortening along the car (solid) beside the
chords' mean shortening (dashed): a bay that folds shortens both together, one that turns or
shears loses span and keeps its chords; (2) the axial mode's force against its capacity.

    python3 -I tools/beam-law/fold_plot.py [variant ...]   → out/beam-law/fold/<variant>-<S>.svg
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOLD = os.path.normpath(os.path.join(HERE, "..", "..", "out", "beam-law", "fold"))
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
W, H, PAD = 720, 230, 52


def panel(rows, bays, key, y0, title, ymax, ylabel):
    ts = [float(r["t_ms"]) for r in rows]
    tmin, tmax = 0.0, max(ts)
    x = lambda t: PAD + (t - tmin) / (tmax - tmin) * (W - 2 * PAD)
    y = lambda v: y0 + H - 30 - min(max(v, 0.0), ymax) / ymax * (H - 60)
    o = [f'<text x="{PAD}" y="{y0 + 18}" class="t">{title}</text>']
    for k in range(5):
        v = ymax * k / 4
        o.append(f'<line x1="{PAD}" x2="{W - PAD}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/><text x="{PAD - 6}" y="{y(v) + 4:.1f}" class="a" text-anchor="end">{v:.2f}</text>')
    for t in range(0, int(tmax) - 25, 20):
        o.append(f'<text x="{x(t):.1f}" y="{y0 + H - 12}" class="a" text-anchor="middle">{t}</text>')
    o.append(f'<text x="{W - PAD}" y="{y0 + H - 12}" class="a" text-anchor="end">ms after impact</text><text x="{PAD}" y="{y0 + 32}" class="a">{ylabel}</text>')
    for i, b in enumerate(bays):
        for kk, dash in key:
            pts = [(float(r["t_ms"]), kk(r)) for r in rows if r["bay"] == b and float(r["t_ms"]) >= 0]
            d = " ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in pts)
            o.append(f'<polyline points="{d}" fill="none" stroke="{COLORS[i % 3]}" stroke-width="2" {dash}/>')
        o.append(f'<rect x="{W - PAD - 120}" y="{y0 + 24 + 16 * i}" width="10" height="10" rx="2" fill="{COLORS[i % 3]}"/><text x="{W - PAD - 104}" y="{y0 + 33 + 16 * i}" class="a">{b[:-1]}</text>')
    return o


def main():
    tags = [a for a in sys.argv[1:]] or sorted(d for d in os.listdir(FOLD) if os.path.isdir(os.path.join(FOLD, d)))
    for t in tags:
        for s in ["S11", "S12"]:
            p = os.path.join(FOLD, t, f"{s}-edge-0-bays.csv")
            if not os.path.exists(p):
                continue
            rows = [r for r in csv.DictReader(open(p)) if r["bay"].endswith("l")]
            bays = sorted({r["bay"] for r in rows})
            span = lambda r: float(r["span0_m"]) - float(r["span_m"])
            chord = lambda r: float(r["chord0_mean_m"]) - float(r["chord_mean_m"])
            ax = lambda r: -float(r["G_axial_N"]) / float(r["cap_axial_N"]) if float(r["G_axial_N"]) < 0 else 0.0
            o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{2 * H + 20}" viewBox="0 0 {W} {2 * H + 20}">',
                 '<style>.t{font:600 13px sans-serif;fill:#1a1a19}.a{font:11px sans-serif;fill:#5f5e5a}.g{stroke:#e4e3df;stroke-width:1}</style>',
                 f'<rect width="{W}" height="{2 * H + 20}" fill="#ffffff"/>']
            o += panel(rows, bays, [(span, ""), (chord, 'stroke-dasharray="5 4"')], 0, f"{t}, {s}: span shortening (solid) and chords' mean shortening (dashed), left bays", 0.6, "m")
            o += panel(rows, bays, [(ax, "")], H + 20, f"{t}, {s}: axial force over its capacity (1 = the cell folds)", 1.0, "ratio")
            o.append("</svg>")
            out = os.path.join(FOLD, f"{t}-{s}.svg")
            open(out, "w").write("\n".join(o))
            print(out)


if __name__ == "__main__":
    main()
