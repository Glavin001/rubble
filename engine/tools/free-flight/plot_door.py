"""Plots a torn-off part's path from `door_orbit`'s CSV (engine-world/examples/door_orbit.rs):
left, relative to the body's centre of mass (what the lab's chase camera shows); right,
in the world (both centres of mass), side by side as one SVG. Also prints the part's and
the body's centre-of-mass accelerations less gravity (what pushes each).

    python3 tools/free-flight/plot_door.py out/free-flight/door-off/door_orbit.csv sedan_door_FL 3.0 5.5 out.svg
"""
import csv
import math
import sys

path, part, t0, t1, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
rows = [r for r in csv.DictReader(open(path)) if r["part"] == part and t0 <= float(r["t"]) <= t1]
f = lambda r, k: float(r[k])
rel = [(f(r, "rel_x"), f(r, "rel_y"), f(r, "rel_z")) for r in rows]
body = [(f(r, "body_x"), f(r, "body_y"), f(r, "body_z")) for r in rows]
door = [tuple(b[i] + d[i] for i in range(3)) for b, d in zip(body, rel)]

W, H, PAD = 560, 560, 30


def panel(ox, series, title, a, b):
    xs = [p[a] for s, _ in series for p in s]
    ys = [p[b] for s, _ in series for p in s]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 1.1
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    sx = lambda x: ox + W / 2 + (x - cx) / span * (W - 2 * PAD)
    sy = lambda y: H / 2 - (y - cy) / span * (H - 2 * PAD)
    o = [f'<rect x="{ox}" y="0" width="{W}" height="{H}" fill="white" stroke="lightgray"/>',
         f'<text x="{ox + 10}" y="18">{title} (grid: none; {span:.1f} m across)</text>']
    for s, colour in series:
        pts = " ".join(f"{sx(p[a]):.1f},{sy(p[b]):.1f}" for p in s)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
        o.append(f'<circle cx="{sx(s[0][a]):.1f}" cy="{sy(s[0][b]):.1f}" r="4" fill="{colour}"/>')
        for k in range(0, len(s), 15):  # a tick every 0.25 s
            o.append(f'<circle cx="{sx(s[k][a]):.1f}" cy="{sy(s[k][b]):.1f}" r="2" fill="{colour}"/>')
    return o


# Side view: the plane of the largest motion (x or z against y).
ax = 0 if (max(p[0] for p in door) - min(p[0] for p in door)) > (max(p[2] for p in door) - min(p[2] for p in door)) else 2
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{2 * W + 10}" height="{2 * W + 10}" font-family="sans-serif" font-size="13">',
       '<rect width="100%" height="100%" fill="white"/>']
svg += panel(0, [(rel, "#d62728"), ([(0, 0, 0)] * 2, "black")], f"{part} relative to the body's centre (side, {'xz'[ax // 2]}–y)", ax, 1)
svg += panel(W + 10, [(door, "#d62728"), (body, "black")], "world frame (side): part red, body black", ax, 1)
svg.append(f'<text x="10" y="{H + 22}">{part}, t {t0:.2f}–{t1:.2f} s; dots every 0.25 s, large dot = start</text>')
pa = [f(r, "accel_less_g") for r in rows[1:]]
ba = [f(r, "body_accel_less_g") for r in rows[1:]]
med = lambda v: sorted(v)[len(v) // 2]
line = (f"centre-of-mass acceleration less g: {part} median {med(pa):.2f}, max {max(pa):.1f} m/s²; "
        f"body median {med(ba):.2f}, max {max(ba):.1f} m/s²")
svg.append(f'<text x="10" y="{H + 42}">{line}</text></svg>')
open(out, "w").write("\n".join(svg))
print(line)
for r in rows[::15]:
    print(f"  t {f(r, 't'):5.2f}  part |a-g| {f(r, 'accel_less_g'):7.2f}  body |a-g| {f(r, 'body_accel_less_g'):7.2f}  gap {f(r, 'nearest_body'):5.2f} m")
