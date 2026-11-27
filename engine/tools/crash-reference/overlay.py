#!/usr/bin/env python3
"""Overlay Rubble's crash time histories on the NHTSA tests' (the crash reference suite).

    python3 -I engine/tools/crash-reference/overlay.py REF_DIR OUT_DIR RUN.jsonl [RUN.jsonl ...]

REF_DIR: the committed reference numbers, engine/docs/crash-reference/nhtsa/<test>.json
(engine/tools/crash-reference/nhtsa_reference.py). RUN.jsonl: `crash_ref -- S11 --json` output
(one CrashReport a line, with its `series`); the first report of each scenario id is drawn (run
0 when the file starts with it). For each scenario it writes OUT_DIR/<id>-<contact>.svg (and .png
when Google Chrome is installed, rendered headless) with the tests' curves and Rubble's on the
same axes, and prints a "beside the real tests" table (markdown) of the scalars.

Curves: tests: CFC 60 (SAE J211), 1 kHz; Rubble: the suite's 4 ms series (velocity samples,
acceleration their central difference: the mean over 8 ms) and, when the run carries it (CPU
reference), the 2 kHz trace through CFC 60. Frontal: the rear-seat crossmember (Rubble: its
place, the cabin floor behind the B-pillar), X; side: the struck car's CG, Y (+ right).
Standard library only.
"""
import json
import math
import os
import shutil
import subprocess
import sys

PAL = dict(t1="#2563eb", t2="#0891b2", rub="#dc2626", rub2="#f97316", com="#7c3aed", grid="#e5e7eb")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def nice(lo, hi, n=6):
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    x, ticks = math.floor(lo / step) * step, []
    while x <= hi + 1e-9 * step:
        ticks.append(round(x, 10))
        x += step
    return ticks


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(x0, y0, w, h, title, xl, yl, series, xr=None):
    """series: (label, xs, ys, colour, dash, width)"""
    inside = lambda x: xr is None or xr[0] <= x <= xr[1]
    pts = [(x, y) for s in series for x, y in zip(s[1], s[2]) if y is not None and inside(x)]
    if not pts:
        return '<text x="%d" y="%d" font-size="14">%s: no data</text>' % (x0 + 10, y0 + 30, esc(title))
    xlo, xhi = xr or (min(p[0] for p in pts), max(p[0] for p in pts))
    ylo, yhi = min(p[1] for p in pts), max(p[1] for p in pts)
    pad = 0.05 * ((yhi - ylo) or 1)
    ylo, yhi = ylo - pad, yhi + pad
    xt, yt = nice(xlo, xhi), nice(ylo, yhi)
    xlo, xhi, ylo, yhi = min(xt[0], xlo), max(xt[-1], xhi), min(yt[0], ylo), max(yt[-1], yhi)
    L, R, T, B = 70, 15, 30, 45
    pw, ph = w - L - R, h - T - B
    X = lambda x: x0 + L + (x - xlo) / (xhi - xlo) * pw
    Y = lambda y: y0 + T + (yhi - y) / (yhi - ylo) * ph
    o = ['<rect x="%g" y="%g" width="%g" height="%g" fill="white" stroke="#9ca3af"/>' % (x0 + L, y0 + T, pw, ph),
         '<text x="%g" y="%g" font-size="15" font-weight="bold">%s</text>' % (x0 + L, y0 + 20, esc(title))]
    for v in xt:
        if xlo <= v <= xhi:
            o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="%s"/>' % (X(v), X(v), y0 + T, y0 + T + ph, PAL["grid"]))
            o.append('<text x="%g" y="%g" font-size="11" text-anchor="middle">%g</text>' % (X(v), y0 + T + ph + 15, v))
    for v in yt:
        if ylo <= v <= yhi:
            o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="%s"/>' % (x0 + L, x0 + L + pw, Y(v), Y(v), PAL["grid"]))
            o.append('<text x="%g" y="%g" font-size="11" text-anchor="end">%g</text>' % (x0 + L - 5, Y(v) + 4, v))
    if ylo < 0 < yhi:
        o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="#6b7280"/>' % (x0 + L, x0 + L + pw, Y(0), Y(0)))
    o.append('<text x="%g" y="%g" font-size="12" text-anchor="middle">%s</text>' % (x0 + L + pw / 2, y0 + h - 8, esc(xl)))
    o.append('<text transform="translate(%g,%g) rotate(-90)" font-size="12" text-anchor="middle">%s</text>' % (x0 + 16, y0 + T + ph / 2, esc(yl)))
    ly = y0 + T + 14
    o.append('<rect x="%g" y="%g" width="%g" height="%g" fill="white" fill-opacity="0.85" stroke="#d1d5db"/>' % (x0 + L + pw - 305, ly - 13, 300, 15 * len(series) + 4))
    for (label, xs, ys, col, dash, wid) in series:
        p = " ".join("%.1f,%.1f" % (X(a), Y(b)) for a, b in zip(xs, ys) if b is not None and inside(a))
        d = 'stroke-dasharray="6,4"' if dash else ""
        o.append('<polyline fill="none" stroke="%s" stroke-width="%g" %s points="%s"/>' % (col, wid, d, p))
        o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="%s" stroke-width="2" %s/>' % (x0 + L + pw - 300, x0 + L + pw - 275, ly - 4, ly - 4, col, d))
        o.append('<text x="%g" y="%g" font-size="11">%s</text>' % (x0 + L + pw - 270, ly, esc(label)))
        ly += 15
    return "\n".join(o)


def table(x0, y0, w, rows):
    o = ['<text x="%g" y="%g" font-size="15" font-weight="bold">Beside the real tests</text>' % (x0 + 70, y0 + 20)]
    y = y0 + 42
    for r in rows:
        cells = [esc(str(c)) for c in r]
        o.append('<text x="%g" y="%g" font-size="11" font-family="Menlo,monospace">%s</text>' % (x0 + 70, y, " | ".join(cells)))
        y += 15
    return "\n".join(o)


def svg(w, h, title, body):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="Helvetica,Arial,sans-serif">'
            '<rect width="100%%" height="100%%" fill="#f9fafb"/><text x="20" y="28" font-size="18" font-weight="bold">%s</text>%s</svg>') % (w, h, esc(title), body)


def fmt(x, d=2):
    return "-" if x is None else ("%.*f" % (d, x) if isinstance(x, (int, float)) else str(x))


def kn(xs):
    return [None if x is None else x / 1000 for x in xs]


def frontal(rep, refs):
    s, m = rep["series"], rep["metrics"]
    b = s["bodies"][0]
    t = s["t_ms"]
    who = "generated sedan, %.0f kg (Rubble, %s)" % (b["mass_kg"], rep["contact"])
    cols = [PAL["t1"], PAL["t2"]]
    tests = [(r, cols[k % 2], "%s test %d, %.0f kg" % (r["vehicle"]["make_model"].title(), r["test"], r["vehicle"]["test_mass_kg"])) for k, r in enumerate(refs)]
    fine = s.get("fine")
    acc = [(lab + " (CFC 60)", r["curves"]["t_ms"], r["curves"]["ax_g"], c, False, 1.4) for r, c, lab in tests]
    acc.append((who + ", 4 ms", t, b["reference"]["ax_g"], PAL["rub"], False, 2.2))
    if fine:
        acc.append((who + ", 2 kHz CFC 60", fine["t_ms"], fine["bodies"][0]["reference"]["ax_g"], PAL["rub2"], True, 1.4))
    vel = [(lab, r["curves"]["t_ms"], r["curves"]["vx_mps"], c, False, 1.4) for r, c, lab in tests] + [(who, t, b["reference"]["vx"], PAL["rub"], False, 2.2), ("Rubble centre of mass", t, b["com"]["vx"], PAL["com"], True, 1.2)]
    dis = [(lab, r["curves"]["t_ms"], r["curves"]["dx_m"], c, False, 1.4) for r, c, lab in tests] + [(who, t, b["reference"]["dx"], PAL["rub"], False, 2.2)]
    frc = [(lab + " (load cells)", r["curves"]["t_ms"], kn(r["curves"].get("force_n", [])), c, False, 1.4) for r, c, lab in tests if "force_n" in r["curves"]]
    frc.append((who + ": -d(sum m v)/dt", t, kn(s["box_force_n"]), PAL["rub"], False, 2.2))

    def until_stop(ts, xs, ys, stop):
        k = [i for i, x in enumerate(ts) if 0 <= x <= stop + 30]
        return [xs[i] for i in k], [ys[i] for i in k]
    fd = []
    for r, c, lab in tests:
        if "force_n" in r["curves"]:
            xs, ys = until_stop(r["curves"]["t_ms"], r["curves"]["dx_m"], kn(r["curves"]["force_n"]), r["scalars"].get("stop_ms", 100))
            fd.append((lab, xs, ys, c, False, 1.4))
    xs, ys = until_stop(t, b["reference"]["dx"], kn(s["box_force_n"]), m.get("car.ref.stop_ms", 100))
    fd.append((who, xs, ys, PAL["rub"], False, 2.2))
    panels = [("Acceleration, rear-seat crossmember X", "time after contact (ms)", "acceleration (g)", acc, (-10, 200)),
              ("Velocity X", "time after contact (ms)", "velocity (m/s)", vel, (-10, 200)),
              ("Displacement X from contact", "time after contact (ms)", "displacement (m)", dis, (-10, 200)),
              ("Wall force", "time after contact (ms)", "force (kN)", frc, (-10, 200)),
              ("Wall force against crossmember displacement (to t_m + 30 ms)", "displacement (m)", "force (kN)", fd, None)]
    g = lambda k: m.get("car." + k)
    rows = [("quantity", "Rubble") + tuple("test %d" % r["test"] for r, _, _ in tests)]
    sc = lambda r, k: r["scalars"].get(k)
    rows += [
        ("mass (kg)", fmt(b["mass_kg"], 0)) + tuple(fmt(r["vehicle"]["test_mass_kg"], 0) for r, _, _ in tests),
        ("speed (km/h)", fmt(3.6 * m["car.speed_before"], 1)) + tuple(fmt(r["closing_speed_kmh"], 1) for r, _, _ in tests),
        ("peak decel, 4 ms estimator (g)", fmt(g("ref.peak_decel_g"), 1)) + tuple(fmt(sc(r, "as_rubble_samples")["peak_decel_g"], 1) for r, _, _ in tests),
        ("peak decel, CFC 60 (g)", fmt(g("ref.cfc60_peak_decel_g"), 1)) + tuple(fmt(sc(r, "peak_decel_g"), 1) for r, _, _ in tests),
        ("time of peak, CFC 60 (ms)", fmt(g("ref.cfc60_peak_ms"), 1)) + tuple(fmt(sc(r, "peak_ms"), 1) for r, _, _ in tests),
        ("t_m, v = 0 (ms)", fmt(g("ref.stop_ms"), 1)) + tuple(fmt(sc(r, "stop_ms"), 1) for r, _, _ in tests),
        ("dynamic crush (m)", fmt(g("ref.dyn_crush"), 3)) + tuple(fmt(sc(r, "dyn_crush_m"), 3) for r, _, _ in tests),
        ("mean decel to t_m (g)", fmt(g("ref.mean_decel_g"), 1)) + tuple(fmt(sc(r, "mean_decel_g"), 1) for r, _, _ in tests),
        ("rebound, t_m-200 ms (m/s)", fmt(g("ref.rebound"), 2)) + tuple(fmt(sc(r, "rebound_mps"), 2) for r, _, _ in tests),
        ("restitution", fmt(g("ref.restitution"), 3)) + tuple(fmt(sc(r, "restitution"), 3) for r, _, _ in tests),
        ("residual crush (m)", fmt(g("crush_residual"), 3)) + tuple("%s (length), %s (max)" % (fmt(r["vehicle"].get("length_loss_mm", 0) / 1000, 3), fmt((r["vehicle"].get("max_crush_mm") or 0) / 1000, 3)) for r, _, _ in tests),
        ("peak wall force (kN)", fmt((m.get("all.box_force_peak_n") or 0) / 1000, 0)) + tuple(fmt(sc(r, "peak_force_n") and sc(r, "peak_force_n") / 1000, 0) for r, _, _ in tests),
        ("time of peak force (ms)", fmt(m.get("all.box_force_peak_ms"), 1)) + tuple(fmt(sc(r, "peak_force_ms"), 1) for r, _, _ in tests),
    ]
    return panels, rows


def side(rep, refs, pole):
    s, m = rep["series"], rep["metrics"]
    car = next(x for x in s["bodies"] if x["name"] == "car")
    t = s["t_ms"]
    who = "generated sedan, %.0f kg (Rubble, %s)" % (car["mass_kg"], rep["contact"])
    cols = [PAL["t1"], PAL["t2"]]
    tests = [(r, cols[k % 2], "%s test %d, %.0f kg" % (r["vehicle"]["make_model"].title(), r["test"], r["vehicle"]["test_mass_kg"])) for k, r in enumerate(refs)]
    c = lambda r, k: r["curves"][k]
    acc = [(lab + " CG (CFC 60)", c(r, "t_ms"), c(r, "ay_g"), col, False, 1.4) for r, col, lab in tests] + [(who + " COM, 4 ms", t, car["com"]["ay_g"], PAL["rub"], False, 2.2)]
    if s.get("fine"):
        fb = next(x for x in s["fine"]["bodies"] if x["name"] == "car")
        acc.append((who + " COM, 2 kHz CFC 60", s["fine"]["t_ms"], fb["com"]["ay_g"], PAL["rub2"], True, 1.4))
    vel = [(lab + " CG", c(r, "t_ms"), c(r, "vy_mps"), col, False, 1.4) for r, col, lab in tests] + [(who + " COM", t, car["com"]["vy"], PAL["rub"], False, 2.2)]
    dis = [(lab + " CG", c(r, "t_ms"), c(r, "dy_m"), col, False, 1.4) for r, col, lab in tests] + [(who + " COM", t, car["com"]["dy"], PAL["rub"], False, 2.2)]
    g = lambda k: m.get("car." + k)
    sc = lambda r, k: r["scalars"].get(k)
    rows = [("quantity", "Rubble") + tuple("test %d" % r["test"] for r, _, _ in tests)]
    rows += [("struck car mass (kg)", fmt(car["mass_kg"], 0)) + tuple(fmt(r["vehicle"]["test_mass_kg"], 0) for r, _, _ in tests)]
    if pole:
        a = math.radians(75)
        along = lambda dx, dy: [None if p is None or q is None else p * math.cos(a) - q * math.sin(a) for p, q in zip(dx, dy)]
        frc = [(lab + " (load cells)", c(r, "t_ms"), kn(c(r, "force_n")), col, False, 1.4) for r, col, lab in tests if "force_n" in r["curves"]]
        frc.append((who + ": -d(sum m v)/dt", t, kn(s["box_force_n"]), PAL["rub"], False, 2.2))
        fd = [(lab, along(c(r, "dx_m"), c(r, "dy_m"))[:160], kn(c(r, "force_n"))[:160], col, False, 1.4) for r, col, lab in tests if "force_n" in r["curves"]]
        k = [i for i, x in enumerate(t) if 0 <= x <= 150]
        da = along(car["com"]["dx"], car["com"]["dy"])
        fd.append((who + " COM", [da[i] for i in k], [s["box_force_n"][i] / 1000 for i in k], PAL["rub"], False, 2.2))
        panels = [("Lateral acceleration (Y, + right)", "time after contact (ms)", "acceleration (g)", acc, (-10, 200)),
                  ("Lateral velocity", "time after contact (ms)", "velocity (m/s)", vel, (-10, 200)),
                  ("Lateral displacement from contact", "time after contact (ms)", "displacement (m)", dis, (-10, 200)),
                  ("Pole force", "time after contact (ms)", "force (kN)", frc, (-10, 200)),
                  ("Pole force against CG displacement along the 75 deg travel (to 150 ms)", "displacement (m)", "force (kN)", fd, None)]
        rows += [
            ("speed (km/h)", fmt(3.6 * m["car.speed_before"], 1)) + tuple(fmt(r["closing_speed_kmh"], 1) for r, _, _ in tests),
            ("CG lateral peak, 4 ms estimator (g)", fmt(g("com.ay_peak_g"), 1)) + tuple(fmt(sc(r, "as_rubble_samples_ay_peak_g"), 1) for r, _, _ in tests),
            ("CG lateral peak, CFC 60 (g)", fmt(g("com.cfc60_ay_peak_g"), 1)) + tuple(fmt(sc(r, "ay_peak_g"), 1) for r, _, _ in tests),
            ("CG dv_y by 150 ms (m/s)", fmt(g("com.dvy_150ms"), 2)) + tuple(fmt(sc(r, "vy_change_150ms_mps"), 2) for r, _, _ in tests),
            ("CG dv_y largest (m/s)", fmt(g("com.dvy_peak"), 2)) + tuple(fmt(sc(r, "vy_change_peak_mps"), 2) for r, _, _ in tests),
            ("peak pole force (kN)", fmt((m.get("all.box_force_peak_n") or 0) / 1000, 0)) + tuple(fmt((sc(r, "peak_force_n") or 0) / 1000, 0) for r, _, _ in tests),
            ("time of peak force (ms)", fmt(m.get("all.box_force_peak_ms"), 1)) + tuple(fmt(sc(r, "peak_force_ms"), 1) for r, _, _ in tests),
            ("residual side crush (m)", fmt(g("side_crush_left_residual"), 3)) + tuple(fmt(r["vehicle"]["max_crush_mm"] / 1000, 3) for r, _, _ in tests),
            ("peak side crush, dynamic (m)", fmt(g("side_crush_left_peak"), 3)) + ("-",) * len(tests),
            ("CG travel at stop (m)", "-") + tuple(fmt(sc(r, "travel_dyn_crush_m"), 3) for r, _, _ in tests),
        ]
    else:
        bar = next(x for x in s["bodies"] if x["name"] == "barrier")
        # The tests' MDB: its CG X velocity change at 150 ms (a mark: the curves are not held).
        bacc = [("test %d MDB CG X dv at 150 ms (%.0f kg)" % (r["test"], r["barrier"]["test_mass_kg"]), [135, 165], [sc(r, "mdb_dvx_150ms_mps")] * 2, col, False, 3.0) for r, col, lab in tests]
        panels = [("Struck car lateral acceleration (Y, + right)", "time after contact (ms)", "acceleration (g)", acc, (-10, 200)),
                  ("Struck car lateral velocity", "time after contact (ms)", "velocity (m/s)", vel, (-10, 200)),
                  ("Struck car lateral displacement", "time after contact (ms)", "displacement (m)", dis, (-10, 200)),
                  ("Barrier velocity change along its axis (Rubble)", "time after contact (ms)", "dv (m/s)",
                   [("barrier COM, Rubble (%.0f kg)" % bar["mass_kg"], t, [v - bar["com"]["vx"][next(i for i, x in enumerate(t) if x >= 0)] for v in bar["com"]["vx"]], PAL["rub"], False, 2.2)] + bacc, (-10, 200))]
        rows += [
            ("barrier mass (kg)", fmt(bar["mass_kg"], 0)) + tuple(fmt(r["barrier"]["test_mass_kg"], 0) for r, _, _ in tests),
            ("barrier speed (km/h)", fmt(3.6 * m["barrier.speed_before"], 1)) + tuple(fmt(r["closing_speed_kmh"], 1) for r, _, _ in tests),
            ("CG lateral peak, 4 ms estimator (g)", fmt(g("com.ay_peak_g"), 1)) + tuple(fmt(sc(r, "as_rubble_samples_ay_peak_g"), 1) for r, _, _ in tests),
            ("CG lateral peak, CFC 60 (g)", fmt(g("com.cfc60_ay_peak_g"), 1)) + tuple(fmt(sc(r, "ay_peak_g"), 1) for r, _, _ in tests),
            ("CG dv_y largest (m/s)", fmt(g("com.dvy_peak"), 2)) + tuple(fmt(sc(r, "vy_change_peak_mps"), 2) for r, _, _ in tests),
            ("  when (ms)", fmt(g("com.dvy_peak_ms"), 0)) + tuple(fmt(sc(r, "vy_change_peak_ms"), 0) for r, _, _ in tests),
            ("CG dv_y by 150 ms (m/s)", fmt(g("com.dvy_150ms"), 2)) + tuple(fmt(sc(r, "vy_change_150ms_mps"), 2) for r, _, _ in tests),
            ("barrier dv_x by 150 ms (m/s)", fmt(m.get("barrier.com.dvx_150ms"), 2)) + tuple(fmt(sc(r, "mdb_dvx_150ms_mps"), 2) for r, _, _ in tests),
            ("residual side crush (m)", fmt(g("side_crush_left_residual"), 3)) + tuple(fmt(r["vehicle"]["max_crush_mm"] / 1000, 3) for r, _, _ in tests),
            ("peak side crush, dynamic (m)", fmt(g("side_crush_left_peak"), 3)) + ("-",) * len(tests),
        ]
    return panels, rows


def render(path_svg, title, panels, rows):
    W, H = 1500, 1420
    pw, ph = W // 2, 440
    body = []
    for k, p in enumerate(panels):
        x0, y0 = (k % 2) * pw, 40 + (k // 2) * ph
        body.append(panel(x0 + 5, y0, pw - 10, ph - 10, p[0], p[1], p[2], p[3], p[4]))
    k = len(panels)
    body.append(table((k % 2) * pw + 5, 40 + (k // 2) * ph, pw - 10, rows))
    open(path_svg, "w").write(svg(W, H, title, "\n".join(body)))
    png = path_svg[:-4] + ".png"
    if os.path.exists(CHROME):
        try:
            subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--screenshot=" + png, "--window-size=%d,%d" % (W, H), "file://" + os.path.abspath(path_svg)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        except (OSError, subprocess.SubprocessError):
            pass
    return png if os.path.exists(png) else path_svg


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    ref_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    refs = [json.load(open(os.path.join(ref_dir, f))) for f in sorted(os.listdir(ref_dir)) if f.endswith(".json")]
    for run in sys.argv[3:]:
        seen = set()
        for line in open(run):
            line = line.strip()
            if not line.startswith("{"):
                continue
            rep = json.loads(line)
            if rep["id"] in seen or not rep.get("series"):
                continue
            seen.add(rep["id"])
            mine = sorted((r for r in refs if r["scenario"] == rep["id"]), key=lambda r: -r["test"])
            if not mine:
                continue
            cfg = mine[0]["configuration"]
            if cfg == "VEHICLE INTO BARRIER":
                panels, rows = frontal(rep, mine)
            else:
                panels, rows = side(rep, mine, cfg == "VEHICLE INTO POLE")
            title = "%s (%s, run %d): Rubble's generated sedan against NHTSA tests %s" % (rep["id"], rep["contact"], rep["run"], ", ".join(str(r["test"]) for r in mine))
            p = render(os.path.join(out, "%s-%s.svg" % (rep["id"], rep["contact"])), title, panels, rows)
            print("## %s, %s contact, run %d\n" % (rep["id"], rep["contact"], rep["run"]))
            print("| " + " | ".join(rows[0]) + " |")
            print("|" + "---|" * len(rows[0]))
            for r in rows[1:]:
                print("| " + " | ".join(r) + " |")
            print("\nwrote %s\n" % os.path.abspath(p))


if __name__ == "__main__":
    main()
