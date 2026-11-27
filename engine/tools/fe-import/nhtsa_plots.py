#!/usr/bin/env python3
"""Plot NHTSA crash-test signals (EV5 curve files) and compute pulse metrics.

    python3 -I engine/tools/fe-import/nhtsa_plots.py NHTSA_ROOT OUT_DIR [RUBBLE_S02_JSON]

NHTSA_ROOT holds one folder per test, nhtsa-<testno>/, each with channels.json (curve list
with sensor attachment, axis, units, sample rate) and ev/v<test>.<curve> two-column text
files (time s, value in G or N). For each test it writes OUT_DIR/nhtsa_<test>.svg (four
panels) and adds the test's metrics to OUT_DIR/nhtsa_metrics.json. Convert the SVGs to PNG
with any SVG renderer (the spike used headless Chrome). Standard library only.

Processing (stated so the numbers can be checked):
- bias: the mean over the pre-trigger samples [-20 ms, 0) is subtracted from each channel;
- filter: SAE J211-1 CFC 60 (second-order Butterworth run forward then backward, the
  J211 Appendix C coefficients with wd = 2*pi*CFC*2.0775) for accelerations and forces;
- integration: trapezoidal from t = 0, initial velocity = the test's closing speed for the
  moving vehicle (0 for a struck, stationary vehicle); SAE J211 axes (+X forward, +Y right);
- wall force: the sum of all barrier/pole load-cell channels, CFC 60.
"""
import json
import math
import os
import sys

G = 9.80665


def read_ev(path):
    t, v = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                t.append(float(p[0])); v.append(float(p[1]))
    return t, v


def cfc(x, dt, cls=60.0):
    wd = 2 * math.pi * cls * 2.0775
    wa = math.sin(wd * dt / 2) / math.cos(wd * dt / 2)
    den = 1 + math.sqrt(2) * wa + wa * wa
    a0 = wa * wa / den; a1 = 2 * a0; a2 = a0
    b1 = -2 * (wa * wa - 1) / den; b2 = (-1 + math.sqrt(2) * wa - wa * wa) / den

    def run(s):
        y = [0.0] * len(s)
        y[0] = s[0]; y[1] = s[1] if len(s) > 1 else s[0]
        for i in range(2, len(s)):
            y[i] = a0 * s[i] + a1 * s[i - 1] + a2 * s[i - 2] + b1 * y[i - 1] + b2 * y[i - 2]
        return y
    f = run(x)
    return run(f[::-1])[::-1]


def debias(t, v):
    pre = [x for tt, x in zip(t, v) if -0.02 <= tt < 0]
    m = sum(pre) / len(pre) if pre else 0.0
    return [x - m for x in v]


def integrate(t, a, v0):
    """a in m/s^2 from t=0 on; returns v (m/s) and s (m) on the same grid (0 before t=0)."""
    v = [v0] * len(t); s = [0.0] * len(t)
    i0 = next(i for i, tt in enumerate(t) if tt >= 0)
    for i in range(i0 + 1, len(t)):
        h = t[i] - t[i - 1]
        v[i] = v[i - 1] + 0.5 * h * (a[i] + a[i - 1])
        s[i] = s[i - 1] + 0.5 * h * (v[i] + v[i - 1])
    return v, s


# ---------------------------------------------------------------- SVG
PAL = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]


def nice(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m; break
    a = math.floor(lo / step) * step
    ticks = []
    x = a
    while x <= hi + 1e-9 * step:
        ticks.append(round(x, 10)); x += step
    return ticks


def panel(x0, y0, w, h, title, xlabel, ylabel, series, xr=None, yr=None, notes=()):
    """series: list of (label, xs, ys, color, dashed)"""
    xs_all = [x for s in series for x in s[1]]; ys_all = [y for s in series for y in s[2]]
    if not xs_all:
        return '<text x="%d" y="%d" font-size="14">%s: no data</text>' % (x0 + 10, y0 + 30, title)
    xlo, xhi = xr or (min(xs_all), max(xs_all))
    ylo, yhi = yr or (min(ys_all), max(ys_all))
    pad = 0.05 * (yhi - ylo or 1); ylo -= pad; yhi += pad
    xt, yt = nice(xlo, xhi), nice(ylo, yhi)
    xlo, xhi = min(xt[0], xlo), max(xt[-1], xhi); ylo, yhi = min(yt[0], ylo), max(yt[-1], yhi)
    L, R, T, B = 70, 15, 30, 45
    pw, ph = w - L - R, h - T - B
    X = lambda x: x0 + L + (x - xlo) / (xhi - xlo) * pw
    Y = lambda y: y0 + T + (yhi - y) / (yhi - ylo) * ph
    o = ['<rect x="%g" y="%g" width="%g" height="%g" fill="white" stroke="#999"/>' % (x0 + L, y0 + T, pw, ph)]
    o.append('<text x="%g" y="%g" font-size="15" font-weight="bold">%s</text>' % (x0 + L, y0 + 20, title))
    for v in xt:
        if xlo <= v <= xhi:
            o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="#e5e7eb"/>' % (X(v), X(v), y0 + T, y0 + T + ph))
            o.append('<text x="%g" y="%g" font-size="11" text-anchor="middle">%g</text>' % (X(v), y0 + T + ph + 15, v))
    for v in yt:
        if ylo <= v <= yhi:
            o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="#e5e7eb"/>' % (x0 + L, x0 + L + pw, Y(v), Y(v)))
            o.append('<text x="%g" y="%g" font-size="11" text-anchor="end">%g</text>' % (x0 + L - 5, Y(v) + 4, v))
    if ylo < 0 < yhi:
        o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="#6b7280"/>' % (x0 + L, x0 + L + pw, Y(0), Y(0)))
    o.append('<text x="%g" y="%g" font-size="12" text-anchor="middle">%s</text>' % (x0 + L + pw / 2, y0 + h - 8, xlabel))
    o.append('<text transform="translate(%g,%g) rotate(-90)" font-size="12" text-anchor="middle">%s</text>'
             % (x0 + 16, y0 + T + ph / 2, ylabel))
    ly = y0 + T + 14
    for (label, xs, ys, col, dash) in series:
        step = max(1, len(xs) // 1500)
        pts = " ".join("%.1f,%.1f" % (X(a), Y(b)) for a, b in list(zip(xs, ys))[::step]
                       if xlo <= a <= xhi)
        o.append('<polyline fill="none" stroke="%s" stroke-width="1.6" %s points="%s"/>'
                 % (col, 'stroke-dasharray="6,4"' if dash else "", pts))
        o.append('<line x1="%g" x2="%g" y1="%g" y2="%g" stroke="%s" stroke-width="2" %s/>'
                 % (x0 + L + pw - 230, x0 + L + pw - 205, ly - 4, ly - 4, col, 'stroke-dasharray="6,4"' if dash else ""))
        o.append('<text x="%g" y="%g" font-size="11">%s</text>' % (x0 + L + pw - 200, ly, label))
        ly += 15
    for k, n in enumerate(notes):
        o.append('<text x="%g" y="%g" font-size="11" fill="#374151">%s</text>' % (x0 + L + 8, y0 + T + ph - 10 - 14 * (len(notes) - 1 - k), n))
    return "\n".join(o)


def svg_doc(w, h, body, title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="Helvetica,Arial,sans-serif">'
            '<rect width="100%%" height="100%%" fill="#f9fafb"/>'
            '<text x="20" y="28" font-size="18" font-weight="bold">%s</text>%s</svg>') % (w, h, title, body)


# ---------------------------------------------------------------- per test
def load(test_dir, pred):
    ch = json.load(open(os.path.join(test_dir, "channels.json")))
    out = []
    for c in ch["tsv_channels"]:
        if pred(c):
            t, v = read_ev(os.path.join(test_dir, c["file"]))
            out.append((c, t, v))
    return ch, out


def accel_ms2(c, t, v):
    v = debias(t, v)
    dt = t[1] - t[0]
    f = cfc(v, dt)
    scale = G if c["units"].upper().startswith("G") else 1.0
    return [x * scale for x in f]


def avg(series):
    n = min(len(s) for s in series)
    return [sum(s[i] for s in series) / len(series) for i in range(n)]


def analyse(test_dir, rubble=None):
    ch = json.load(open(os.path.join(test_dir, "channels.json")))
    tno = ch["test_no"]; title = ch["title"]; cfg = ch["configuration"]; V = ch["closing_speed_kmh"] / 3.6
    up = lambda c, k: k in (c.get("comment") or "").upper()
    is_acc = lambda c: c["sensor_type"].startswith("ACCEL") and (c.get("data_status") or "").startswith("AS MEASURED") \
        or (c["sensor_type"].startswith("ACCEL") and (c.get("data_status") or "") == "COMPUTED")
    met = dict(test=tno, title=title, configuration=cfg, closing_speed_kmh=ch["closing_speed_kmh"])
    panels = []
    if "BARRIER" in title.upper() or cfg == "VEHICLE INTO BARRIER" and "POLE" not in title.upper():
        cm = lambda c: (c.get("comment") or "").upper().strip()
        _, acc = load(test_dir, lambda c: is_acc(c) and "REAR SEAT CROSSMEMBER" in cm(c) and c["axis"].startswith("X")
                      and not cm(c).endswith(("XR", "RAX")))
        _, eng = load(test_dir, lambda c: is_acc(c) and "ENGINE" in cm(c) and c["axis"].startswith("X")
                      and (c.get("data_status") or "").startswith("AS MEASURED"))
        _, lc = load(test_dir, lambda c: c["sensor_type"].startswith("LOAD CELL"))
        t = acc[0][1]
        a = avg([accel_ms2(*x) for x in acc])
        v, s = integrate(t, a, V)
        ag = [x / G for x in a]
        series_a = [("rear-seat crossmember X, mean of %d (CFC60)" % len(acc), t, ag, PAL[0], False)]
        for k, x in enumerate(eng):
            series_a.append((x[0]["comment"][:28].lower() + " (CFC60)", x[1], [y / G for y in accel_ms2(*x)], PAL[2 + k], False))
        i_pk = min(range(len(ag)), key=lambda i: ag[i] if t[i] >= 0 else 1e9)
        i_v0 = next((i for i in range(len(v)) if t[i] > 0 and v[i] <= 0), None)
        met.update(channels_used=[x[0]["file"] for x in acc], sample_rate_hz=acc[0][0]["sample_rate_hz"],
                   record_s=[t[0], t[-1]], peak_decel_g=round(-ag[i_pk], 1), t_peak_ms=round(1000 * t[i_pk], 1))
        if i_v0:
            tm = t[i_v0]
            met.update(t_m_ms=round(tm * 1000, 1), dynamic_crush_m=round(s[i_v0], 3),
                       mean_decel_to_tm_g=round(V / tm / G, 1))
            iend = min(len(v) - 1, next(i for i, tt in enumerate(t) if tt >= 0.2))
            met.update(rebound_mps_at_200ms=round(-min(v[i_v0:iend]), 2),
                       restitution=round(-min(v[i_v0:iend]) / V, 3))
        panels.append(("Vehicle acceleration", "time (s)", "acceleration (g)", series_a, (-0.01, 0.2)))
        panels.append(("Velocity (integrated, v0 = %.2f m/s)" % V, "time (s)", "velocity (m/s)",
                       [("rear-seat crossmember", t, v, PAL[0], False)], (-0.01, 0.2)))
        if lc:
            F = None
            for x in lc:
                f = cfc(debias(x[1], x[2]), x[1][1] - x[1][0])
                F = f if F is None else [p + q for p, q in zip(F, f)]
            Fk = [abs(x) / 1000 if False else x / 1000 for x in F]
            sign = 1 if max(Fk) >= -min(Fk) else -1
            Fk = [sign * x for x in Fk]
            tf = lc[0][1]
            met.update(wall_load_cells=len(lc), peak_wall_force_kN=round(max(Fk), 1),
                       t_peak_force_ms=round(1000 * tf[max(range(len(Fk)), key=lambda i: Fk[i])], 1))
            # impulse check: integral of F dt vs momentum change m*dv (needs the test mass)
            panels.append(("Total barrier force (%d load cells, CFC60)" % len(lc), "time (s)", "force (kN)",
                           [("wall force", tf, Fk, PAL[1], False)], (-0.01, 0.2)))
            n = min(len(s), len(Fk))
            idx = [i for i in range(n) if 0 <= t[i] <= (t[i_v0] if i_v0 else 0.15) + 0.03]
            panels.append(("Wall force vs vehicle displacement", "displacement of rear-seat crossmember (m)",
                           "force (kN)", [("test %d" % tno, [s[i] for i in idx], [Fk[i] for i in idx], PAL[1], False)], None))
        else:
            panels.append(("Displacement (integrated)", "time (s)", "displacement (m)",
                           [("rear-seat crossmember", t, s, PAL[0], False)], (-0.01, 0.2)))
        if rubble and abs(ch["closing_speed_kmh"] - 56) < 2:
            notes = ["Rubble generated sedan, suite row S02 (different car: %s kg, %s km/h):" % (rubble.get("mass_kg", "?"), rubble.get("speed_kmh", "?")),
                     "peak %.1f g, mean %.1f g over %.0f ms, crush %.3f m (scalars only, no time history)"
                     % (rubble["decel_peak_g"], rubble["decel_mean_g"], rubble["pulse_ms"], rubble["crush_peak"])]
            p = panels[0]
            series = list(p[3])
            series.append(("Rubble S02 peak (other car)", [0, 0.2], [-rubble["decel_peak_g"]] * 2, PAL[4], True))
            series.append(("Rubble S02 mean (other car)", [0, rubble["pulse_ms"] / 1000], [-rubble["decel_mean_g"]] * 2, PAL[3], True))
            panels[0] = (p[0], p[1], p[2], series, p[4], notes)
    else:
        # oblique (car struck by a moving barrier), side MDB (car struck), side pole (car moves sideways)
        side = "SIDE" in title.upper()
        pole = "POLE" in title.upper()
        car_v = 1 if pole else 2
        ax = "Y" if side else "X"
        _, cg = load(test_dir, lambda c: is_acc(c) and c.get("vehicle_no") == car_v and "VEHICLE CG" in (c.get("sensor_attachment") or "")
                     and c["axis"].startswith(("X", "Y")) and not (c.get("comment") or "").upper().endswith(("RX", "RY"))
                     and "ANGULAR" not in (c.get("comment") or "").upper() and "REDUNDANT" not in (c.get("comment") or "").upper())
        _, mdb = load(test_dir, lambda c: is_acc(c) and c.get("vehicle_no") == 1 and not pole
                      and "VEHICLE CG" in (c.get("sensor_attachment") or "") and c["axis"].startswith("X")
                      and not (c.get("comment") or "").upper().endswith("RX") and "ANGULAR" not in (c.get("comment") or "").upper())
        _, lc = load(test_dir, lambda c: c["sensor_type"].startswith("LOAD CELL"))
        series_a, vel = [], []
        for k, x in enumerate(cg):
            a = accel_ms2(*x)
            axis = x[0]["axis"][0]
            v0 = (-V if (pole and axis == "Y") else 0.0)   # pole: the car moves left (-Y) into the pole
            v, s = integrate(x[1], a, v0)
            series_a.append(("car CG %s (CFC60)" % axis, x[1], [y / G for y in a], PAL[k], False))
            vel.append(("car CG %s velocity" % axis, x[1], v, PAL[k], False))
            ag = [y / G for y in a]
            i_pk = max(range(len(ag)), key=lambda i: abs(ag[i]) if x[1][i] >= 0 else 0)
            met["car_cg_%s_peak_g" % axis] = round(ag[i_pk], 1)
            met["car_cg_%s_dv_mps_at_150ms" % axis] = round(v[next(i for i, tt in enumerate(x[1]) if tt >= 0.15)] - v0, 2)
            met["sample_rate_hz"] = x[0]["sample_rate_hz"]; met["record_s"] = [x[1][0], x[1][-1]]
            if pole and axis == "Y":
                pole_s = (x[1], s)
        for k, x in enumerate(mdb):
            a = accel_ms2(*x)
            series_a.append(("moving barrier CG X (CFC60)", x[1], [y / G for y in a], PAL[3], True))
            met["barrier_cg_x_peak_g"] = round(min(y / G for y in a), 1)
        panels.append(("Accelerations", "time (s)", "acceleration (g)", series_a, (-0.01, 0.2)))
        panels.append(("Velocity change (integrated)", "time (s)", "velocity (m/s)", vel, (-0.01, 0.2)))
        if lc:
            F = None
            for x in lc:
                f = cfc(debias(x[1], x[2]), x[1][1] - x[1][0])
                F = f if F is None else [p + q for p, q in zip(F, f)]
            Fk = [x / 1000 for x in F]
            sign = 1 if max(Fk) >= -min(Fk) else -1
            Fk = [sign * x for x in Fk]
            met.update(pole_load_cells=len(lc), peak_pole_force_kN=round(max(Fk), 1))
            panels.append(("Total pole force (%d load cells, CFC60)" % len(lc), "time (s)", "force (kN)",
                           [("pole force", lc[0][1], Fk, PAL[1], False)], (-0.01, 0.2)))
            if pole and cg:
                tt, s = pole_s
                n = min(len(s), len(Fk))
                idx = [i for i in range(n) if 0 <= tt[i] <= 0.12]
                panels.append(("Pole force vs car CG displacement toward the pole", "CG displacement (m)", "force (kN)",
                               [("test %d" % tno, [-s[i] for i in idx], [Fk[i] for i in idx], PAL[1], False)], None))
        if len(panels) < 4:
            panels.append(("(no wall/pole load cells in this test)", "", "", [], None))
    return met, panels


def render(tno, title, panels, out):
    W, H = 1400, 900
    body = []
    for k, p in enumerate(panels[:4]):
        x0, y0 = (k % 2) * W // 2, 40 + (k // 2) * (H - 40) // 2
        body.append(panel(x0 + 5, y0, W // 2 - 10, (H - 40) // 2 - 5, p[0], p[1], p[2], p[3],
                          xr=p[4] if len(p) > 4 else None, notes=p[5] if len(p) > 5 else ()))
    path = os.path.join(out, "nhtsa_%d.svg" % tno)
    open(path, "w").write(svg_doc(W, H, "\n".join(body), "NHTSA test %d: %s" % (tno, title)))
    return path


def main():
    root, out = sys.argv[1], sys.argv[2]
    rubble = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    os.makedirs(out, exist_ok=True)
    allm = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("nhtsa-"):
            continue
        d = os.path.join(root, name)
        met, panels = analyse(d, rubble)
        p = render(met["test"], met["title"], panels, out)
        allm.append(met)
        print(json.dumps(met))
        print("wrote", p)
    json.dump(allm, open(os.path.join(out, "nhtsa_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
