#!/usr/bin/env python3
"""Derive the crash reference suite's NHTSA reference numbers from the downloaded test data.

    python3 -I engine/tools/crash-reference/nhtsa_reference.py camry DATA_ROOT OUT_DIR
    python3 -I engine/tools/crash-reference/nhtsa_reference.py recheck PULSES_DIR
    python3 -I engine/tools/crash-reference/nhtsa_reference.py oblique DATA_ROOT OUT_DIR

`oblique`: the same for the three Camry oblique tests (9124, 8790 left; 9121 right; an RMDB
into the standing car turned 15 deg): the car's CG velocity change, the RMDB's, the car's
velocity along the RMDB's travel, their closing speed and common-velocity time, the momentum
balance, yaw where an angular rate sensor exists, and the report's crush tables (`oblique`
below).

`camry`: DATA_ROOT holds one folder per NHTSA test, nhtsa-<test>/ (the layout of the local,
gitignored inspiration/ccsa-camry-2012/: channels.json, ev/v<test>.<curve> two-column text
curves in NHTSA units, api/*.json, report/*.pdf). For each test the suite uses it writes
OUT_DIR/<test>.json: our own numbers derived from the public-domain US-government data, small
(curves decimated to 1 kHz after filtering), with the source test number and channel names.
`recheck`: PULSES_DIR holds the per-curve TSV files (v0NNNNtsv.093/.094) of the 56 km/h sedan
tests in engine/docs/crash-reference-targets.md section 8; prints the same scalars for them.

This is a tool, not a test dependency: the suite reads the committed JSON, never the data.
Standard library only (pdftotext, when installed, reads the side tests' crush tables).

Processing (the same for every test, so the numbers can be checked):
- bias: the mean over the pre-trigger samples [-20 ms, 0) is subtracted from each channel;
- filter: SAE J211-1 CFC 60 (a two-pole Butterworth run forward then backward, the J211
  Appendix C coefficients, wd = 2 pi CFC 2.0775, each pass starting at rest at its first
  sample) for accelerations and load cells;
- axes: SAE J211 vehicle axes as recorded (+X forward, +Y right); a frontal test's car moves
  +X at the test speed, a pole test's at the test speed along 75 deg (cos 75 forward, sin 75
  toward the struck left side, i.e. -Y), a struck car starts at rest;
- integration: trapezoidal from t = 0 (first contact, the record's zero);
- frontal location: the rear-seat crossmember X accelerometers, left and right (not the
  redundant RAX/XR ones), each filtered, then averaged; side tests: the struck car's CG;
- wall or pole force: the sum of every load cell, each filtered; positive pushing the car;
- scalars: peak deceleration (largest -a after contact) and its time; t_m, the first time v
  reaches 0 (linear between samples) and the displacement then (dynamic crush) and v0/t_m
  (mean deceleration, g); rebound: -min v from t_m to 200 ms after contact, restitution =
  rebound / v0 (the location's velocity along X; both read the same way on Rubble's curves by
  engine-softbody's crash_reference::metrics::series); peak force and its time; impulse
  int F dt from 0 to 200 ms against m dv to 200 ms;
- side tests: the struck car's CG lateral velocity (+Y right) at its largest and at 150 ms,
  the MDB's CG X velocity change, the reports' per-level maximum exterior crush;
- decimated: every curve after filtering, sampled at 1 kHz (the nearest sample) from -10 to
  250 ms; "as Rubble samples": the CFC 60 velocity sampled every 4 ms and differentiated by
  central differences, the suite's own estimator, so that the filter's effect on the peak is
  shown on the test's own record (frontal: deceleration along X; side: the CG's lateral
  acceleration, signed).
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys

G = 9.80665
REBOUND_UNTIL = 0.200
CURVE_FROM, CURVE_TO = -0.010, 0.250


# ------------------------------------------------------------------ signal
def read_ev(path):
    t, v = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                try:
                    t.append(float(p[0])); v.append(float(p[1]))
                except ValueError:
                    break
    return t, v


def cfc(x, dt, cls=60.0):
    wd = 2 * math.pi * cls * 2.0775
    wa = math.tan(wd * dt / 2)
    den = 1 + math.sqrt(2) * wa + wa * wa
    a0 = wa * wa / den; a1 = 2 * a0; a2 = a0
    b1 = -2 * (wa * wa - 1) / den; b2 = (-1 + math.sqrt(2) * wa - wa * wa) / den

    def run(s):
        x1 = x2 = y1 = y2 = s[0]
        out = []
        for xi in s:
            y = a0 * xi + a1 * x1 + a2 * x2 + b1 * y1 + b2 * y2
            x2, x1, y2, y1 = x1, xi, y1, y
            out.append(y)
        return out
    return run(run(x)[::-1])[::-1]


def debias(t, v):
    pre = [x for tt, x in zip(t, v) if -0.02 <= tt < 0]
    m = sum(pre) / len(pre) if pre else 0.0
    return [x - m for x in v]


def integrate_from_zero(t, a, v0):
    """Trapezoid from t = 0 (v0 there, constant before); returns v and s (s = 0 at t = 0)."""
    i0 = next(i for i, tt in enumerate(t) if tt >= 0)
    v = [v0] * len(t); s = [0.0] * len(t)
    for i in range(i0 + 1, len(t)):
        h = t[i] - t[i - 1]
        v[i] = v[i - 1] + 0.5 * h * (a[i] + a[i - 1])
        s[i] = s[i - 1] + 0.5 * h * (v[i] + v[i - 1])
    for i in range(i0 - 1, -1, -1):
        s[i] = s[i + 1] - (t[i + 1] - t[i]) * v0
    return v, s


def at(t, y, x):
    for i in range(1, len(t)):
        if t[i] >= x:
            f = (x - t[i - 1]) / (t[i] - t[i - 1]) if t[i] > t[i - 1] else 0.0
            return y[i - 1] + f * (y[i] - y[i - 1])
    return y[-1]


def accel(test_dir, c):
    t, v = read_ev(os.path.join(test_dir, c["file"]))
    v = debias(t, v)
    scale = G if c["units"].upper().startswith("G") else 1.0
    return t, [x * scale for x in cfc(v, t[1] - t[0])]


def mean_curves(curves):
    n = min(len(c[1]) for c in curves)
    return curves[0][0][:n], [sum(c[1][i] for c in curves) / len(curves) for i in range(n)]


def pulse(t, a, v, s, v0):
    """The scalars of a frontal pulse (a, v, s along X; v0 the speed at contact)."""
    after = [i for i in range(len(t)) if t[i] >= 0]
    ip = max(after, key=lambda i: -a[i])
    out = dict(peak_decel_g=-a[ip] / G, peak_ms=1000 * t[ip])
    j = next((i for i in after if v[i] <= 0), None)
    if j is not None:
        tm = t[j - 1] + (t[j] - t[j - 1]) * v[j - 1] / (v[j - 1] - v[j]) if v[j - 1] > 0 else t[j]
        rb = -min([v[i] for i in range(j, len(t)) if t[i] <= REBOUND_UNTIL] + [0.0])
        out.update(stop_ms=1000 * tm, dyn_crush_m=at(t, s, tm), mean_decel_g=v0 / tm / G,
                   rebound_mps=rb, restitution=rb / v0)
    return out


def as_rubble_samples(t, v, step=0.004):
    """The CFC 60 velocity sampled every 4 ms from contact and differentiated by central
    differences (Rubble's estimator): the peak deceleration and its time."""
    ts = [k * step for k in range(int(0.25 / step) + 1)]
    vs = [at(t, v, x) for x in ts]
    a = [(vs[k + 1] - vs[k - 1]) / (2 * step) for k in range(1, len(ts) - 1)]
    k = max(range(len(a)), key=lambda i: -a[i])
    return dict(peak_decel_g=-a[k] / G, peak_ms=1000 * ts[k + 1])


def as_rubble_lateral(t, v, step=0.004):
    """The same estimator on a lateral velocity: the largest-magnitude acceleration (signed, g)."""
    ts = [k * step for k in range(int(0.25 / step) + 1)]
    vs = [at(t, v, x) for x in ts]
    a = [(vs[k + 1] - vs[k - 1]) / (2 * step) for k in range(1, len(ts) - 1)]
    k = max(range(len(a)), key=lambda i: abs(a[i]))
    return a[k] / G


def decimate(t, ys, rate=1000.0):
    out_t, out = [], [[] for _ in ys]
    k = int(round(CURVE_FROM * rate))
    dt = t[1] - t[0]
    while k / rate <= CURVE_TO + 1e-9:
        x = k / rate
        i = int(round((x - t[0]) / dt))
        if 0 <= i < len(t):
            out_t.append(round(1000 * x, 3))
            for o, y in zip(out, ys):
                o.append(None if y is None else float("%.5g" % y[i]))
        k += 1
    return out_t, out


# ------------------------------------------------------------------ test metadata
def api(test_dir, name):
    p = os.path.join(test_dir, "api", name)
    if not os.path.exists(p):
        return {}
    r = json.load(open(p)).get("results") or [{}]
    return r[0] if isinstance(r, list) and r else {}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def crush_levels(test_dir):
    """The report's 'maximum exterior crush' table (level, description, height, crush,
    distance from impact, mm), read with pdftotext when it is installed."""
    exe = shutil.which("pdftotext")
    rep = os.path.join(test_dir, "report")
    if not exe or not os.path.isdir(rep):
        return None
    pdfs = sorted(f for f in os.listdir(rep) if f.lower().endswith(".pdf"))
    if not pdfs:
        return None
    try:
        text = subprocess.run([exe, "-layout", os.path.join(rep, pdfs[0]), "-"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pat = re.compile(r"^\s*([1-5])\s+(Sill Top|Occupant Hip Point|Mid Door|Window Sill|Window Top)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
    rows, seen = [], set()
    for line in text.splitlines():
        m = pat.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            rows.append(dict(level=int(m.group(1)), where=m.group(2), height_mm=int(m.group(3)), max_crush_mm=int(m.group(4)), from_impact_mm=int(m.group(5))))
    return rows or None


def vehicle_scalars(test_dir, veh):
    d = api(test_dir, "vehicle-detail-info-veh%d.json" % veh)
    prof = [num(d.get("damageProfileDistances" + k)) for k in ("One", "Two", "Three", "Four", "Five", "Six")]
    out = dict(
        make_model="%s %s %s" % (d.get("modelYear"), d.get("vehicleMake"), d.get("vehicleModel")),
        test_mass_kg=num(d.get("vehicleTestWeight")), curb_mass_kg=num(d.get("vehicleCurbWeight")),
        length_mm=num(d.get("vehicleLength")), width_mm=num(d.get("vehicleWidth")),
        max_crush_mm=num(d.get("maximumCrushDistance")), crush_width_mm=num(d.get("totalLengthofIndentation")),
    )
    if any(prof):
        out["crush_profile_c1_c6_mm"] = prof
    pre, post = num(d.get("pretestTotalLengthofVehicleatcenterline")), num(d.get("postTestTotalLengthofVehicleatcenterline"))
    if pre and post:
        out["length_loss_mm"] = pre - post
    return out


# ------------------------------------------------------------------ per configuration
def channels(test_dir):
    return json.load(open(os.path.join(test_dir, "channels.json")))


def usable(c):
    st = (c.get("data_status") or "")
    return c["sensor_type"].startswith("ACCEL") and (st.startswith("AS MEASURED") or st == "COMPUTED")


def frontal(test_dir, ch, ref):
    V = ch["closing_speed_kmh"] / 3.6
    cm = lambda c: (c.get("comment") or "").upper().strip()
    xs = [c for c in ch["tsv_channels"] if usable(c) and "REAR SEAT CROSSMEMBER" in cm(c) and c["axis"].startswith("X") and not cm(c).endswith(("XR", "RAX"))]
    each = [accel(test_dir, c) for c in xs]
    t, a = mean_curves(each)
    v, s = integrate_from_zero(t, a, V)
    ref.update(location="rear-seat crossmember X, mean of left and right", channels=[c["file"].split("/")[-1] + ": " + c["comment"] for c in xs])
    sc = pulse(t, a, v, s, V)
    # Each side alone (the spread a location's choice makes).
    sides = []
    for c, (tc, ac) in zip(xs, each):
        vc, sc_ = integrate_from_zero(tc, ac, V)
        sides.append(dict(channel=c["file"].split("/")[-1], **{k: round(x, 4) for k, x in pulse(tc, ac, vc, sc_, V).items()}))
    lc = [c for c in ch["tsv_channels"] if c["sensor_type"].startswith("LOAD CELL")]
    F = None
    if lc:
        for c in lc:
            tf, f = read_ev(os.path.join(test_dir, c["file"]))
            f = cfc(debias(tf, f), tf[1] - tf[0])
            F = f if F is None else [p + q for p, q in zip(F, f)]
        if max(F) < -min(F):
            F = [-x for x in F]
        n = min(len(F), len(t))
        F = F[:n]
        ip = max((i for i in range(n) if t[i] >= 0), key=lambda i: F[i])
        imp = sum(0.5 * (F[i] + F[i - 1]) * (t[i] - t[i - 1]) for i in range(1, n) if 0 < t[i] <= REBOUND_UNTIL)
        sc.update(peak_force_n=F[ip], peak_force_ms=1000 * t[ip], impulse_ns_to_200ms=imp)
        ref["load_cells"] = len(lc)
    sc["as_rubble_samples"] = {k: round(x, 3) for k, x in as_rubble_samples(t, v).items()}
    ref["scalars"] = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in sc.items()}
    ref["scalars_each_side"] = sides
    m = ref["vehicle"]["test_mass_kg"]
    if F is not None and m:
        ref["scalars"]["m_dv_ns_to_200ms"] = round(m * (V - at(t, v, REBOUND_UNTIL)), 1)
    ct, (ca, cv, cs, cf) = decimate(t, [[x / G for x in a], v, s, F])
    ref["curves"] = dict(t_ms=ct, ax_g=ca, vx_mps=cv, dx_m=cs)
    if F is not None:
        ref["curves"]["force_n"] = cf


def side(test_dir, ch, ref, pole):
    V = ch["closing_speed_kmh"] / 3.6
    car = 1 if pole else 2
    cm = lambda c: (c.get("comment") or "").upper()
    cg = {c["axis"][0]: c for c in ch["tsv_channels"] if usable(c) and c.get("vehicle_no") == car and "VEHICLE CG" in (c.get("sensor_attachment") or "") and "MDB" not in cm(c) and c["axis"][0] in "XY"}
    v0 = dict(X=V * math.cos(math.radians(75)), Y=-V * math.sin(math.radians(75))) if pole else dict(X=0.0, Y=0.0)
    curves, sc = {}, {}
    for ax in "XY":
        t, a = accel(test_dir, cg[ax])
        v, s = integrate_from_zero(t, a, v0[ax])
        curves[ax] = (t, a, v, s)
    ref.update(location="struck car's CG accelerometer, X and Y", channels=[cg[k]["file"].split("/")[-1] + ": " + cg[k]["comment"] for k in "XY"])
    t, ay, vy, sy = curves["Y"]
    after = [i for i in range(len(t)) if 0 <= t[i] <= 0.25]
    iv = max(after, key=lambda i: abs(vy[i] - v0["Y"]))
    ia = max(after, key=lambda i: abs(ay[i]))
    sc.update(ay_peak_g=ay[ia] / G, ay_peak_ms=1000 * t[ia], vy_change_peak_mps=vy[iv] - v0["Y"], vy_change_peak_ms=1000 * t[iv],
              vy_change_150ms_mps=at(t, vy, 0.15) - v0["Y"], vy_150ms_mps=at(t, vy, 0.15),
              as_rubble_samples_ay_peak_g=as_rubble_lateral(t, vy))
    if pole:
        # Along the travel (75 deg): the velocity component toward the pole.
        tx, ax_, vx, sx = curves["X"]
        n = min(len(tx), len(t))
        ux, uy = math.cos(math.radians(75)), -math.sin(math.radians(75))
        vt = [vx[i] * ux + vy[i] * uy for i in range(n)]
        st = [sx[i] * ux + sy[i] * uy for i in range(n)]
        at_ = [ax_[i] * ux + ay[i] * uy for i in range(n)]
        j = next((i for i in range(n) if t[i] >= 0 and vt[i] <= 0), None)
        if j is not None:
            tm = t[j - 1] + (t[j] - t[j - 1]) * vt[j - 1] / (vt[j - 1] - vt[j])
            rb = -min([vt[i] for i in range(j, n) if t[i] <= REBOUND_UNTIL] + [0.0])
            sc.update(travel_stop_ms=1000 * tm, travel_dyn_crush_m=at(t[:n], st, tm), travel_rebound_mps=rb, travel_restitution=rb / V)
        ip = max(range(n), key=lambda i: -at_[i] if t[i] >= 0 else -1e9)
        sc.update(travel_peak_decel_g=-at_[ip] / G, travel_peak_ms=1000 * t[ip])
    else:
        mdb = [c for c in ch["tsv_channels"] if usable(c) and c.get("vehicle_no") == 1 and "MDB CG X" in cm(c)]
        if mdb:
            tb, ab = accel(test_dir, mdb[0])
            vb, _ = integrate_from_zero(tb, ab, 0.0)
            sc.update(mdb_dvx_150ms_mps=at(tb, vb, 0.15), mdb_ax_peak_g=min(ab) / G)
            ref["channels"].append(mdb[0]["file"].split("/")[-1] + ": " + mdb[0]["comment"])
    lc = [c for c in ch["tsv_channels"] if c["sensor_type"].startswith("LOAD CELL")]
    F = None
    if lc:
        for c in lc:
            tf, f = read_ev(os.path.join(test_dir, c["file"]))
            f = cfc(debias(tf, f), tf[1] - tf[0])
            F = f if F is None else [p + q for p, q in zip(F, f)]
        if max(F) < -min(F):
            F = [-x for x in F]
        n = min(len(F), len(t))
        F = F[:n]
        ip = max((i for i in range(n) if t[i] >= 0), key=lambda i: F[i])
        sc.update(peak_force_n=F[ip], peak_force_ms=1000 * t[ip])
        ref["load_cells"] = len(lc)
    ref["scalars"] = {k: round(x, 4) for k, x in sc.items()}
    lv = crush_levels(test_dir)
    if lv:
        ref["exterior_crush_levels"] = lv
    tx, ax_, vx, sx = curves["X"]
    n = min(len(tx), len(t))
    ys = [[x / G for x in ax_[:n]], [x / G for x in ay[:n]], vx[:n], vy[:n], sx[:n], sy[:n]] + ([F] if F is not None else [])
    ct, out = decimate(t[:n], ys)
    ref["curves"] = dict(t_ms=ct, ax_g=out[0], ay_g=out[1], vx_mps=out[2], vy_mps=out[3], dx_m=out[4], dy_m=out[5])
    if F is not None:
        ref["curves"]["force_n"] = out[6]


TESTS = {
    8545: "S11", 7520: "S11",
    9692: "S12", 7793: "S12",
    7523: "S13", 8558: "S13",
    7517: "S14", 8560: "S14",
}


def camry(root, out):
    os.makedirs(out, exist_ok=True)
    for tno, scen in TESTS.items():
        d = os.path.join(root, "nhtsa-%d" % tno)
        ch = channels(d)
        td = api(d, "test-detail.json")
        title = ch["title"].upper()
        pole, sidet = "POLE" in title, ch["configuration"] == "IMPACTOR INTO VEHICLE"
        car = 2 if sidet else 1
        ref = dict(
            source="NHTSA Vehicle Crash Test Database, test %d (public domain, US government); derived by engine/tools/crash-reference/nhtsa_reference.py" % tno,
            test=tno, scenario=scen, title=ch["title"], test_type=td.get("testType"), configuration=ch["configuration"],
            closing_speed_kmh=ch["closing_speed_kmh"], lab=td.get("testPerformer"), date=(td.get("testDate") or "")[:10],
            vehicle=vehicle_scalars(d, car),
            processing="bias over [-20, 0) ms removed; SAE J211-1 CFC 60 (two-pole Butterworth, forward and backward); trapezoidal integration from contact; SAE axes (+X forward, +Y right); curves at 1 kHz from -10 to 250 ms",
        )
        if sidet:
            ref["barrier"] = vehicle_scalars(d, 1)
        if sidet or pole:
            side(d, ch, ref, pole)
        else:
            frontal(d, ch, ref)
        p = os.path.join(out, "%d.json" % tno)
        with open(p, "w") as f:
            json.dump(ref, f, separators=(",", ":"))
            f.write("\n")
        print(tno, scen, json.dumps(ref["scalars"]))


# ------------------------------------------------------------------ oblique (RMDB into a turned car)
OBLIQUE = {9124: "left", 8790: "left", 9121: "right"}
OBLIQUE_ANGLE = 15.0


def report_text(test_dir):
    """The test's main report as text (the largest PDF in report/, pdftotext -layout), or None."""
    exe = shutil.which("pdftotext")
    rep = os.path.join(test_dir, "report")
    if not exe or not os.path.isdir(rep):
        return None
    pdfs = sorted((f for f in os.listdir(rep) if f.lower().endswith(".pdf")), key=lambda f: -os.path.getsize(os.path.join(rep, f)))
    if not pdfs:
        return None
    try:
        return pdfs[0], subprocess.run([exe, "-layout", os.path.join(rep, pdfs[0]), "-"], capture_output=True, text=True, timeout=180).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def oblique_report(test_dir):
    """The oblique report's measured numbers, read from its text: the Crash Vehicle Summary's
    masses and static crush, Data Sheet 12's crush profile (C1..C6 with the bumper cover, B1..B9
    without; 'UR' unrecoverable kept as None) and Data Sheet 13's wheelbases and door openings,
    each with the PDF page it is on."""
    got = report_text(test_dir)
    if not got:
        return None
    name, text = got
    out = dict(file=name)
    pages = text.split("\f")

    def find(pat, conv=float):
        for i, p in enumerate(pages, 1):
            m = re.search(pat, p, re.M)
            if m:
                return conv(m.group(1)), i
        return None, None

    for key, pat in (("rmdb_speed_kmh", r"RMDB Velocity at Impact\s+kph\s+([\d.]+)"), ("rmdb_mass_kg", r"RMDB Test Weight\s+kg\s+([\d.]+)"),
                     ("rmdb_max_static_crush_mm", r"RMDB Maximum Static Crush\s+mm\s+([\d.]+)"), ("car_mass_kg", r"Vehicle Test Weight\s+kg\s+([\d.]+)"),
                     ("car_max_static_crush_mm", r"Vehicle Maximum Static Crush\s+mm\s+([\d.]+)"), ("vehicle_angle_deg", r"Actual Vehicle Angle\s+degrees\s+([\d.]+)"),
                     ("lateral_offset_mm", r"Lateral Offset from Target Point.*?mm\s+([-+\d.]+)"), ("vertical_offset_mm", r"Vertical Offset from Target Point.*?mm\s+([-+\d.]+)")):
        v, pg = find(pat)
        if v is not None:
            out[key] = v
            out[key + "_page"] = pg

    def num_or_none(x):
        try:
            return float(x)
        except ValueError:
            return None
    rows = {}
    for i, p in enumerate(pages, 1):
        for m in re.finditer(r"^\s*([CB][1-9])\s+(?:Crush zone|Bumper Beam) \d at (left|right) side\s+mm\s+(\S+)\s+(\S+)\s+(\S+)\s*$", p, re.M):
            if m.group(1) not in rows:
                rows[m.group(1)] = dict(side=m.group(2), pre_mm=num_or_none(m.group(3)), post_mm=num_or_none(m.group(4).rstrip("*")), diff_mm=num_or_none(m.group(5).rstrip("*")), page=i)
    if rows:
        out["crush_profile_bumper_cover"] = [dict(point=k, **rows[k]) for k in sorted(rows) if k[0] == "C"]
        out["crush_profile_bumper_beam"] = [dict(point=k, **rows[k]) for k in sorted(rows) if k[0] == "B"]
    for side in ("Left", "Right"):
        for i, p in enumerate(pages, 1):
            m = re.search(r"%s Side Wheelbase\s+mm\s+(\d+)\s+(\d+)\s+(-?\d+)" % side, p)
            if m:
                out["wheelbase_%s_mm" % side.lower()] = dict(pre=int(m.group(1)), post=int(m.group(2)), diff=int(m.group(3)), page=i)
                break
        for lvl in ("Upper", "Lower"):
            for i, p in enumerate(pages, 1):
                m = re.search(r"%s Side %s\s+mm\s+(\d+)\s+(\d+)\s+(-?\d+)" % (side, lvl), p)
                if m:
                    out["door_opening_%s_%s_mm" % (side.lower(), lvl.lower())] = dict(pre=int(m.group(1)), post=int(m.group(2)), diff=int(m.group(3)), page=i)
                    break
    return out


def oblique(test_dir, ch, ref, side):
    """An RMDB moving along +X_g at the closing speed into a standing car turned 15 deg: the car's
    CG accelerations (SAE body axes) integrated from rest, the RMDB's CG X integrated from the
    closing speed, the car's velocity along the RMDB's travel and the closing speed between them."""
    V = ch["closing_speed_kmh"] / 3.6
    cm = lambda c: (c.get("comment") or "").upper()
    car = {c["axis"][0]: c for c in ch["tsv_channels"] if usable(c) and c.get("vehicle_no") == 2 and "VEHICLE CG" in (c.get("sensor_attachment") or "") and c["axis"][0] in "XY" and not re.search(r"\bR[XYZ]\b|REDUNDANT", cm(c))}
    mdb = {c["axis"][0]: c for c in ch["tsv_channels"] if usable(c) and c.get("vehicle_no") == 1 and "VEHICLE CG" in (c.get("sensor_attachment") or "") and c["axis"][0] in "XY"}
    rate = [c for c in ch["tsv_channels"] if c.get("vehicle_no") == 2 and c["sensor_type"].startswith("ANGULAR") and c["axis"].startswith("Z")]
    t, ax = accel(test_dir, car["X"])
    _, ay = accel(test_dir, car["Y"])
    vx, sx = integrate_from_zero(t, ax, 0.0)
    vy, sy = integrate_from_zero(t, ay, 0.0)
    tb, bx = accel(test_dir, mdb["X"])
    _, by = accel(test_dir, mdb["Y"])
    vbx, sbx = integrate_from_zero(tb, bx, V)
    vby, _ = integrate_from_zero(tb, by, 0.0)
    n = min(len(t), len(tb))
    chans = [car[k]["file"].split("/")[-1] + ": " + car[k]["comment"] for k in "XY"] + [mdb[k]["file"].split("/")[-1] + ": " + mdb[k]["comment"] + (" (" + mdb[k]["data_status"] + ")" if mdb[k].get("data_status") == "COMPUTED" else "") for k in "XY"]
    # Yaw: measured by an angular rate sensor (SAE: + nose right) where the test has one.
    yaw_rate = yaw = None
    if rate:
        ty, w = read_ev(os.path.join(test_dir, rate[0]["file"]))
        w = cfc(debias(ty, w), ty[1] - ty[0])
        yaw, _ = integrate_from_zero(ty, w, 0.0)
        yaw_rate = [at(ty, w, x) for x in t[:n]]
        yaw = [at(ty, yaw, x) for x in t[:n]]
        chans.append(rate[0]["file"].split("/")[-1] + ": " + rate[0]["comment"])
    # The car in the RMDB's frame: the RMDB travels toward the car's rear and toward the struck
    # side's opposite (left oblique: toward +Y, right).
    a = math.radians(OBLIQUE_ANGLE)
    sgn = 1.0 if side == "left" else -1.0
    ux, uy = -math.cos(a), sgn * math.sin(a)  # the travel direction in the car's first frame
    if yaw is not None:
        # Body-axis accelerations turned into the car's first frame by the measured yaw, then integrated.
        c_ = [math.cos(math.radians(y)) for y in yaw]
        s_ = [math.sin(math.radians(y)) for y in yaw]
        gx = [ax[i] * c_[i] - ay[i] * s_[i] for i in range(n)]
        gy = [ax[i] * s_[i] + ay[i] * c_[i] for i in range(n)]
        vgx, _ = integrate_from_zero(t[:n], gx, 0.0)
        vgy, _ = integrate_from_zero(t[:n], gy, 0.0)
    else:
        vgx, vgy = vx[:n], vy[:n]
    along = [vgx[i] * ux + vgy[i] * uy for i in range(n)]
    across = [-vgx[i] * uy + vgy[i] * ux for i in range(n)]
    closing = [vbx[i] - along[i] for i in range(n)]
    sc = {}
    after = [i for i in range(n) if 0 <= t[i] <= REBOUND_UNTIL + 0.05]
    ia = max(after, key=lambda i: math.hypot(ax[i], ay[i]))
    sc.update(car_a_res_peak_g=math.hypot(ax[ia], ay[ia]) / G, car_a_res_peak_ms=1000 * t[ia],
              car_ax_peak_g=-max(-ax[i] for i in after) / G, car_ay_peak_g=max((ay[i] for i in after), key=abs) / G)
    for ms in (100, 150, 250):
        x = ms / 1000.0
        sc["car_dvx_%dms_mps" % ms] = at(t, vx, x)
        sc["car_dvy_%dms_mps" % ms] = at(t, vy, x)
        sc["car_dv_res_%dms_mps" % ms] = math.hypot(at(t, vx, x), at(t, vy, x))
        sc["car_dv_dir_%dms_deg" % ms] = math.degrees(math.atan2(abs(at(t, vy, x)), -at(t, vx, x)))
        sc["car_v_along_travel_%dms_mps" % ms] = at(t[:n], along, x)
        sc["car_v_across_travel_%dms_mps" % ms] = at(t[:n], across, x)
        sc["rmdb_vx_%dms_mps" % ms] = at(tb, vbx, x)
        sc["rmdb_vy_%dms_mps" % ms] = at(tb, vby, x)
    k = [i for i in range(n) if 0 <= t[i] <= REBOUND_UNTIL]
    im = min(k, key=lambda i: closing[i])
    sc.update(closing_min_mps=closing[im], closing_min_ms=1000 * t[im])
    j = next((i for i in range(n) if t[i] >= 0 and closing[i] <= 0), None)
    if j is not None:
        tc = t[j - 1] + (t[j] - t[j - 1]) * closing[j - 1] / (closing[j - 1] - closing[j])
        sc["common_velocity_ms"] = 1000 * tc
        sc["common_velocity_mps"] = at(t[:n], along, tc)
        # The approach along the travel until then: the integral of the closing speed.
        app = 0.0
        for i in range(1, n):
            if t[i] <= 0:
                continue
            if t[i] > tc:
                break
            app += 0.5 * (closing[i] + closing[i - 1]) * (t[i] - t[i - 1])
        sc["approach_at_common_velocity_m"] = app
    m_c, m_b = ref["vehicle"].get("test_mass_kg"), ref["barrier"].get("test_mass_kg")
    rp = ref.get("report") or {}
    m_c, m_b = rp.get("car_mass_kg", m_c), rp.get("rmdb_mass_kg", m_b)
    if m_c and m_b:
        x = 0.15
        sc["momentum_along_rmdb_lost_150ms"] = m_b * (V - at(tb, vbx, x))
        sc["momentum_along_car_gained_150ms"] = m_c * at(t[:n], along, x)
        sc["momentum_balance_150ms"] = sc["momentum_along_car_gained_150ms"] / sc["momentum_along_rmdb_lost_150ms"]
        sc["common_velocity_plastic_mps"] = m_b * V / (m_b + m_c)
    if yaw is not None:
        k = [i for i in range(n) if 0 <= t[i] <= 0.29]
        iw = max(k, key=lambda i: abs(yaw_rate[i]))
        sc.update(yaw_rate_peak_dps=yaw_rate[iw], yaw_rate_peak_ms=1000 * t[iw], yaw_150ms_deg=at(t[:n], yaw, 0.15), yaw_250ms_deg=at(t[:n], yaw, 0.25), yaw_290ms_deg=at(t[:n], yaw, 0.29))
    ref.update(location="struck car's CG accelerometer X and Y (SAE body axes, not redundant ones); the RMDB's CG X and Y", channels=chans)
    ref["frame"] = ("car_dv*: the car's CG velocity change in its own body axes, integrated without correction for yaw; "
                    "car_v_along/across_travel: in the car's first frame turned onto the RMDB's travel (%s oblique: the travel is %g deg off the car's -X toward %s)%s; "
                    "rmdb_v*: the RMDB's CG in its body axes from the closing speed; closing = rmdb_vx - car_v_along_travel."
                    % (side, OBLIQUE_ANGLE, "+Y (right)" if side == "left" else "-Y (left)",
                       "; this test's angular rate sensor turns the body accelerations into the first frame before integrating" if yaw is not None else
                       "; no angular rate sensor in this test, so the body-axis velocities are used: valid while the yaw is small (the overhead film shows it small to about 150 ms)"))
    ref["scalars"] = {k: round(x, 4) for k, x in sc.items()}
    ys = [[x / G for x in ax[:n]], [x / G for x in ay[:n]], vx[:n], vy[:n], [x / G for x in bx[:n]], vbx[:n], vby[:n], along, across, closing]
    names = ["car_ax_g", "car_ay_g", "car_dvx_mps", "car_dvy_mps", "rmdb_ax_g", "rmdb_vx_mps", "rmdb_vy_mps", "car_v_along_travel_mps", "car_v_across_travel_mps", "closing_mps"]
    if yaw is not None:
        ys += [yaw_rate, yaw]
        names += ["yaw_rate_dps", "yaw_deg"]
    ct, out = decimate(t[:n], ys)
    ref["curves"] = dict(t_ms=ct, **{k: v for k, v in zip(names, out)})


def obliques(root, out):
    os.makedirs(out, exist_ok=True)
    for tno, side in OBLIQUE.items():
        d = os.path.join(root, "nhtsa-%d" % tno)
        ch = channels(d)
        td = api(d, "test-detail.json")
        ref = dict(
            source="NHTSA Vehicle Crash Test Database, test %d (public domain, US government); derived by engine/tools/crash-reference/nhtsa_reference.py oblique" % tno,
            test=tno, scenario=None, side=side, title=ch["title"], test_type=td.get("testType"), configuration=ch["configuration"],
            closing_speed_kmh=ch["closing_speed_kmh"], impact_angle_deg=num(td.get("impactAngle")), lab=td.get("testPerformer"), date=(td.get("testDate") or "")[:10],
            vehicle=vehicle_scalars(d, 2), barrier=vehicle_scalars(d, 1),
            processing="bias over [-20, 0) ms removed; SAE J211-1 CFC 60 (two-pole Butterworth, forward and backward) on accelerations and the angular rate; trapezoidal integration from contact (t = 0, the record's zero); SAE axes (+X forward, +Y right, yaw + nose right); curves at 1 kHz from -10 to 250 ms",
        )
        rp = oblique_report(d)
        if rp:
            ref["report"] = rp
            api_max = ref["vehicle"].get("max_crush_mm")
            if api_max and rp.get("car_max_static_crush_mm") and abs(api_max - rp["car_max_static_crush_mm"]) > 1:
                ref["api_errata"] = "the API's maximum crush (%g mm) and C1 disagree with the report (%g mm; C1 unrecoverable): the API's C1 holds C1's pre-test distance" % (api_max, rp["car_max_static_crush_mm"])
        oblique(d, ch, ref, side)
        p = os.path.join(out, "%d.json" % tno)
        with open(p, "w") as f:
            json.dump(ref, f, separators=(",", ":"))
            f.write("\n")
        print(tno, side, json.dumps(ref["scalars"]))


def recheck(pulses):
    """Section 8's tests from their per-curve files: the same scalars, and whether a file
    has the bad-server signature (both curves identical, or cut at 131,072 bytes)."""
    V = 56.3 / 3.6
    tests = sorted({f[1:6] for f in os.listdir(pulses) if f.endswith((".093", ".094"))})
    for tn in tests:
        files = [os.path.join(pulses, "v%stsv.%s" % (tn, c)) for c in ("093", "094")]
        raw = [open(f, "rb").read() for f in files]
        bad = raw[0] == raw[1] or any(len(r) == 131072 for r in raw)
        each = []
        for f in files:
            t, v = read_ev(f)
            v = debias(t, v)
            each.append((t, [x * G for x in cfc(v, t[1] - t[0])]))
        t, a = mean_curves(each)
        v, s = integrate_from_zero(t, a, V)
        sc = pulse(t, a, v, s, V)
        sc.update({"as_rubble_samples_" + k: x for k, x in as_rubble_samples(t, v).items()})
        print(int(tn), "BAD (identical or truncated file)" if bad else "ok", "record to %.3f s" % t[-1], json.dumps({k: round(x, 3) for k, x in sc.items()}))


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "camry":
        camry(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "oblique":
        obliques(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "recheck":
        recheck(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
