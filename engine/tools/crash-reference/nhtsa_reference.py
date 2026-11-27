#!/usr/bin/env python3
"""Derive the crash reference suite's NHTSA reference numbers from the downloaded test data.

    python3 -I engine/tools/crash-reference/nhtsa_reference.py camry DATA_ROOT OUT_DIR
    python3 -I engine/tools/crash-reference/nhtsa_reference.py recheck PULSES_DIR

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
    elif len(sys.argv) >= 3 and sys.argv[1] == "recheck":
        recheck(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
