#!/usr/bin/env python3
"""The front rules' comparison table, variants side by side (after tools/sections/front_ab.sh).

    python3 -I tools/sections/front_ab.py off r1 r12 ...          # from engine/, a markdown table
    PY=.../venv/bin/python; $PY tools/sections/front_ab.py --plot out/sections/fixes off r1 ...

Reads out/gen/sedan_<TAG>/{s11s12.jsonl, audit/*} and prints one column per variant: the
rigged axial yield across the crash boxes' and the rails' bays (the lattice cut, both sides)
beside the stated sections, the push's force at 0.1-0.4 m, S11's time to stop, dynamic crush,
rebound, restitution, residual crush, the wall force at 0.1-0.4 m of the rear-seat
crossmember's travel and its peak, the elastic energy at the stop and released by zone, the
engine's least gap to the firewall, and S12's. With --plot DIR (matplotlib) the push's and
S11's force against displacement for every variant, with the two Camry tests' S11 curves.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))
REF = os.path.join(ENGINE, "docs", "crash-reference", "nhtsa")

# The stated sections (sedan-front-audit.md §1, the calculator): two crash boxes, two rails.
STATED = {"crash_box": (2 * 45.4, 2 * 70.0), "front_rail": (2 * 61.6, 2 * 95.0)}

ZONES = {
    "front lower": {"crash_box", "front_rail", "bumper_beam", "subframe", "rad_support", "radiator"},
    "front upper": {"hood", "fender", "upper_rail", "tower", "bumper_cover"},
    "cabin": {"cabin", "firewall", "cowl", "door", "dash"},
}


def zone_of(cls):
    parts = cls.split("+")
    zs = set()
    for p in parts:
        z = next((k for k, v in ZONES.items() if p in v), "other")
        zs.add(z)
    if zs == {"front lower", "front upper"}:
        return "front lower+upper"
    if len(zs) == 1:
        return zs.pop()
    if "cabin" in zs:
        return "cabin"
    return "other"


def load(tag):
    d = os.path.join(ENGINE, "out", "gen", f"sedan_{tag}")
    out = {"tag": tag, "dir": d}
    with open(os.path.join(d, "s11s12.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r
    a = os.path.join(d, "audit")
    try:
        with open(os.path.join(a, "sedan_push.csv")) as f:
            out["push"] = list(csv.DictReader(f))
    except FileNotFoundError:
        out["push"] = []
    try:
        with open(os.path.join(a, "sedan_front_cuts.csv")) as f:
            out["cuts"] = list(csv.DictReader(f))
    except FileNotFoundError:
        out["cuts"] = []
    try:
        out["energy"] = open(os.path.join(a, "sedan_s11_energy.txt")).read()
        with open(os.path.join(a, "sedan_s11_elastic_by_class.csv")) as f:
            out["by_class"] = list(csv.DictReader(f))
    except FileNotFoundError:
        out["energy"], out["by_class"] = "", []
    return out


def at(xs, ys, x0):
    """y at x0 on the first crossing of x0 (linear), or None."""
    for i in range(1, len(xs)):
        if (xs[i - 1] - x0) * (xs[i] - x0) <= 0 and xs[i] != xs[i - 1]:
            t = (x0 - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return None


def s11_curve(r):
    """(displacement of the rear-seat crossmember, box force kN) from contact to t_m."""
    ser = r["series"]
    ref = ser["bodies"][0]["reference"]
    t = ser["t_ms"]
    k = [i for i in range(len(t)) if t[i] >= 0.0]
    d, f = [], []
    for i in k:
        if d and ref["dx"][i] < d[-1]:
            break
        d.append(ref["dx"][i])
        f.append(ser["box_force_n"][i] / 1e3)
    return d, f


def cut_at(v, z0):
    rows = v["cuts"]
    if not rows:
        return None, {}
    zs = sorted({float(x["z_m"]) for x in rows}, key=lambda z: abs(z - z0))
    z = zs[0]
    sel = [x for x in rows if abs(float(x["z_m"]) - z) < 1e-6]
    return float(sel[0]["total_kN"]), {x["class"]: float(x["kN"]) for x in sel}


def fmt(x, n=1):
    return "–" if x is None else f"{x:.{n}f}"


def rows_for(v):
    m11 = v["S11"]["metrics"]
    m12 = v.get("S12", {}).get("metrics", {})
    r = {}
    def split(cls):
        y = sum(x for k, x in cls.items() if "(elastic)" not in k)
        return y, sum(cls.values()) - y

    _, cls = cut_at(v, 0.70)
    y, e = split(cls)
    box = cls.get("crash_box", 0.0)
    braces = sum(x for k, x in cls.items() if k.startswith("crash_box+") and "(elastic)" not in k and "bumper" not in k)
    panels = sum(x for k, x in cls.items() if any(p in k for p in ("hood", "fender")) and "(elastic)" not in k)
    r["cut z 0.70 (crash-box bay, both sides; stated 2 × 45 static, 2 × 70 crash rate): yielding / never yielding (kN)"] = f"{y:.0f} / {e:.0f}"
    r["  of which crash boxes / box–other braces / hood + fenders (kN)"] = f"{box:.0f} / {braces:.0f} / {panels:.0f}"
    _, cls47 = cut_at(v, 0.47)
    y, e = split(cls47)
    sub = sum(x for k, x in cls47.items() if "subframe" in k and "crash_box" in k)
    rail = cls47.get("crash_box+front_rail", 0.0) + cls47.get("front_rail", 0.0)
    r["cut z 0.47 (rails' bay; stated 2 × 62 static, 2 × 95 crash rate): yielding / never yielding (kN)"] = f"{y:.0f} / {e:.0f}"
    r["  of which rail chords and braces / subframe–crash box (kN)"] = f"{rail:.0f} / {sub:.0f}"
    push = v["push"]
    pd = [float(x["disp_m"]) for x in push]
    pf = [float(x["wall_kN"]) for x in push]
    r["push force at 0.1 / 0.2 / 0.3 / 0.4 m (kN)"] = " / ".join(fmt(at(pd, pf, x), 0) for x in (0.1, 0.2, 0.3, 0.4))
    if push:
        cb = [float(x["crash_box_len_m"]) for x in push]
        ra = [float(x["rail_a2_a4_m"]) for x in push]
        rb = [float(x["rail_a4_a5_m"]) for x in push]
        r["push at 0.39 m: crash-box chord / rail chords (m)"] = f"{fmt(at(pd, cb, 0.39), 3)} / {fmt(at(pd, ra, 0.39), 3)}, {fmt(at(pd, rb, 0.39), 3)}"
        el = [float(x["elastic_kJ"]) for x in push]
        pl = [float(x["plastic_kJ"]) for x in push]
        r["push at 0.39 m: elastic / plastic (kJ)"] = f"{fmt(at(pd, el, 0.39))} / {fmt(at(pd, pl, 0.39))}"
    d, f = s11_curve(v["S11"])
    r["S11 force at 0.1 / 0.2 / 0.3 / 0.4 m (kN)"] = " / ".join(fmt(at(d, f, x), 0) for x in (0.1, 0.2, 0.3, 0.4))
    r["S11 t_m (ms)"] = fmt(m11["car.ref.stop_ms"])
    r["S11 dynamic crush (m)"] = fmt(m11["car.ref.dyn_crush"], 3)
    r["S11 rebound (m/s) / restitution"] = f"{m11['car.ref.rebound']:.2f} / {m11['car.ref.restitution']:.3f}"
    r["S11 residual crush (m)"] = fmt(m11["car.crush_residual"], 3)
    r["S11 peak wall force (kN) at (ms)"] = f"{m11['all.box_force_peak_n'] / 1e3:.0f} at {m11['all.box_force_peak_ms']:.1f}"
    r["S11 4 ms / CFC 60 peak (g)"] = f"{m11['car.ref.peak_decel_g']:.1f} / {m11['car.ref.cfc60_peak_decel_g']:.1f}"
    r["S11 cabin: door opening change peak / residual (mm)"] = f"{1e3 * m11['car.gauge.door_left.change_peak']:.0f} / {1e3 * m11['car.gauge.door_left.change_residual']:.0f}"
    r["S11 plastic work: cabin share"] = f"{m11['car.zone.cabin.work_frac']:.3f}"
    e = v["energy"]
    mt = re.search(r"elastic total ([0-9.]+)", e)
    me = re.search(r"0.3 s after contact ([0-9.]+) kJ", e)
    if mt and me:
        r["S11 elastic at the stop / 0.3 s later (kJ)"] = f"{float(mt.group(1)):.1f} / {float(me.group(1)):.1f}"
    rel = {}
    for x in v["by_class"]:
        z = zone_of(x["class"])
        rel[z] = rel.get(z, 0.0) + float(x["released_J"]) / 1e3
    if rel:
        r["S11 released by zone (kJ)"] = ", ".join(f"{k} {rel[k]:.1f}" for k in sorted(rel, key=lambda k: -rel[k]))
    mg = re.search(r"least ([-0-9.]+) m at ([-0-9.]+) ms, within 3 cm at (\S+) ms", e)
    if mg:
        r["S11 engine rear to firewall: least (m) at (ms); within 3 cm"] = f"{mg.group(1)} at {mg.group(2)}; {mg.group(3)}"
    if m12:
        r["S12 t_m (ms) / crush (m) / restitution / residual (m)"] = f"{m12['car.ref.stop_ms']:.1f} / {m12['car.ref.dyn_crush']:.3f} / {m12['car.ref.restitution']:.3f} / {m12['car.crush_residual']:.3f}"
        r["S12 peak wall force (kN) at (ms)"] = f"{m12['all.box_force_peak_n'] / 1e3:.0f} at {m12['all.box_force_peak_ms']:.1f}"
    r["hard: energy gain S11 / S12 (J); into the wall (m)"] = f"{m11['all.energy_gain_j']:.1f} / {m12.get('all.energy_gain_j', float('nan')):.1f}; {m11['all.box_depth_m']:.3f}"
    return r


def table(vs):
    rs = [rows_for(v) for v in vs]
    keys = list(rs[0].keys())
    for r in rs[1:]:
        keys += [k for k in r if k not in keys]
    out = ["| | " + " | ".join(v["tag"] for v in vs) + " |", "|---|" + "---|" * len(vs)]
    for k in keys:
        out.append(f"| {k} | " + " | ".join(r.get(k, "–") for r in rs) + " |")
    return "\n".join(out)


def plot(vs, dest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(dest, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=130)
    for t, c in (("8545", "#555"), ("7520", "#999")):
        try:
            ref = json.load(open(os.path.join(REF, f"{t}.json")))
        except FileNotFoundError:
            continue
        cv = ref["curves"]
        # (to t_m: while the displacement still grows)
        k = [i for i in range(len(cv["t_ms"])) if cv["t_ms"][i] >= 0]
        dd, ff = [], []
        for i in k:
            if dd and cv["dx_m"][i] < dd[-1]:
                break
            dd.append(cv["dx_m"][i])
            ff.append(cv["force_n"][i] / 1e3)
        axs[1].plot(dd, ff, color=c, lw=1.4, ls="--", label=f"Camry test {t} (56 km/h, to t_m)")
    for v in vs:
        push = v["push"]
        axs[0].plot([float(x["disp_m"]) for x in push], [float(x["wall_kN"]) for x in push], lw=1.6, label=v["tag"])
        d, f = s11_curve(v["S11"])
        axs[1].plot(d, f, lw=1.6, label=v["tag"])
    axs[0].set_title("slow push (1 m/s): wall force against the cabin's travel")
    axs[1].set_title("S11 56.3 km/h: wall force against the rear-seat crossmember's travel (to t_m)")
    for a in axs:
        a.set_xlabel("displacement (m)")
        a.set_ylabel("force (kN)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    axs[0].set_xlim(0, 0.8)
    axs[1].set_xlim(0, 0.8)
    fig.tight_layout()
    p = os.path.join(dest, "front_ab_" + "_".join(v["tag"] for v in vs) + ".png")
    fig.savefig(p)
    print(p)


def main():
    a = sys.argv[1:]
    dest = None
    if a and a[0] == "--plot":
        dest, a = a[1], a[2:]
    vs = [load(t) for t in a]
    print(table(vs))
    if dest:
        plot(vs, dest)


if __name__ == "__main__":
    main()
