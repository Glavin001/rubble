#!/usr/bin/env python3
"""A crash's pulse beside the real tests': speed, travel since first contact and wall force
over time, one line per run.

    pulse_plot.py --out FILE.png [--title T] --ref docs/crash-reference/nhtsa/8545.json ... \
        --run "Generated sedan=out/x/sedan.json" ...

`--ref`: the suite's committed NHTSA references (`nhtsa_reference.py`; curves `t_ms`,
`vx_mps`, `dx_m`, `force_n`). `--run`: `crash_ref ID --json` reports (their `series`: the
reference location's speed where the car has reference nodes, else its centre of mass; the
boxes' force at the report's 4 ms samples). Travel is the speed integrated from first
contact (t = 0), as the references' is. Needs matplotlib.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def travel(t_ms, v):
    """Trapezoidal integral of `v` (m/s) from t = 0 (m); before it, 0."""
    out, s = [], 0.0
    for i, t in enumerate(t_ms):
        if i and t > 0.0:
            t0 = max(t_ms[i - 1], 0.0)
            s += 0.5 * (v[i] + v[i - 1]) * (t - t0) * 1e-3
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--until", type=float, default=150.0, help="ms after first contact")
    a = ap.parse_args()

    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for path in a.ref:
        d = json.load(open(path))
        c = d["curves"]
        label = f"real: {d['vehicle']['make_model'].title()}, NHTSA test {d['test']} ({d['vehicle']['test_mass_kg']:.0f} kg)"
        ax[0].plot(c["t_ms"], [v * 3.6 for v in c["vx_mps"]], "k", lw=2.2, alpha=0.55 if path != a.ref[0] else 1.0, label=label)
        ax[1].plot(c["t_ms"], c["dx_m"], "k", lw=2.2, alpha=0.55 if path != a.ref[0] else 1.0)
        if "force_n" in c:
            ax[2].plot(c["t_ms"], [f / 1e3 for f in c["force_n"]], "k", lw=2.2, alpha=0.55 if path != a.ref[0] else 1.0)
    for spec in a.run:
        name, path = spec.split("=", 1)
        r = json.load(open(path))
        s = r["series"]
        b = s["fine"]["bodies"][0]
        at = "reference" if r["metrics"].get("car.ref.nodes", 0.0) > 0.0 else "com"
        t, v = s["fine"]["t_ms"], b[at]["vx"]
        line, = ax[0].plot(t, [x * 3.6 for x in v], lw=1.8, label=f"Rubble: {name} ({r['metrics']['car.mass']:.0f} kg)")
        ax[1].plot(t, travel(t, v), lw=1.8, color=line.get_color())
        ax[2].plot(s["t_ms"], [f / 1e3 for f in s["box_force_n"]], lw=1.8, color=line.get_color())
    ax[0].set_ylabel("speed toward the wall (km/h)")
    ax[1].set_ylabel("travel since first contact (m)")
    ax[2].set_ylabel("wall force (kN)")
    ax[2].set_xlabel("time after first contact (ms)")
    ax[0].axhline(0.0, color="0.7", lw=0.8)
    ax[1].set_ylim(-0.1, 0.8)
    for x in ax:
        x.set_xlim(-10.0, a.until)
        x.grid(alpha=0.25)
    ax[0].legend(fontsize=9)
    ax[0].set_title(a.title)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
