#!/usr/bin/env python3
"""Plots for engine/docs/beam-yield-law.md (needs matplotlib: a throwaway venv under out/).

    PY=.../venv/bin/python
    $PY tools/beam-law/plot.py [TAG ...]          # from engine/; default tags: off lp lpx

Reads out/beam-law/one_beam.csv and off_centre_*.csv (beam_yield_law.rs's ignored telemetry) and
out/gen/sedan_<TAG>/s11s12.jsonl (tools/sections/front_ab.sh), writes out/beam-law/*.png.
"""
import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "out/beam-law"
# The Camry tests' numbers (crash-reference.md §4.3; front-load-path.md §4.2): S11 peak force and its
# time, t_m, dynamic crush; S12 t_m and crush.
CAMRY_S11 = [("8545", 921, 29.5, 70.9, 0.663), ("7520", 829, 30.1, 75.9, 0.685)]
CAMRY_S12 = [("9692", 69.3, 0.475), ("7793", 74.7, 0.489)]


def one_beam():
    rows = list(csv.DictReader(open(os.path.join(OUT, "one_beam.csv"))))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    for ax, member in zip(axs, ["crash_box", "rail_bay"]):
        rs = [r for r in rows if r["member"] == member]
        for law, style in [("jbeam_hardening", "C3-"), ("jbeam_flat", "C1-"), ("crush_law", "C0-")]:
            x = [float(r["shortening"]) for r in rs if r["law"] == law]
            y = [float(r["force"]) / 1e3 for r in rs if r["law"] == law]
            ax.plot(x, y, style, lw=1.4, label={"jbeam_hardening": "JBeam rule (hardens, MIN_REST 0.1 m)", "jbeam_flat": "JBeam rule, deformLimitStress flat", "crush_law": "crush law (peak, drop, plateau, densify)"}[law])
        peak = [float(r["force"]) / 1e3 for r in rs if r["law"] == "target_peak"][0]
        stroke, plat = [(float(r["shortening"]), float(r["force"]) / 1e3) for r in rs if r["law"] == "target_plateau"][0]
        ax.axhline(peak, color="k", ls=":", lw=1, label=f"section peak (effective-width squash) {peak:.1f} kN")
        ax.plot([0, stroke], [plat, plat], "k--", lw=1.2, label=f"section plateau (Abramowicz–Jones) {plat:.1f} kN")
        ax.axvline(stroke, color="0.5", ls="--", lw=1, label=f"73 % stroke {stroke:.3f} m")
        ax.set_ylim(0, 2.6 * peak)
        ax.set_title(f"one chord of the sedan's {member.replace('_', ' ')} (a quarter of the section), quasi-static")
        ax.set_xlabel("shortening (m)")
        ax.set_ylabel("force (kN)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "one_beam_force_shortening.png"))


def off_centre():
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), dpi=120)
    for row, member in enumerate(["crash_box", "rail_bay"]):
        for col, r in enumerate(["0", "0.3", "1"]):
            ax = axs[row][col]
            for law, c in [("crush_cell_x", "C0"), ("crush_chords_x", "C3"), ("jbeam_flat", "C1")]:
                p = os.path.join(OUT, f"off_centre_{member}_{law}_{r}.csv")
                if not os.path.exists(p):
                    continue
                rs = list(csv.DictReader(open(p)))
                t = [float(x["t"]) * 1e3 for x in rs]
                ax.plot(t, [float(x["top"]) * 1e3 for x in rs], c + "-", lw=1.3, label=f"{law}: near pair")
                ax.plot(t, [float(x["bottom"]) * 1e3 for x in rs], c + "--", lw=1.3, label=f"{law}: far pair")
            ax.set_title(f"{member.replace('_', ' ')}: load {r} × M/N off centre")
            ax.set_xlabel("t (ms)")
            ax.set_ylabel("chord pair shortening (mm)")
            ax.grid(alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=7)
    fig.suptitle("A ram through a pin off the cell's centre: the crush cell folds (both pairs alike); independent chords turn (the near pair only)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "off_centre_cell.png"))


def runs(tag):
    p = f"out/gen/sedan_{tag}/s11s12.jsonl"
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)}


def crush_force(r):
    ser = r["series"]
    ref = ser["bodies"][0]["reference"]
    t = ser["t_ms"]
    d, f, tt = [], [], []
    for i in range(len(t)):
        if t[i] < 0:
            continue
        d.append(ref["dx"][i])
        f.append(ser["box_force_n"][i] / 1e3)
        tt.append(t[i])
    return tt, d, f


def crashes(tags):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), dpi=120)
    for k, tag in enumerate(tags):
        rr = runs(tag)
        for sid, ax_t, ax_d in [("S11", axs[0], axs[1])]:
            if sid not in rr:
                continue
            t, d, f = crush_force(rr[sid])
            ax_t.plot(t, f, f"C{k}-", lw=1.2, label=tag)
            m = [i for i in range(1, len(d)) if d[i] < d[i - 1]]
            end = m[0] if m else len(d)
            ax_d.plot(d[:end], f[:end], f"C{k}-", lw=1.2, label=tag)
        if "S12" in rr:
            t, d, f = crush_force(rr["S12"])
            axs[2].plot(t, f, f"C{k}-", lw=1.2, label=tag)
    for name, pk, tp, tm, cr in CAMRY_S11:
        axs[0].plot([tp], [pk], "k^", ms=7)
        axs[0].annotate(f"Camry {name}: peak {pk} kN at {tp} ms, t_m {tm} ms", (tp, pk), fontsize=7, xytext=(5, -10), textcoords="offset points")
        axs[1].axvline(cr, color="k", ls=":", lw=0.8)
    for name, tm, cr in CAMRY_S12:
        axs[2].axvline(tm, color="k", ls=":", lw=0.8)
    axs[0].set(title="S11 (56 km/h wall): wall force against time", xlabel="t (ms)", ylabel="wall force (kN)", xlim=(0, 120))
    axs[1].set(title="S11: wall force against crush (Camry dynamic crush dotted)", xlabel="crush (m)", ylabel="wall force (kN)")
    axs[2].set(title="S12 (40 km/h wall): wall force against time (Camry t_m dotted)", xlabel="t (ms)", ylabel="wall force (kN)", xlim=(0, 140))
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "s11_s12_overlay.png"))


def main():
    tags = sys.argv[1:] or ["off", "lp", "lpx"]
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "one_beam.csv")):
        one_beam()
    if glob.glob(os.path.join(OUT, "off_centre_*.csv")):
        off_centre()
    crashes(tags)
    print("wrote", sorted(glob.glob(os.path.join(OUT, "*.png"))))


if __name__ == "__main__":
    main()
