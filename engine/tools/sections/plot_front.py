#!/usr/bin/env python3
"""Plots for engine/docs/sedan-front-audit.md (local; matplotlib from the oracle's venv).

    engine/out/sections/venv/bin/python engine/tools/sections/plot_front.py

Reads, from engine/out/sections/ (written by `cargo test --release -p engine-softbody-lab
--test sedan_front_audit -- --ignored` and `crash_ref -- S11 --json > s11.jsonl`):
sedan_push.csv, s11.jsonl, sedan_prediction.json, sedan_front_cuts.csv; and the committed
NHTSA reference curves engine/docs/crash-reference/nhtsa/{8545,7520}.json. Writes
front_force_displacement.png, push_internals.png and front_cuts.png beside them.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ENGINE, "out", "sections")
REF = os.path.join(ENGINE, "docs", "crash-reference", "nhtsa")


def rows(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def prediction(p, scale_key, mult=1.0, upper=False, dmax=0.8):
    """The stated sections folding in series: the impact beam meets the wall at d0; the two
    crash boxes fold over 0.73 of their length, then the two rails. With `upper`, the upper
    rails and the radiator support's uprights fold in parallel from d0 as well."""
    d0 = p["z_nose"] - p["z_impact_beam_front"]
    f = p["stroke_fraction"]
    cb, rail = p["crash_box"], p["front_rail"]
    d1 = d0 + f * cb["length_m"]
    extra = 2 * (p["upper_rail"][scale_key] + p["radiator_support"][scale_key]) if upper else 0.0
    xs = [0.0, d0, d0, d1, d1, dmax]
    ys = [0.0, 0.0, 2 * cb[scale_key] + extra, 2 * cb[scale_key] + extra, 2 * rail[scale_key] + extra, 2 * rail[scale_key] + extra]
    return xs, [mult * y / 1e3 for y in ys]


def main():
    p = json.load(open(os.path.join(OUT, "sedan_prediction.json")))
    push = rows("sedan_push.csv")
    s11 = json.loads(open(os.path.join(OUT, "s11.jsonl")).readline())
    ser = s11["series"]
    ref = ser["bodies"][0]["reference"]
    tests = [(json.load(open(os.path.join(REF, f"{t}.json"))), t) for t in ("8545", "7520")]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
    for (r, t), c in zip(tests, ("#2563eb", "#0891b2")):
        cv = r["curves"]
        k = [i for i, x in enumerate(cv["t_ms"]) if x >= 0 and x <= r["scalars"]["stop_ms"] + 5]
        ax.plot([cv["dx_m"][i] for i in k], [cv["force_n"][i] / 1e3 for i in k], color=c, lw=1.6, label=f"NHTSA {t} Camry {r['vehicle']['test_mass_kg']:.0f} kg, 56 km/h (to t_m)")
    k = [i for i, x in enumerate(ser["t_ms"]) if x >= 0 and x <= s11["metrics"]["car.ref.stop_ms"] + 4]
    ax.plot([ref["dx"][i] for i in k], [ser["box_force_n"][i] / 1e3 for i in k], color="#dc2626", lw=2.2, label="Rubble sedan 1,448 kg, S11 56.3 km/h (CPU, to t_m)")
    ax.plot([float(x["disp_m"]) for x in push], [float(x["wall_kN"]) for x in push], color="#f97316", lw=2.0, label="Rubble sedan pushed at 1 m/s (quasi-static)")
    x, y = prediction(p, "pm_static_n")
    ax.plot(x, y, color="#16a34a", lw=1.6, ls="--", label="stated sections, Abramowicz–Jones, static σ₀: 2 crash boxes then 2 rails")
    x, y = prediction(p, "pm_deck_n")
    ax.plot(x, y, color="#15803d", lw=1.6, ls="-.", label="the same at 15.6 m/s (Cowper–Symonds, Camry deck C 8000, p 8)")
    x, y = prediction(p, "pm_deck_n", upper=True)
    ax.plot(x, y, color="#65a30d", lw=1.2, ls=":", label="… with upper rails and radiator uprights in parallel")
    x, y = prediction(p, "pm_deck_n", mult=p["crash_feel_front"])
    ax.plot(x, y, color="#7c3aed", lw=1.2, ls="--", label=f"… × {p['crash_feel_front']:.1f} (the default beamng_feel)")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1100)
    ax.set_xlabel("displacement into the wall (m): tests and S11 the rear-seat crossmember, push the cabin")
    ax.set_ylabel("wall force (kN)")
    ax.set_title("Generated sedan's front against the Camry tests and its own stated sections")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "front_force_displacement.png"))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    d = [float(x["disp_m"]) for x in push]
    for key, lab in [("crash_box_len_m", "crash box chord (0.18 m)"), ("rail_a2_a4_m", "rail chord a2–a4 (0.38 m)"), ("rail_a4_a5_m", "rail chord a4–a5 (0.45 m)"), ("engine_rear_to_firewall_m", "engine rear to firewall"), ("engine_front_to_radiator_m", "engine front to radiator")]:
        axs[0].plot(d, [float(x[key]) for x in push], label=lab)
    axs[0].set_xlabel("push displacement (m)")
    axs[0].set_ylabel("length / gap (m)")
    axs[0].set_title("What shortens in the quasi-static push")
    axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[1].plot(d, [float(x["elastic_kJ"]) for x in push], label="elastic (stored in beams)")
    axs[1].plot(d, [float(x["plastic_kJ"]) for x in push], label="plastic (yield work)")
    w, acc = [0.0], 0.0
    for a, b in zip(push, push[1:]):
        acc += 0.5 * (float(a["wall_kN"]) + float(b["wall_kN"])) * (float(b["disp_m"]) - float(a["disp_m"]))
        w.append(acc)
    axs[1].plot(d, w, label="∫F dx (wall work)", ls="--", color="k")
    axs[1].axhline(177.1, color="#dc2626", lw=0.8, ls=":", label="S11 kinetic energy 177 kJ")
    axs[1].set_xlabel("push displacement (m)")
    axs[1].set_ylabel("kJ")
    axs[1].set_title("Energy in the quasi-static push")
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "push_internals.png"))

    cuts = rows("sedan_front_cuts.csv")
    zs = sorted({float(r["z_m"]) for r in cuts}, reverse=True)
    yl = {z: 0.0 for z in zs}
    el = {z: 0.0 for z in zs}
    for r in cuts:
        v = float(r["kN"])
        if v == float("inf") or v > 1e6:
            continue
        if r["class"].startswith(("engine", "wheel")):
            continue
        (el if "(elastic)" in r["class"] else yl)[float(r["z_m"])] += v
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    ax.plot(zs, [yl[z] for z in zs], label="beams that yield: Σ threshold × cos")
    ax.plot(zs, [el[z] for z in zs], label="beams that never yield (elastic until they break): Σ strength × cos")
    for name, z in [("nose", p["z_nose"]), ("impact beam", p["z_impact_beam_front"]), ("crash box front", p["z_crash_box_front"]), ("crash box rear", p["z_crash_box_front"] - p["crash_box"]["length_m"]), ("engine front", p["z_engine_front"])]:
        ax.axvline(z, color="#999", lw=0.8, ls=":")
        ax.text(z, ax.get_ylim()[1] * 0.0 + 50, name, rotation=90, fontsize=8, va="bottom")
    ax.invert_xaxis()
    ax.set_xlabel("z of the cutting plane (m, car frame, + forward)")
    ax.set_ylabel("axial capacity across the plane (kN)")
    ax.set_title("Rigged lattice cut across the car (engine block and wheels left out)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "front_cuts.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
