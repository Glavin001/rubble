#!/usr/bin/env python3
"""Markdown tables of the AVBD vehicle study's CPU measurements
(engine/docs/measurements/avbd-vehicles/<car>-<config>-<scene>.json, written by
`avbd_vehicles`): one row per config, explicit first.

    python3 engine/tools/avbd-vehicles-table.py [dir]
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "docs", "measurements", "avbd-vehicles")


def load(car, cfg, scene):
    p = os.path.join(D, f"{car}-{cfg}-{scene}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["result"]


def f(x, nd=1):
    if x is None:
        return "–"
    if isinstance(x, (int,)):
        return str(x)
    try:
        if x != x or abs(x) > 1e6:
            return "diverged"
        return f"{x:.{nd}f}"
    except TypeError:
        return str(x)


def key(cfg):
    if cfg == "explicit":
        return (0, 0, 0)
    hz, it = cfg.split("x")
    return (1, float(hz), float(it.split("c")[0]))


cars = sorted({n.split("-")[0] for n in os.listdir(D) if n.endswith(".json")})
for car in cars:
    cfgs = sorted({n[len(car) + 1:].rsplit("-", 1)[0] for n in os.listdir(D)
                   if n.startswith(car + "-") and n.endswith(".json")}, key=key)
    print(f"\n#### {car}\n")
    print("| config | rest sag mm | rest shape mm | rest drift mm | landing deform peak / end mm | landing COM dip mm | "
          "wall dyn / residual crush mm | springback | peak g | restitution | wall broken beams / hardware | wall KE end kJ | "
          "pole dyn / residual mm | pole broken | CPU s per sim s (wall) |")
    print("|" + "---|" * 15)
    for c in cfgs:
        r, b, w, p = (load(car, c, s) for s in ("rest", "bump", "wall", "pole"))
        row = [c]
        row += [f(r and r["sag_mm"]), f(r and r["shape_change_rms_mm"]), f(r and r["drift_xz_mm"], 0)] if r else ["–"] * 3
        row += [f"{f(b['deform_peak_mm'])} / {f(b['deform_end_mm'])}", f(b["com_dip_mm"])] if b else ["–"] * 2
        if w:
            dmg = w.get("damage", {})
            ke = w.get("energy", {}).get("ke_end_kj")
            row += [f"{f(w['dynamic_crush_mm'], 0)} / {f(w['residual_crush_max_mm'], 0)}", f(w["springback_ratio"], 2),
                    f(w["peak_g"]), f(w["restitution"], 2),
                    f"{dmg.get('broken_beams')} / {dmg.get('hardware_broken')}", f(ke, 1)]
        else:
            row += ["–"] * 6
        if p:
            row += [f"{f(p.get('dynamic_crush_mm'), 0)} / {f(p.get('residual_crush_mm'), 0)}",
                    str(p.get("damage", {}).get("broken_beams"))]
        else:
            row += ["–"] * 2
        row += [f(w and w.get("wall_s_per_sim_s"), 2)]
        print("| " + " | ".join(row) + " |")
