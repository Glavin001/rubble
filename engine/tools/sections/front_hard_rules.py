#!/usr/bin/env python3
"""Summary of tools/sections/front_hard_rules.sh: per variant, scenario and contact setting the
hard rules over the 8 runs (energy created, momentum, into the wall, finite, crossings) and the
crush measures, worst or range.

    python3 -I tools/sections/front_hard_rules.py TAG [TAG ...]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))


def rng(xs, n=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    lo, hi = min(xs), max(xs)
    return f"{lo:.{n}f}" if abs(hi - lo) < 10 ** -n else f"{lo:.{n}f}–{hi:.{n}f}"


def main():
    print("| variant | scenario | contact | runs | energy created, max (J) | its band | into the wall, max (m) | finite | crossings, max | asserted failures | restitution | dyn. crush (m) | residual (m) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in sys.argv[1:]:
        d = os.path.join(ENGINE, "out", "gen", f"hard_{tag}")
        for s in ["S11", "S12", "S01", "S02", "S08"]:
            for c in ["discrete", "edge", "continuous"]:
                p = os.path.join(d, f"{s}-{c}.jsonl")
                if not os.path.exists(p):
                    continue
                rs = [json.loads(l) for l in open(p) if l.strip()]
                if not rs:
                    print(f"| {tag} | {s} | {c} | 0 | | | | | | | | | |")
                    continue
                m = [r["metrics"] for r in rs]
                eg = max(x.get("all.energy_gain_j", float("nan")) for x in m)
                band = ""
                fails = {}
                for r in rs:
                    for res in r.get("results", []):
                        if res.get("metric") == "all.energy_gain_j" and not band:
                            band = f"≤ {res.get('hi')}"
                        if res.get("asserted") and not res.get("pass", True):
                            k = res.get("name", res.get("metric", "?"))
                            fails[k] = fails.get(k, 0) + 1
                depth = max(x.get("all.box_depth_m", 0.0) for x in m)
                fin = min(x.get("all.finite", 1.0) for x in m)
                cross = [v for x in m for k, v in x.items() if k.endswith("crossings_max")]
                car = "car" if any(k.startswith("car.") for k in m[0]) else next((k.split(".")[0] for k in m[0] if k.endswith(".restitution") or ".ref." in k), "car")
                rest = [x.get(f"{car}.ref.restitution") for x in m]
                dyn = [x.get(f"{car}.ref.dyn_crush") for x in m]
                res_ = [x.get(f"{car}.crush_residual") for x in m]
                fl = ", ".join(f"{k} {n}/{len(rs)}" for k, n in sorted(fails.items())) or "none"
                print(f"| {tag} | {s} | {c} | {len(rs)} | {eg:.1f} | {band} | {depth:.3f} | {fin:.0f} | {max(cross) if cross else '–'} | {fl} | {rng(rest)} | {rng(dyn)} | {rng(res_)} |")


if __name__ == "__main__":
    main()
