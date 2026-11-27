#!/usr/bin/env python3
"""crash-reference.md §4.3's "beside the real tests" rows for S11 and S12 from front_ab.sh
variants (run 0, edge contact), the Camry tests' from docs/crash-reference/nhtsa/.

    python3 -I tools/sections/front_beside.py BEFORE_TAG AFTER_TAG
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))


def rep(tag, sid):
    for line in open(os.path.join(ENGINE, "out", "gen", f"sedan_{tag}", "s11s12.jsonl")):
        r = json.loads(line)
        if r["id"] == sid:
            return r["metrics"]


ROWS = [
    ("mass (kg)", "car.mass", 0),
    ("speed (km/h)", "car.speed_before", 1),
    ("peak decel, 4 ms estimator (g)", "car.ref.peak_decel_g", 1),
    ("peak decel, CFC 60 (g)", "car.ref.cfc60_peak_decel_g", 1),
    ("time of peak, CFC 60 (ms)", "car.ref.cfc60_peak_ms", 1),
    ("t_m, v = 0 (ms)", "car.ref.stop_ms", 1),
    ("dynamic crush (m)", "car.ref.dyn_crush", 3),
    ("mean decel to t_m (g)", "car.ref.mean_decel_g", 1),
    ("rebound, t_m-200 ms (m/s)", "car.ref.rebound", 2),
    ("restitution", "car.ref.restitution", 3),
    ("residual crush (m)", "car.crush_residual", 3),
    ("peak wall force (kN)", "all.box_force_peak_n", 0),
    ("time of peak force (ms)", "all.box_force_peak_ms", 1),
]


def main():
    before, after = sys.argv[1], sys.argv[2]
    for sid in ("S11", "S12"):
        b, a = rep(before, sid), rep(after, sid)
        print(f"{sid}: | quantity | before ({before}) | now ({after}) |")
        for name, k, n in ROWS:
            sc = 1e-3 if k == "all.box_force_peak_n" else (3.6 if k == "car.speed_before" else 1.0)
            print(f"| {name} | {b[k] * sc:.{n}f} | {a[k] * sc:.{n}f} |")


if __name__ == "__main__":
    main()
