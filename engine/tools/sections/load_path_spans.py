#!/usr/bin/env python3
"""Does the front fold in sequence? Over the slow push (out/gen/sedan_<TAG>/audit/sedan_push.csv,
front_ab.sh), per variant and push depth: each bay's shortening along the car (crash box a1-a2,
rail a2-a4, rail a4-a5, a5 to the firewall, firewall to the held cabin), the crash box's and the
rails' chord lengths, the bumper beam's distance to the box front (does it stay on) and the box
front's drop (does it turn). A bay that folds shortens its span and its chords together; one that
turns loses span and keeps its chords.

    python3 -I tools/sections/load_path_spans.py off lp_c lp ...   [--md]
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def at(rows, d):
    for a, b in zip(rows, rows[1:]):
        if float(a["disp_m"]) <= d <= float(b["disp_m"]):
            return b
    return None


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    print("| variant | push (m) | box a1–a2: Δz / chord (m) | rail a2–a4: Δz / chord | rail a4–a5: Δz / chord | a5–firewall Δz | firewall–cabin Δz | beam to box front (m) | box front drop (m) | force (kN) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        p = os.path.join(ENGINE, "out", "gen", f"sedan_{t}", "audit", "sedan_push.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        r0 = rows[0]
        for m in MARKS:
            r = at(rows, m)
            if r is None:
                continue
            d = lambda k: float(r0[k]) - float(r[k])
            print(f"| {t} | {m:.1f} | {d('z_a1_a2_m'):.3f} / {float(r['crash_box_len_m']):.3f} | {d('z_a2_a4_m'):.3f} / {float(r['rail_a2_a4_m']):.3f} | "
                  f"{d('z_a4_a5_m'):.3f} / {float(r['rail_a4_a5_m']):.3f} | {d('z_a5_firewall_m'):.3f} | {d('firewall_to_held_m'):.3f} | "
                  f"{float(r['beam_to_crash_box_m']):.3f} | {float(r['a1_drop_m']):.3f} | {float(r['wall_kN']):.0f} |")


if __name__ == "__main__":
    main()
