#!/usr/bin/env python3
"""Where the slow push's travel goes along the front, at 0.39 m (the depth S11 reached before the
rules), per variant (out/gen/sedan_<TAG>/audit/sedan_push.csv, tools/sections/front_ab.sh).

    python3 -I tools/sections/front_spans.py off r1 r2 ...
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))


def main():
    print("| variant | crash box a1–a2: z shortening (m) | chords' length upper / lower (m, from 0.180) | rail a2–a4 (z) | rail a4–a5 (z) | a5 to firewall (z) | firewall to cabin rear | crash box front's rise (m) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in sys.argv[1:]:
        p = os.path.join(ENGINE, "out", "gen", f"sedan_{t}", "audit", "sedan_push.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        if "z_a1_a2_m" not in rows[0]:
            continue
        r0 = rows[0]
        r = next(b for a, b in zip(rows, rows[1:]) if float(a["disp_m"]) <= 0.39 <= float(b["disp_m"]))

        def d(k):
            return float(r0[k]) - float(r[k])

        print(f"| {t} | {d('z_a1_a2_m'):.3f} | {float(r['crash_box_len_m']):.3f} / {float(r['crash_box_low_len_m']):.3f} | {d('z_a2_a4_m'):.3f} | {d('z_a4_a5_m'):.3f} | {d('z_a5_firewall_m'):.3f} | {d('firewall_to_held_m'):.3f} | {-float(r['a1_drop_m']):.3f} |")


if __name__ == "__main__":
    main()
