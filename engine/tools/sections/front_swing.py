#!/usr/bin/env python3
"""Does the crash box fold or swing? Over the slow push, per variant: the box's span along the
car (z, front nodes to rear nodes, outboard top corner) against its two outboard chords' lengths.
A folding box shortens its chords as its span; a swinging one keeps its chords and loses span.

    python3 -I tools/sections/front_swing.py off cell ...
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, "..", ".."))


def main():
    marks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("| variant | " + " | ".join(f"{m:.1f} m: span / chords (m)" for m in marks) + " |")
    print("|---|" + "---|" * len(marks))
    for t in sys.argv[1:]:
        p = os.path.join(ENGINE, "out", "gen", f"sedan_{t}", "audit", "sedan_push.csv")
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p)))
        cells = []
        for m in marks:
            r = next((b for a, b in zip(rows, rows[1:]) if float(a["disp_m"]) <= m <= float(b["disp_m"])), None)
            cells.append("–" if r is None else f"{float(r['z_a1_a2_m']):.3f} / {float(r['crash_box_len_m']):.3f}, {float(r['crash_box_low_len_m']):.3f}")
        print(f"| {t} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
