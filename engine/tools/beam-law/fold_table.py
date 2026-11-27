#!/usr/bin/env python3
"""The fold tables (beam-yield-law.md §9) from `tools/beam-law/fold.sh`'s CSVs: per variant and
scenario, each left bay's plastic work by mode at the end, its largest mode force against its
capacity, its span and chords' shortening at the end, and the stations' largest drop and the
largest upward load on a4. Markdown on stdout.

    python3 -I tools/beam-law/fold_table.py [variant ...]   (default: every folder in out/beam-law/fold)
"""
import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOLD = os.path.normpath(os.path.join(HERE, "..", "..", "out", "beam-law", "fold"))
MODES = ["axial", "strong", "weak", "warping"]


def bays(path):
    out = collections.OrderedDict()
    for r in csv.DictReader(open(path)):
        if not r["bay"].endswith("l"):
            continue
        b = out.setdefault(r["bay"], {"ratio": [0.0] * 4, "last": None})
        for i, m in enumerate(MODES):
            g, c = float(r[f"G_{m}_N"]), float(r[f"cap_{m}_N"])
            # (the axial mode against its compressive curve when compressed, its tension otherwise)
            if c > 0:
                b["ratio"][i] = max(b["ratio"][i], abs(g) / c)
        b["last"] = r
    return out


def stations(path):
    out = collections.OrderedDict()
    for r in csv.DictReader(open(path)):
        if not r["station"].endswith("l"):
            continue
        s = out.setdefault(r["station"], {"drop": 0.0, "up": 0.0})
        s["drop"] = min(s["drop"], float(r["dy_m"]))
        s["up"] = max(s["up"], float(r["F_up_N"]))
    return out


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(d for d in os.listdir(FOLD) if os.path.isdir(os.path.join(FOLD, d)))
    print("| variant | scenario | bay | plastic work at the end: axial / strong / weak / warping (kJ) | largest force / capacity: axial / strong / weak | span / chords' shortening at the end (m) |")
    print("|---|---|---|---|---|---|")
    st_rows = []
    for t in tags:
        for s in ["S11", "S12"]:
            p = os.path.join(FOLD, t, f"{s}-edge-0-bays.csv")
            if not os.path.exists(p):
                continue
            for name, b in bays(p).items():
                r = b["last"]
                w = [float(r[f"W_{m}_J"]) / 1e3 for m in MODES]
                span = float(r["span0_m"]) - float(r["span_m"])
                ch = float(r["chord0_mean_m"]) - float(r["chord_mean_m"])
                q = b["ratio"]
                print(f"| {t} | {s} | {name[:-1]} | {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} / {w[3]:.2f} | {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} | {span:.3f} / {ch:.3f} |")
            sp = os.path.join(FOLD, t, f"{s}-edge-0-stations.csv")
            st = stations(sp)
            summ = open(os.path.join(FOLD, t, f"{s}-edge-0-summary.txt")).read()
            stop = re.search(r"stops at Some\(([\d.]+)\)", summ)
            st_rows.append((t, s, stop.group(1) if stop else "–", st))
    print()
    print("| variant | scenario | car stops (ms) | largest drop of a1 / a2 / a4 / a5 (m) | largest upward load on a4 (kN) |")
    print("|---|---|---|---|---|")
    for t, s, stop, st in st_rows:
        d = " / ".join(f"{st[k]['drop']:.3f}" for k in ["a1l", "a2l", "a4l", "a5l"] if k in st)
        up = st.get("a4l", {}).get("up", 0.0) / 1e3
        print(f"| {t} | {s} | {float(stop):.0f} | {d} | {up:.0f} |" if stop != "–" else f"| {t} | {s} | – | {d} | {up:.0f} |")


if __name__ == "__main__":
    main()
