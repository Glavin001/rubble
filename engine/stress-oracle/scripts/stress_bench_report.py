#!/usr/bin/env python3
"""Three-way destruction benchmark report: the stress-ref oracle (CPU), PhysX GPU with
stress-gpu, and rubble's integrated pipeline (rubble AVBD + stress-gpu on one device, no
frame rewinds), from `stress_bench` JSON lines (crates/stress-rubble/examples/stress_bench.rs;
the PhysX lines come from the same harness built against PhysX's engine).

    stress_bench_report.py OUT.md RUN.jsonl [RUN.jsonl ...]

Each scene's outcome (the impactor's exit speed, fragments, broken bonds) is compared with
the oracle's; frame times are summarised per solver over every repeat (mean, median, p95,
max, first frame, frames over 8.3 / 16.7 ms).
"""
import json
import statistics as st
import sys

out_path, inputs = sys.argv[1], sys.argv[2:]
runs = {}
for path in inputs:
    for line in open(path):
        r = json.loads(line)
        runs.setdefault((r["regime"], r["solver"]), []).append(r)
scenes, solvers = [], []
for (sc, so) in runs:
    if sc not in scenes:
        scenes.append(sc)
    if so not in solvers:
        solvers.append(so)
order = ["oracle", "physx-gpu", "rubble-gpu", "rubble-gpu-redo"]
solvers.sort(key=lambda s: order.index(s) if s in order else len(order))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


lines = ["# Destruction benchmark: oracle vs PhysX GPU vs rubble", ""]
lines += ["## Outcomes", "", "Impactor exit speed (m/s) / fragments / broken bonds (first repeat).", ""]
lines += ["| scene | " + " | ".join(solvers) + " |", "|---" * (len(solvers) + 1) + "|"]
for sc in scenes:
    cells = []
    for so in solvers:
        rs = runs.get((sc, so))
        if not rs:
            cells.append("—")
            continue
        r = rs[0]
        extra = ""
        if r.get("redone_frames"):
            extra += f" (rewound {r['redone_frames']})"
        cells.append(f"{r['ram']:.2f} / {r['fragments']:.0f} / {r['broken_bonds']:.0f}{extra}")
    lines.append(f"| {sc} | " + " | ".join(cells) + " |")
lines += ["", "## Agreement with the oracle", "",
          "Each solver's outcome over its repeats (exit speed m/s, fragments, broken bonds: min-max), and the exit speed's difference from the oracle's.", ""]
lines += ["| scene | " + " | ".join(solvers) + " |", "|---" * (len(solvers) + 1) + "|"]
for sc in scenes:
    ref = runs.get((sc, "oracle"))
    ref_ram = ref[0]["ram"] if ref else None
    cells = []
    for so in solvers:
        rs = runs.get((sc, so))
        if not rs:
            cells.append("—")
            continue
        def rng(k, f):
            lo, hi = f % min(r[k] for r in rs), f % max(r[k] for r in rs)
            return lo if lo == hi else f"{lo}–{hi}"
        cell = f"{rng('ram', '%.2f')} / {rng('fragments', '%.0f')} / {rng('broken_bonds', '%.0f')}"
        if ref_ram is not None and so != "oracle" and ref_ram > 0:
            d = st.median([r["ram"] for r in rs]) - ref_ram
            cell += f" (Δv {d:+.2f})"
        cells.append(cell)
    lines.append(f"| {sc} | " + " | ".join(cells) + " |")
lines += ["", "## Frame time (ms, every frame of every repeat)", ""]
for metric, fn in [
    ("mean", st.mean),
    ("median", st.median),
    ("p95", lambda xs: pct(xs, 0.95)),
    ("max (peak)", max),
    ("first frame", None),
]:
    lines += [f"### {metric}", "", "| scene | " + " | ".join(solvers) + " |", "|---" * (len(solvers) + 1) + "|"]
    for sc in scenes:
        cells = []
        for so in solvers:
            rs = runs.get((sc, so))
            if not rs:
                cells.append("—")
                continue
            if fn is None:
                v = st.median([r["frame_ms"][0] for r in rs])
            else:
                v = fn([x for r in rs for x in r["frame_ms"]])
            cells.append(f"{v:.1f}")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    lines.append("")
lines += ["### frames over budget (all repeats)", "", "| scene | " + " | ".join(f"{s} >8.3 / >16.7" for s in solvers) + " |", "|---" * (len(solvers) + 1) + "|"]
for sc in scenes:
    cells = []
    for so in solvers:
        rs = runs.get((sc, so))
        if not rs:
            cells.append("—")
            continue
        xs = [x for r in rs for x in r["frame_ms"]]
        cells.append(f"{sum(x > 8.33 for x in xs)} / {sum(x > 16.67 for x in xs)} of {len(xs)}")
    lines.append(f"| {sc} | " + " | ".join(cells) + " |")
open(out_path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
