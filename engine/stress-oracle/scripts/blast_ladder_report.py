#!/usr/bin/env python3
"""Tables for the Blast-equivalence ladder (engine/docs/stress-fracture-blast-ladder.md).

    blast_ladder_report.py ladder DIR       ladder_<rung>.jsonl (stress_bench --set accuracy
                                            --config <rung>) against oracle_accuracy.jsonl
    blast_ladder_report.py packs DIR        packs_<config>.jsonl against blast_packs.jsonl
                                            (blast_ladder_bench bench, PhysX repository)

Frame times are over every frame of every repeat: mean, median, p95, max (ms).
Accuracy against the oracle, per scene: impactor exit speed (m/s), fragments, broken bonds,
and b5's failure mode (hole: punched through; push_over: most of the wall detached).
"""
import json
import os
import statistics as st
import sys

RUNGS = ["blast", "adaptive", "islands", "softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping", "realtime"]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def stats(xs):
    return f"{st.mean(xs):.1f} / {st.median(xs):.1f} / {pct(xs, 0.95):.1f} / {max(xs):.1f}"


FAILED = {}


def load(path):
    """Runs by regime; a failed run (`stress_bench` caught a panic) is kept in FAILED."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            if "failed" in r:
                FAILED[(path, r["regime"])] = r["failed"]
                continue
            out.setdefault(r["regime"], []).append(r)
    return out


def cell(runs, path, sc, fmt):
    if runs.get(sc):
        return fmt(runs[sc])
    return "failed" if (path, sc) in FAILED else "—"


def outcome(r):
    mode = (r.get("metrics") or {}).get("failure_mode")
    s = f"{r['ram']:.2f} / {r['fragments']:.0f} / {r['broken_bonds']:.0f}"
    return s + (f" {mode}" if mode else "")


def ladder(d):
    oracle = load(f"{d}/oracle_accuracy.jsonl")
    runs = {c: load(f"{d}/ladder_{c}.jsonl") for c in RUNGS}
    scenes = list(oracle) or list(next(iter(runs.values())))
    have = [c for c in RUNGS if runs[c]]
    print("### Cost per rung (ms per frame: mean / median / p95 / max)\n")
    print("| scene | " + " | ".join(have) + " |")
    print("|---" * (len(have) + 1) + "|")
    for sc in scenes:
        cells = [cell(runs[c], f"{d}/ladder_{c}.jsonl", sc, lambda rs: stats([x for r in rs for x in r["frame_ms"]])) for c in have]
        print(f"| {sc} | " + " | ".join(cells) + " |")
    # Impact scenes' total wall time (sum of frames), the budget a game pays.
    # Median over every frame of every scene: robust to the load spikes of a shared machine.
    print("| **median frame, all scenes** | " + " | ".join(f"**{st.median([x for sc in scenes for r in runs[c].get(sc, []) for x in r['frame_ms']]):.1f}**" for c in have) + " |")
    print("| **sum of per-scene medians** | " + " | ".join(f"**{sum(st.median([x for r in runs[c][sc] for x in r['frame_ms']]) for sc in scenes if runs[c].get(sc)):.0f}**" for c in have) + " |")
    if FAILED:
        print("\nFailed runs: " + "; ".join(f"{os.path.basename(p)[7:-6]} / {sc}: {why}" for (p, sc), why in FAILED.items()))
    print("\n### Outcome per rung (exit speed m/s / fragments / broken bonds, b5 failure mode)\n")
    print("| scene | oracle | " + " | ".join(have) + " |")
    print("|---" * (len(have) + 2) + "|")
    for sc in scenes:
        o = oracle.get(sc, [None])[0]
        cells = [cell(runs[c], f"{d}/ladder_{c}.jsonl", sc, lambda rs: outcome(rs[0])) for c in have]
        print(f"| {sc} | {outcome(o) if o else '—'} | " + " | ".join(cells) + " |")
    print("\n### Accuracy against the oracle per rung\n")
    print("Exit speed error is |ours - oracle| / impact speed over the struck scenes; bond and fragment errors are |log2(ours / oracle)| (1 = a factor of two), over scenes where the oracle breaks bonds; failure mode over b5's five speeds.\n")
    print("| rung | mean exit-speed error | mean broken-bond error (factor 2^x) | mean fragment error (2^x) | b5 failure mode matches | scenes within 25 % on broken bonds |")
    print("|---|---|---|---|---|---|")
    import math

    for c in have:
        sp, bb, fr, modes, within = [], [], [], 0, 0
        n_modes = n_bb = 0
        for sc in scenes:
            if not (oracle.get(sc) and runs[c].get(sc)):
                continue
            o, r = oracle[sc][0], runs[c][sc][0]
            v0 = {"wall_ram": 40.0, "house_ram": 12.0, "house_car": None}.get(sc)
            if sc.startswith("b5_v"):
                v0 = float(sc[4:])
            if v0:
                sp.append(abs(r["ram"] - o["ram"]) / v0)
            if o["broken_bonds"] > 0:
                n_bb += 1
                bb.append(abs(math.log2(max(r["broken_bonds"], 1) / o["broken_bonds"])))
                fr.append(abs(math.log2(max(r["fragments"], 1) / max(o["fragments"], 1))))
                within += abs(r["broken_bonds"] - o["broken_bonds"]) <= 0.25 * o["broken_bonds"]
            om, rm = (o.get("metrics") or {}).get("failure_mode"), (r.get("metrics") or {}).get("failure_mode")
            if sc.startswith("b5_v") and om:
                n_modes += 1
                modes += om == rm
        print(f"| {c} | {100 * st.mean(sp):.1f} % | {st.mean(bb):.2f} | {st.mean(fr):.2f} | {modes} / {n_modes} | {within} / {n_bb} |")


def packs(d):
    blast = load(f"{d}/blast_packs.jsonl")
    ours = {c: load(f"{d}/packs_{c}.jsonl") for c in ["blast", "realtime"]}
    names = []
    for k in list(ours["blast"]) + list(blast):
        base = k.rsplit("_", 1)[0] + ("_idle" if k.endswith("idle") else "_ram")
        if base not in names:
            names.append(base)
    print("### Frame time (ms: mean / median / p95 / max; 15 frames x 3 repeats)\n")
    print("| scene | Blast CPU (resim 1), whole frame | Blast CPU, stress step only | ours, blast config | ours, realtime |")
    print("|---|---|---|---|---|")
    for n in names:
        b = [r for r in blast.get(n, []) if r.get("resim") == 1]
        cells = [stats([x for r in b for x in r["frame_ms"]]) if b else "—", stats([x for r in b for x in r["stress_ms"]]) if b else "—"]
        for c in ["blast", "realtime"]:
            rs = ours[c].get(n, [])
            cells.append(stats([x for r in rs for x in r["frame_ms"]]) if rs else "—")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\n### Outcome (exit speed m/s / fragments / broken bonds)\n")
    print("| scene | Blast CPU | ours, blast config | ours, realtime |")
    print("|---|---|---|---|")
    for n in names:
        b = [r for r in blast.get(n, []) if r.get("resim") == 1]
        cells = [f"{b[0]['ram']:.2f} / {b[0]['fragments']} / {b[0]['broken_bonds']}" if b else "—"]
        for c in ["blast", "realtime"]:
            rs = ours[c].get(n, [])
            cells.append(outcome(rs[0]) if rs else "—")
        print(f"| {n} | " + " | ".join(cells) + " |")


{"ladder": ladder, "packs": packs}[sys.argv[1]](sys.argv[2])
