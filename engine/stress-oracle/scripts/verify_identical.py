#!/usr/bin/env python3
"""Bit-identity gate and CPU-time benchmark for stress-ref builds.

    scripts/verify_identical.py --base BASE_BIN --cand CAND_BIN --out DIR [--jobs N]
                                [--only REGEX] [--bench] [--repeat K]

Gate (default): runs every cell of the matrix

    scene (scenes/*.json, scenes/showcases/*.json)
      x methods  {default (the proven methods), established}
      x threads  {1, all}
      + solve modes {adaptive, quasi_static, implicit} on a subset

with both builds and compares the observations byte for byte (wall_seconds
removed). The base build's outputs are cached in DIR/base (the base does not change
between iterations); the candidate's go to DIR/cand. Exits 1 if any cell differs.

Bench (--bench): runs the benchmark cells sequentially with RAYON_NUM_THREADS=1,
K times each, and prints the minimum CPU time (user + sys) of base and candidate.
The machine is shared: CPU time, not wall time, measures the work.
"""
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The default methods (the proven set since cd112933c) and the established ones.
METHODS = {"default": None, "established": "established"}
MODE_SUBSET = ["b2_cantilever", "b4_support_loss_sudden", "b8_frame_sudden", "b9_panel_high", "s_floor_static", "s_arch_keystone_removed", "b7_masonry_v15"]
MODES = ["adaptive", "quasi_static", "implicit"]
BENCH = ["b1_bond_tension", "b2_cantilever", "b3_bar_wave", "b4_support_loss_sudden", "b5_wall_impact_v10", "b8_frame_sudden",
         "b9_panel_high", "s_arch_keystone_removed", "s_car_brick", "s_house_car", "s_floor_drop"]


def scenes():
    out = {}
    for d in ["scenes", "scenes/showcases"]:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith(".json"):
                out[f[:-5]] = os.path.join(ROOT, d, f)
    return out


def cells(all_scenes, only):
    out = []
    for name, path in all_scenes.items():
        for m in METHODS:
            for threads in ["1", "all"]:
                out.append((f"{name}.{m}.t{threads}", path, m, threads, None))
        if name in MODE_SUBSET:
            for m in METHODS:
                for mode in MODES:
                    out.append((f"{name}.{m}.{mode}.t1", path, m, "1", mode))
    if only:
        out = [c for c in out if re.search(only, c[0])]
    return out


def run(binary, cell, out_path):
    tag, scene, method, threads, mode = cell
    env = dict(os.environ)
    env.pop("STRESS_METHODS", None)
    if METHODS[method]:
        env["STRESS_METHODS"] = METHODS[method]
    if threads == "1":
        env["RAYON_NUM_THREADS"] = "1"
    else:
        env.pop("RAYON_NUM_THREADS", None)
    args = [binary, "run", scene, "--out", out_path]
    if mode:
        args += ["--set", f'sim.solve_mode="{mode}"']
    start = time.time()
    p = subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, status, usage = os.wait4(p.pid, 0)
    err = p.stderr.read().decode(errors="replace")
    p.stderr.close()
    cpu = usage.ru_utime + usage.ru_stime
    return cpu, time.time() - start, err if not os.path.exists(out_path) else ""


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "wall_seconds"}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o


def gate(a):
    cs = cells(scenes(), a.only)
    for sub in ["base", "cand"]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    def one(cell):
        tag = cell[0]
        base_out = os.path.join(a.out, "base", tag + ".json")
        cand_out = os.path.join(a.out, "cand", tag + ".json")
        if not os.path.exists(base_out):
            run(a.base, cell, base_out)
        if os.path.exists(cand_out):
            os.remove(cand_out)
        cpu, wall, err = run(a.cand, cell, cand_out)
        if not (os.path.exists(base_out) and os.path.exists(cand_out)):
            return tag, "MISSING", cpu, err[-300:]
        same = strip(json.load(open(base_out))) == strip(json.load(open(cand_out)))
        return tag, "IDENTICAL" if same else "DIFFERENT", cpu, ""

    bad = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for tag, verdict, cpu, err in ex.map(one, cs):
            if verdict != "IDENTICAL":
                bad += 1
            print(f"{tag:58} {verdict:9} {cpu:8.2f} s cpu {err}", flush=True)
    print(f"{len(cs) - bad}/{len(cs)} identical")
    return 1 if bad else 0


def bench(a):
    all_scenes = scenes()
    rows = []
    for name in BENCH:
        for m in METHODS:
            cell = (f"{name}.{m}.t1", all_scenes[name], m, "1", None)
            best = {}
            for which, binary in [("base", a.base), ("cand", a.cand)]:
                times = []
                for _ in range(a.repeat):
                    out = os.path.join(a.out, "bench", which + "." + cell[0] + ".json")
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    cpu, _, _ = run(binary, cell, out)
                    times.append(cpu)
                best[which] = min(times)
            rows.append((cell[0], best["base"], best["cand"]))
            print(f"{cell[0]:40} base {best['base']:8.2f} s  cand {best['cand']:8.2f} s  x{best['base'] / max(best['cand'], 1e-9):5.2f}", flush=True)
    tb, tc = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"{'total':40} base {tb:8.2f} s  cand {tc:8.2f} s  x{tb / tc:5.2f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    sys.exit(bench(a) if a.bench else gate(a))
