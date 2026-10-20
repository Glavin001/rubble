#!/usr/bin/env python3
"""P0 batch runner: many `learned_p0` runs in parallel, one result file each.

    p0_runs.py <out_dir> --jobs N --spec SPEC.json
    p0_runs.py <out_dir> --jobs N --chaos [--members 16] [--scenes a,b] [--modes kin,kin_weibull]

SPEC.json: a list of {"name": ..., "args": [...], "env": {...}} (arguments after the binary;
"env" optional). Each run's
stdout (one JSON line) goes to <out_dir>/<name>.json; a run already there is skipped
(resume). One progress line per finished run, with running totals and an ETA; meant to run
under job.py (status.json, heartbeats).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("LEARNED_P0_BIN", os.path.join(ROOT, "target-p0", "release", "examples", "learned_p0"))
CHAOS_SCENES = ["free_wall", "wall_row", "free_wall_v25", "free_wall_v60", "free_wall_edge", "thick_wall", "b5_v20"]


def chaos_spec(members, scenes, modes):
    spec = []
    for s in scenes:
        spec.append({"name": f"{s}_ref", "args": ["oracle", "--scene", s]})
        for mode in modes:
            for k in range(members):
                spec.append({"name": f"{s}_{mode}_{k:02d}", "args": ["oracle", "--scene", s, "--member", str(k), "--perturb", mode]})
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--spec")
    p.add_argument("--chaos", action="store_true")
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--scenes", default=",".join(CHAOS_SCENES))
    p.add_argument("--modes", default="kin,kin_weibull")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = chaos_spec(a.members, a.scenes.split(","), a.modes.split(",")) if a.chaos else json.load(open(a.spec))
    todo = [r for r in spec if not os.path.exists(os.path.join(a.out, r["name"] + ".json"))]
    print(f"[p0] {len(spec)} runs, {len(spec) - len(todo)} already done, {len(todo)} to run on {a.jobs} workers", flush=True)
    t0 = time.time()
    done = failed = 0

    def one(r):
        t = time.time()
        env = dict(os.environ, **r.get("env", {}))
        res = subprocess.run([BIN] + r["args"], capture_output=True, text=True, env=env)
        line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
        if res.returncode != 0 or line is None:
            with open(os.path.join(a.out, r["name"] + ".err"), "w") as f:
                f.write(res.stderr[-20000:])
            return r, None, time.time() - t
        tmp = os.path.join(a.out, r["name"] + ".json.tmp")
        with open(tmp, "w") as f:
            f.write(line + "\n")
        os.replace(tmp, os.path.join(a.out, r["name"] + ".json"))
        return r, json.loads(line), time.time() - t

    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for fu in as_completed(futs):
            r, d, dt = fu.result()
            if d is None:
                failed += 1
                summary = "FAILED (see .err)"
            else:
                done += 1
                e = d["end"]
                summary = f"ram vy {e['ram_vy']:.3f} fragments {e['fragments']} broken {e['broken_bonds']} largest {e['largest_fraction']:.3f}"
            n = done + failed
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            print(f"[p0] {n}/{len(todo)} | {r['name']} {summary} | {dt:.1f} s | ETA {eta/60:.1f} min | failures {failed}", flush=True)
    print(f"[p0] finished: {done} ok, {failed} failed in {(time.time() - t0)/60:.1f} min", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
