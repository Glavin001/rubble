#!/usr/bin/env python3
"""Rigid regression driver (engine/docs/measurements/rigid-regression-2026-11-07.md).

Interleaved rigid regression runs: for each round, each scene, each build in turn
(the build order rotates per round). Every run is bounded by a hard timeout.

usage: driver.py OUT_CSV ROUNDS BUILD=BINDIR [BUILD=BINDIR ...]
BINDIR holds each build's release `rigid_regression_bench` example and `city_scale`
test binary (engine-world), copied out of its target dir.
env: RR_ROUND=k numbers the first round k (one bounded round per invocation);
RR_ONLY=k1,k2 runs only those scene keys (exact; a city key such as city64_sleep runs all its scenarios); RR_NO_CITY=1 skips the city runs.
"""
import json, os, subprocess, sys, time

out_csv = sys.argv[1]
# The city runs' JSON lines go next to the CSV.
S = os.path.dirname(os.path.abspath(out_csv))
rounds = int(sys.argv[2])
builds = [a.split("=", 1) for a in sys.argv[3:]]

BENCH = [
    ("sparse10k", ["sparse", "10000"], {}),
    ("sparse30k", ["sparse", "30000"], {}),
    ("sparse100k", ["sparse", "100000"], {}),
    ("awake30k", ["awake", "30000"], {}),
    ("awake100k", ["awake", "100000"], {}),
    ("stacks400x10_nosleep", ["stacks", "400", "10"], {"RR_SLEEP": "0"}),
    ("map2000", ["map", "2000"], {}),
    ("map4000", ["map", "4000"], {}),
    ("map1_4000", ["map1", "4000"], {}),
    ("demo_columns100k_nosleep", ["demo", "Box Columns (100k)"], {"RR_SLEEP": "0"}),
    ("demo_pile32k", ["demo", "Box Pile (32k)"], {}),
    ("demo_pyramid30_nosleep", ["demo", "Pyramid 30"], {"RR_SLEEP": "0"}),
    ("demo_grid100k", ["demo", "100k Grid"], {}),
    ("demo_ring28k", ["demo", "Brick Ring (28k)"], {}),
]
CITY = [
    ("city16", {"CITY_K": "16"}),
    ("city64", {"CITY_K": "64"}),
    ("city64_sleep", {"CITY_K": "64", "CITY_SLEEP": "1"}),
    ("city256_sleep", {"CITY_K": "256", "CITY_SLEEP": "1"}),
]

only = [x for x in os.environ.get("RR_ONLY", "").split(",") if x]
skip_city = os.environ.get("RR_NO_CITY") == "1"
log = open(out_csv, "a")


def run(cmd, env, timeout):
    e = dict(os.environ)
    e.update(env)
    t = time.time()
    try:
        p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.time() - t
    except subprocess.TimeoutExpired as x:
        return -9, x.stdout or "", "TIMEOUT", time.time() - t


first = int(os.environ.get("RR_ROUND", "0"))
for r in range(first, first + rounds):
    order = builds[r % len(builds):] + builds[: r % len(builds)]
    for key, args, env in BENCH:
        if only and not any(o == key for o in only):
            continue
        for name, d in order:
            rc, so, se, dt = run([os.path.join(d, "rigid_regression_bench")] + args,
                                 dict(env, RR_LABEL=name), 300)
            line = so.strip().splitlines()[-1] if so.strip() else f"{name},ERROR rc={rc} {se[-300:]!r}"
            log.write(f"{r},{key},{line}\n")
            log.flush()
            print(r, key, name, f"{dt:.1f}s", line, flush=True)
    if skip_city:
        continue
    for key, env in CITY:
        if only and not any(o == key for o in only):
            continue
        for name, d in order:
            jp = os.path.join(S, f"city_{name}_{key}_{r}.jsonl")
            if os.path.exists(jp):
                os.remove(jp)
            rc, so, se, dt = run([os.path.join(d, "city_scale"), "city_scale", "--ignored",
                                  "--nocapture", "--exact", "--test-threads=1"],
                                 dict(env, CITY_JSON=jp), 600)
            if rc != 0:
                log.write(f"{r},{key},{name},ERROR rc={rc} {se[-300:]!r}\n")
            elif os.path.exists(jp):
                for l in open(jp):
                    j = json.loads(l)
                    log.write(f"{r},{key}_{j['scenario']},{name},city,{j['k']},{j['chunks']},"
                              f"{j['wall_median_ms']},{j['wall_p95_ms']},{j['gpu_median_ms']},"
                              f"{j['host_median_ms']},{j['awake']},{j['broken']}\n")
            log.flush()
            print(r, key, name, f"{dt:.1f}s rc={rc}", flush=True)
