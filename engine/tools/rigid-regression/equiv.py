#!/usr/bin/env python3
"""Equivalence recordings for the rigid regression check
(engine/docs/measurements/rigid-regression-2026-11-07.md): each scene recorded by each
build with engine-world's `tests/equivalence.rs` harness (ignored test `record_scene`),
the first build recorded twice (its own run-to-run noise), then every recording compared
with the first build's (`compare`). Every run is bounded by a hard timeout.

usage: equiv.py OUT_DIR SCENES BUILD=BINDIR [BUILD=BINDIR ...]
SCENES: comma-separated (house, arch, grid, ring, gables, city, vehicle).
BINDIR holds the build's `equivalence` test binary. EQUIV_SLEEP (on) is passed through.
"""
import os, subprocess, sys

out = os.path.abspath(sys.argv[1])
scenes = sys.argv[2].split(",")
builds = [a.split("=", 1) for a in sys.argv[3:]]
first = builds[0]
runs = [(first[0] + "_again", first[1])] + builds


def run(cmd, env, timeout):
    e = dict(os.environ)
    e.update(env)
    try:
        p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"


for scene in scenes:
    for name, d in runs:
        rc, so, se = run([os.path.join(d, "equivalence"), "record_scene", "--ignored",
                          "--nocapture", "--exact", "--test-threads=1"],
                         {"EQUIV_OUT": os.path.join(out, name), "EQUIV_SCENE": scene}, 900)
        print(f"record {scene} {name}: rc={rc} {se[-200:] if rc else ''}", flush=True)

for name, _ in runs:
    if name == first[0]:
        continue
    rc, so, se = run([os.path.join(first[1], "equivalence"), "compare", "--ignored",
                      "--nocapture", "--exact", "--test-threads=1"],
                     {"EQUIV_COMPARE": os.path.join(out, first[0]) + "," + os.path.join(out, name)}, 300)
    print(f"=== compare {first[0]} vs {name}: rc={rc}", flush=True)
    print(so, flush=True)
    if rc:
        print(se[-2000:], flush=True)
