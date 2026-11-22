#!/usr/bin/env python3
"""Which of the World's kernels give another result without wgpu's bounds checks
(vehicle-perf-first-principles.md §8.49).

    python3 tools/rigid-gate/bounds-search.py [SKIP,...]        (from engine/)

Runs the rigid gate's hash mode (GATE_SCENES, default the 28k ring; sleep off) with the
bounds checks out of a set of kernels (RUBBLE_RIGID_BOUNDS_OFF_NAMES) and halves the set
while the hashes differ from the default build's: every dependent kernel, not only the
first. SKIP: kernels already known, left checked. One GPU job at a time, each bounded.
"""
import os, subprocess, sys

BIN = os.environ.get("GATE_BIN", "target/fast/examples/rigid_gate")
SCENES = os.environ.get("GATE_SCENES", "Brick Ring (28k)")

def gate(names, list_kernels=False):
    env = dict(os.environ, GATE_MODE="hash", GATE_SLEEP="off", GATE_SCENES=SCENES,
               RUBBLE_RIGID_BOUNDS_OFF_NAMES=",".join(names) or "none")
    if list_kernels:
        env["RUBBLE_RIGID_KERNEL_NAMES"] = "1"
    out = subprocess.run(["python3", "scripts/with-gpu-lock.py", "--limit", "600", "--", "env", "RUBBLE_GPU_LOCK=0", BIN],
                         env=env, capture_output=True, text=True)
    lines = (out.stdout + out.stderr).splitlines()
    return [l for l in lines if l.startswith("hash")], sorted({l.split(": ")[1] for l in lines if l.startswith("rigid kernel: ")})

if not os.path.exists(BIN):
    subprocess.run(["cargo", "build", "--profile", "fast", "-p", "engine-world-lab", "--example", "rigid_gate"], check=True)
ref, names = gate([], list_kernels=True)
skip = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set()
names = [n for n in names if n not in skip]
print(f"{len(names)} kernels in {SCENES}", flush=True)
found = []

def search(part):
    same = gate(part)[0] == ref
    print(("same    " if same else "DIFFER  ") + f"{len(part)}: " + ",".join(part)[:160], flush=True)
    if same:
        return
    if len(part) == 1:
        found.append(part[0])
        return
    half = len(part) // 2
    search(part[:half])
    search(part[half:])

search(names)
print("dependent kernels:", found)
