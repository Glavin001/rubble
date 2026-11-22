#!/usr/bin/env python3
"""How long the edit -> ready loop takes, edit by edit.

    python3 tools/build-time/measure.py [--runs N] [--only NAME,...] [--out FILE.md]
                                        (from engine/)

For each edit below: append a comment line to one file, rebuild the fast loop's
harness (the build fast-check.sh and the scenario matrix run), record the wall time,
the crates cargo compiled, the build scripts it ran and the slangc processes those
started; then put the file back and rebuild untimed, so each edit starts from a
built tree. Each probe is new (a kernel comment, an unused Rust function, never seen by a
build), so caches miss as they do after a real edit; the probe changes no output,
so what rebuilds is what the build graph makes rebuild, not what the edit needed.

slangc runs are counted by a wrapper (SLANGC points at it). The per-compile cache's
key includes slangc's path, so the first build under the wrapper compiles everything
once; that warm-up is not timed.

CMD overrides the build (default: the fast loop). RUNS (or --runs) repeats each edit
and reports the median. Nothing here touches the GPU.

--budget [FILE] (default tools/build-time/budget.txt): exit 1 when an edit's median is
past its line there (seconds, measured on the machine the file names; set with headroom
over the medians in engine/docs/measurements/build-time.md). A change that makes an edit
slower past its budget is a regression of the build: find why, or raise the line in the
same change where review sees it.
"""
import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_CMD = ["cargo", "build", "--profile", "fast", "-p", "engine-world-lab", "--example", "vehicle_lab"]

# name: (file under engine/, what it stands for)
EDITS = {
    "kernel-small": ("slang/kernels/soft_skin.slang", "a small kernel file"),
    "kernel-explicit": ("slang/kernels/explicit_beams.slang", "the explicit vehicles' kernel"),
    "kernel-common": ("slang/physics/common.slang", "the most-included shared kernel file"),
    "kernel-aero": ("slang/kernels/explicit_beams_aero.slang", "the explicit kernel's wings (EB_AERO)"),
    "softbody-runtime": ("crates/engine-softbody/src/explicit_beams.rs", "softbody's runtime model"),
    # (Where it is after the authoring split, else where it was: the tool measures both.)
    "softbody-authoring": (["crates/engine-softbody-authoring/src/authoring/vehicle.rs",
                            "crates/engine-softbody/src/authoring/vehicle.rs"], "vehicle authoring"),
    "exec-plan": ("crates/engine-exec/src/explicit_plan.rs", "the explicit plan"),
    "webgpu-host": ("crates/engine-webgpu/src/explicit_beams_gpu.rs", "the explicit GPU host"),
    "world": ("crates/engine-world/src/threshold.rs", "a World module"),
}

PROBE = "\n// build-time probe (tools/build-time/measure.py); remove if left behind\n"


def probe(path):
    """A new edit each time, as a real one is: a kernel file gets a comment no build
    has seen (so the per-compile cache misses as it would), a Rust file a function no
    build has seen."""
    n = time.time_ns()
    if path.endswith(".rs"):
        return (f"\n// build-time probe (tools/build-time/measure.py); remove if left behind\n"
                f"#[allow(dead_code)]\nfn build_time_probe_{n}() -> u64 {{ {n} }}\n")
    return f"\n// build-time probe {n} (tools/build-time/measure.py); remove if left behind\n"


def edit_file(name):
    """The edit's file (the first of its candidates that exists)."""
    rel = EDITS[name][0]
    for r in [rel] if isinstance(rel, str) else rel:
        if os.path.isfile(os.path.join(ENGINE, r)):
            return r
    sys.exit(f"{name}: none of {rel} exists")


def real_slangc():
    p = os.environ.get("SLANGC") or shutil.which("slangc") or os.path.expanduser("~/.local/bin/slangc")
    if not os.path.isfile(p):
        sys.exit("slangc not found")
    return os.path.realpath(p)


def make_wrapper(work, slangc):
    """A slangc that logs each start, kept at one path so cache keys stay put."""
    log = os.path.join(work, "slangc.log")
    wrapper = os.path.join(work, "slangc")
    with open(wrapper, "w") as fh:
        fh.write(f'#!/bin/sh\necho x >> "{log}"\nexec "{slangc}" "$@"\n')
    os.chmod(wrapper, 0o755)
    return wrapper, log


def count_lines(path):
    try:
        with open(path) as fh:
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0


def build(cmd, env):
    t = time.monotonic()
    proc = subprocess.run(cmd + ["-v"], cwd=ENGINE, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.exit("build failed")
    # Ready means the kernels too: with the fast profile engine-webgpu compiles them at
    # run time (kernel_store), so the edit's kernel work is this step's.
    profile = cmd[cmd.index("--profile") + 1] if "--profile" in cmd else ("release" if "--release" in cmd else "dev")
    warm = subprocess.run(["cargo", "run", "-q", "--profile", profile, "-p", "engine-webgpu", "--bin", "kernels"],
                          cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if warm.returncode != 0:
        sys.stderr.write(warm.stdout[-4000:])
        sys.exit("kernels failed")
    secs = time.monotonic() - t
    compiled = re.findall(r"^\s*Compiling (\S+)", proc.stdout, re.M)
    scripts = re.findall(r"build/([A-Za-z0-9_-]+?)-[0-9a-f]{16}/build-script-build`", proc.stdout)
    return secs, compiled, scripts


def measure(name, cmd, env, log, runs):
    rel = edit_file(name)
    path = os.path.join(ENGINE, rel)
    with open(path, "rb") as fh:
        original = fh.read()
    times = []
    try:
        for _ in range(runs):
            with open(path, "ab") as fh:
                fh.write(probe(path).encode())
            before = count_lines(log)
            secs, compiled, scripts = build(cmd, env)
            slangc = count_lines(log) - before
            times.append(secs)
            with open(path, "wb") as fh:
                fh.write(original)
            build(cmd, env)  # back to a built tree, untimed
    finally:
        with open(path, "wb") as fh:
            fh.write(original)
    # Only the workspace's crates rebuild after an edit inside it.
    crates = sorted(set(compiled))
    return {
        "name": name,
        "file": rel,
        "secs": statistics.median(times),
        "all": times,
        "crates": crates,
        "scripts": sorted(set(scripts)),
        "slangc": slangc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", "1")))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--budget", nargs="?", const=os.path.join(ENGINE, "tools", "build-time", "budget.txt"), default="")
    args = ap.parse_args()
    cmd = os.environ.get("CMD", "").split() or DEFAULT_CMD
    names = [n for n in args.only.split(",") if n] or list(EDITS)
    for n in names:
        if n not in EDITS:
            sys.exit(f"unknown edit {n}; known: {', '.join(EDITS)}")

    work = os.path.join(ENGINE, "out", "build-time", "wrapper")
    os.makedirs(work, exist_ok=True)
    wrapper, log = make_wrapper(work, real_slangc())
    env = dict(os.environ, SLANGC=wrapper)

    # The warm-up builds the tree and then reruns the kernel build scripts once under
    # the wrapper (an edit to a kernel file and back), so every kernel is in the cache
    # under the wrapper's key before anything is timed.
    print(f"warm-up: {' '.join(cmd)}", flush=True)
    secs, _, _ = build(cmd, env)
    warm = os.path.join(ENGINE, edit_file("kernel-common"))
    with open(warm, "rb") as fh:
        original = fh.read()
    try:
        with open(warm, "ab") as fh:
            fh.write(probe(warm).encode())
        secs += build(cmd, env)[0]
    finally:
        with open(warm, "wb") as fh:
            fh.write(original)
    secs += build(cmd, env)[0]
    print(f"  {secs:.1f} s", flush=True)

    rows = []
    for n in names:
        r = measure(n, cmd, env, log, args.runs)
        rows.append(r)
        print(f"{n:20s} {r['secs']:7.1f} s  crates {len(r['crates']):2d}  scripts {len(r['scripts'])}  "
              f"slangc {r['slangc']:4d}   ({', '.join(f'{t:.1f}' for t in r['all'])})", flush=True)

    lines = [
        f"Command: `{' '.join(cmd)}`, runs per edit: {args.runs} (median).",
        "",
        "| Edit | File | Seconds | Crates compiled | Build scripts run | slangc runs |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | `{r['file']}` | {r['secs']:.1f} | {len(r['crates'])} | "
                     f"{len(r['scripts'])} | {r['slangc']} |")
    lines.append("")
    for r in rows:
        lines.append(f"- {r['name']}: crates {', '.join(r['crates']) or '-'}; scripts {', '.join(r['scripts']) or '-'}")
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print()
    print(text)
    if args.budget:
        budget = {}
        for line in open(args.budget):
            line = line.split("#", 1)[0].strip()
            if line:
                name, secs = line.split()
                budget[name] = float(secs)
        over = [r for r in rows if r["name"] in budget and r["secs"] > budget[r["name"]]]
        for r in over:
            print(f"over budget: {r['name']} {r['secs']:.1f} s > {budget[r['name']]:.1f} s")
        print(f"budget: {len(rows)} edits, {len(over)} over")
        sys.exit(1 if over else 0)


if __name__ == "__main__":
    main()
