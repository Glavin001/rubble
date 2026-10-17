#!/usr/bin/env python3
"""Peak destruction frame time of rubble's integrated pipeline (rubble rigid bodies and
the GPU stress solver on one device), and a bit-identity guard.

    destruction_peak.py metric   one number: the worst frame (ms) over the destruction
                                 scenes, each scene's worst frame taken as the median
                                 over --repeats runs (GPU contention makes single
                                 frames noisy)
    destruction_peak.py guard    exit 0 when every scene's end-state digest equals the
                                 recorded one (`--record` writes them)

Runs the release `stress_bench` example (built first unless --no-build) with
STRESS_GPU_FIXED_HORIZON=1, so outcomes are deterministic and only timing varies.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "target", "release", "examples", "stress_bench")
# (set, regime): the destruction scenes the metric covers.
METRIC_SCENES = [("realtime", "house_ram"), ("towns", "town16_one_ram")]
GUARD_SETS = ["realtime", "accuracy"]
DIGESTS = os.path.join(ROOT, "scripts", "destruction_digests.json")


def build():
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "stress-rubble", "--features", "gpu", "--example", "stress_bench"], cwd=ROOT, check=True)


def run(set_name, only=None):
    args = [BENCH, "--solver", "rubble-gpu", "--set", set_name] + (["--only", only] if only else [])
    env = dict(os.environ, STRESS_GPU_FIXED_HORIZON="1")
    out = subprocess.run(args, cwd=ROOT, env=env, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def metric(repeats):
    worst = 0.0
    for set_name, regime in METRIC_SCENES:
        peaks = []
        for _ in range(repeats):
            rows = run(set_name, regime)
            if len(rows) != 1 or "frame_ms" not in rows[0]:
                sys.exit(f"{regime}: no result")
            peaks.append(max(rows[0]["frame_ms"]))
        peak = statistics.median(peaks)
        print(f"{regime}: worst frame {peak:.1f} ms (runs {', '.join(f'{p:.1f}' for p in peaks)})", file=sys.stderr)
        worst = max(worst, peak)
    print(f"{worst:.1f}")


def digests():
    return {f"{s}/{r['regime']}": r.get("digest", "failed: " + str(r.get("failed"))) for s in GUARD_SETS for r in run(s)}


def guard(record):
    now = digests()
    if record:
        with open(DIGESTS, "w") as f:
            json.dump(now, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"recorded {len(now)} digests")
        return
    with open(DIGESTS) as f:
        want = json.load(f)
    bad = [k for k in want if now.get(k) != want[k]] + [k for k in now if k not in want]
    for k in bad:
        print(f"{k}: {now.get(k)} != {want.get(k)}", file=sys.stderr)
    print(f"{len(want) - len(bad)}/{len(want)} digests identical")
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["metric", "guard"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--record", action="store_true")
    p.add_argument("--no-build", action="store_true")
    a = p.parse_args()
    if not a.no_build:
        build()
    metric(a.repeats) if a.mode == "metric" else guard(a.record)


if __name__ == "__main__":
    main()
