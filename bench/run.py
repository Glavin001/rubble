#!/usr/bin/env python3
"""Run the cross-engine benchmark suite and write a comparison report.

  python3 bench/run.py [--group webgpu|metal|cuda|cpu|all] [--engines name,...]
                       [--scenes tier|id,...] [--sweep] [--repeat N] [--smoke] [--dry-run]
                       [--out DIR] [--no-build] [--max-seconds S] [--list]
  python3 bench/run.py report DIR

Comparison groups (README "Comparisons"):
  webgpu  rubble's WGSL path vs three-avbd (same API family)
  metal   rubble's native Metal path vs the GPU engines that run on Metal here
          (three-avbd on Dawn/Metal, PhysX's CUDA GPU module through CuMetal)
  cuda    rubble's CUDA path (pending) vs PhysX GPU (CUDA on NVIDIA, CuMetal on macOS)
  cpu     rapier vs PhysX CPU, one thread and many

Every engine runs every scene it supports, one process per run, strictly one at a time, on the
same scene files and the same timing protocol. Each run is judged from its pose dumps by
bench-judge, and DIR/report.md compares wall time at matched quality, per group.
`--smoke` runs a few steps per run to prove every adapter builds, loads and steps (almost no
compute); `--dry-run` prints the run plan without running anything.
"""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent
SCENES = BENCH / "scenes" / "out"
TARGET = BENCH / "target" / "release"
RESULTS = BENCH / "results"
TIERS = ["small", "steady", "fracture", "joints", "paper", "large"]
# PhysX 5.10 package whose CUDA GPU module runs on Metal through CuMetal (sibling repos
# ../PhysX and ../cuda-metal); override with PHYSX_CUMETAL_DIR
CUMETAL_PHYSX = Path(os.environ.get("PHYSX_CUMETAL_DIR", ROOT.parent / "PhysX" / "out" / "install" / "macos-cumetal" / "release"))


def find_node():
    """Node ≥ 22.6 (TypeScript type stripping) for the scene exporter and three-avbd."""
    cands = [shutil.which("node")]
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if nvm.is_dir():
        cands += [str(p / "bin" / "node") for p in sorted(nvm.iterdir(), reverse=True)]
    for c in filter(None, cands):
        try:
            v = subprocess.run([c, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
            major, minor = (int(x) for x in v.split(".")[:2])
            if (major, minor) >= (22, 6):
                return c
        except (OSError, ValueError):
            pass
    return None


NODE = find_node()
TS = ["--experimental-transform-types", "--no-warnings"]


def cpu_threads():
    """Threads for the "-mt" CPU runs: the performance cores on Apple silicon (its efficiency
    cores slow a fork-join step down), every logical core elsewhere."""
    if platform.system() == "Darwin":
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout.strip()
        if out.isdigit():
            return int(out)
    return os.cpu_count() or 1


def physx_mode():
    """How target/release/bench-physx was built: 'cumetal' (PhysX 5.10, CPU + GPU on Metal),
    'sdk' (official SDK; GPU only with CUDA), 'vendored' (PhysX 5.1, CPU) or None."""
    exe = TARGET / "bench-physx"
    if not exe.exists():
        return None
    if platform.system() == "Darwin":
        libs = subprocess.run(["otool", "-L", str(exe)], capture_output=True, text=True).stdout
        if "libcumetal" in libs:
            return "cumetal"
    stamp = TARGET / "bench-physx.mode"
    return stamp.read_text().strip() if stamp.exists() else "vendored"


def physx_gpu_available():
    mode = physx_mode()
    return mode == "cumetal" or (mode == "sdk-gpu" and shutil.which("nvidia-smi") is not None)


# --- Engines -------------------------------------------------------------------------------
# Each engine: command prefix, whether it runs on the GPU queue (then it is also timed in sync
# mode), what scenes it supports, per-scene settings it ships with, the iteration values a
# --sweep tries (None = the engine's own default / the scene's reference count) and named
# production variants a --sweep also tries.
PHYSX = lambda: str(TARGET / "bench-physx")  # noqa: E731
PHYSX_VARIANTS = {"depen10": ["--set", "max_depen_vel=10"]}
ENGINES = {
    "rubble-wgpu": {
        "cmd": lambda: [str(TARGET / "bench-rubble"), "--set", "backend=wgpu"],
        "gpu": True,
        "supports": lambda m: m["counts"].get("hulls", 0) == 0,
        "unsupported": "convex hulls are not in rubble's native wgpu host yet (they run in its browser demo)",
        # rubble's own per-scene production defaults (engine/crates/engine-webgpu/README.md)
        "scene_args": {"brick_ring_110k": ["--set", "contacts=persistent"]},
        "sweep": [None, "half", "double"],
    },
    "rubble-metal": {
        "cmd": lambda: [str(TARGET / "bench-rubble"), "--set", "backend=metal"],
        "gpu": True,
        "available": lambda: platform.system() == "Darwin",
        "sweep": [None, "half", "double"],
    },
    "rubble-cuda": {
        "cmd": lambda: [],
        "gpu": True,
        # engine/crates/engine-cuda has a host (CudaSim, tested on Modal); a bench-rubble
        # adapter for it is next. It runs on NVIDIA, and on macOS through CuMetal like PhysX
        "available": lambda: False,
        "why": "pending: no bench adapter for engine-cuda's CudaSim yet",
    },
    "three-avbd": {
        "cmd": lambda: [NODE, *TS, str(BENCH / "adapters" / "three-avbd" / "run.ts")],
        "gpu": True,
        "supports": lambda m: m["counts"].get("compoundBodies", 0) == 0 and m.get("version", 1) < 3,
        "unsupported": "three-avbd has no multi-hull (compound) bodies, and no hinge / cone / limit / motor joints (format v3)",
        "available": lambda: NODE is not None and (BENCH / "adapters" / "three-avbd" / "node_modules" / "webgpu").exists(),
        "sweep": [None, "half", "double"],
    },
    "rapier-1t": {"cmd": lambda: [str(TARGET / "bench-rapier"), "--threads", "1"], "sweep": [1, 2, 4, 8, 16], "supports": lambda m: m.get("version", 1) < 3, "unsupported": "the rapier adapter maps ball joints only (format v3 D6 joints pending)"},
    "rapier-mt": {"cmd": lambda: [str(TARGET / "bench-rapier"), "--threads", str(cpu_threads())], "sweep": [1, 2, 4, 8, 16], "supports": lambda m: m.get("version", 1) < 3, "unsupported": "the rapier adapter maps ball joints only (format v3 D6 joints pending)"},
    # Box3D (Erin Catto) and Jolt: CPU engines built from bench/third_party (fetch.sh)
    "box3d-1t": {"cmd": lambda: [str(TARGET / "bench-box3d"), "--threads", "1"], "available": lambda: (TARGET / "bench-box3d").exists(), "sweep": [1, 2, 4, 8, 16], "supports": lambda m: m.get("version", 1) < 3, "unsupported": "the Box3D adapter maps ball joints only (format v3 D6 joints pending)"},
    "box3d-mt": {"cmd": lambda: [str(TARGET / "bench-box3d"), "--threads", str(cpu_threads())], "available": lambda: (TARGET / "bench-box3d").exists(), "sweep": [1, 2, 4, 8, 16], "supports": lambda m: m.get("version", 1) < 3, "unsupported": "the Box3D adapter maps ball joints only (format v3 D6 joints pending)"},
    "jolt-1t": {"cmd": lambda: [str(TARGET / "bench-jolt"), "--threads", "1"], "available": lambda: (TARGET / "bench-jolt").exists(), "sweep": [2, 5, 10, 20]},
    "jolt-mt": {"cmd": lambda: [str(TARGET / "bench-jolt"), "--threads", str(cpu_threads())], "available": lambda: (TARGET / "bench-jolt").exists(), "sweep": [2, 5, 10, 20]},
    # PhysX's default maxDepenetrationVelocity is unbounded; the `depen10` variant caps it at
    # 10 m/s, a common production setting (rapier bounds its corrective velocity by default)
    "physx-cpu-1t": {"cmd": lambda: [PHYSX(), "--threads", "1", "--set", "gpu=off"], "available": lambda: physx_mode() is not None, "sweep": [1, 2, 4, 8, 16], "variants": PHYSX_VARIANTS},
    "physx-cpu-mt": {"cmd": lambda: [PHYSX(), "--threads", str(cpu_threads()), "--set", "gpu=off"], "available": lambda: physx_mode() is not None, "sweep": [1, 2, 4, 8, 16], "variants": PHYSX_VARIANTS},
    "physx-gpu": {
        "cmd": lambda: [PHYSX(), "--set", "gpu=on"],
        "gpu": True,
        "available": physx_gpu_available,
        "why": "needs the CuMetal PhysX package (macOS) or an official SDK GPU build with an NVIDIA driver",
        "sweep": [1, 2, 4, 8, 16],
        "variants": PHYSX_VARIANTS,
    },
}
GROUPS = {
    "webgpu": ["rubble-wgpu", "three-avbd"],
    "metal": ["rubble-metal", "three-avbd", "physx-gpu"],
    "cuda": ["rubble-cuda", "physx-gpu"],
    "cpu": ["rapier-1t", "rapier-mt", "box3d-1t", "box3d-mt", "jolt-1t", "jolt-mt", "physx-cpu-1t", "physx-cpu-mt"],
}
GROUPS["all"] = list(dict.fromkeys(sum(GROUPS.values(), [])))


def available(name):
    return ENGINES[name].get("available", lambda: True)()


def run_env():
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local' / 'bin'}{os.pathsep}{env['PATH']}"  # slangc (engine/TOOLCHAIN.md)
    # CuMetal's writable pipeline cache stays inside the bench (its read-only archive ships
    # beside libcumetal in the PhysX package)
    env.setdefault("CUMETAL_CACHE_DIR", str(RESULTS / "cumetal-cache"))
    return env


def sh(cmd, **kw):
    print("+", " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    return subprocess.run(cmd, **kw)


def build(engines):
    env = run_env()
    pkgs = ["bench-judge"]
    if any(e.startswith("rapier") for e in engines):
        pkgs.append("bench-rapier")
    if any(e.startswith(("box3d", "jolt")) for e in engines):
        # pinned sources for the CPU engines built from source (bench/third_party/fetch.sh)
        sh([str(BENCH / "third_party" / "fetch.sh")], cwd=BENCH)
    if any(e.startswith("box3d") for e in engines):
        pkgs.append("bench-box3d")
    if any(e.startswith("jolt") for e in engines) and (BENCH / "adapters" / "jolt" / "Cargo.toml").exists():
        pkgs.append("bench-jolt")
    if any(e.startswith("rubble") for e in engines):
        pkgs.append("bench-rubble")
    if sh(["cargo", "build", "--release", *sum((["-p", p] for p in pkgs), [])], cwd=BENCH, env=env).returncode:
        sys.exit("cargo build failed")
    if any(e.startswith("physx") for e in engines):
        # Prefer the CuMetal package (PhysX 5.10, CPU + GPU on Metal), then an official SDK
        # (PHYSX_SDK_DIR; GPU with CUDA), else PhysX 5.1 vendored by physx-sys (CPU)
        if platform.system() == "Darwin" and (CUMETAL_PHYSX / "lib" / "libcumetal.dylib").exists():
            env["PHYSX_CUMETAL_DIR"] = str(CUMETAL_PHYSX)
            feats, mode = ["--no-default-features", "--features", "gpu"], "cumetal"
        elif env.get("PHYSX_SDK_DIR"):
            gpu = shutil.which("nvidia-smi") is not None
            feats, mode = ["--no-default-features"] + (["--features", "gpu"] if gpu else []), "sdk-gpu" if gpu else "sdk"
        else:
            env.pop("PHYSX_CUMETAL_DIR", None)
            feats, mode = [], "vendored"
        if sh(["cargo", "build", "--release", "-p", "bench-physx", *feats], cwd=BENCH, env=env).returncode:
            print("warning: bench-physx did not build; PhysX runs are skipped (adapters/physx/README.md)")
        else:
            (TARGET / "bench-physx.mode").write_text(mode + "\n")
    if "three-avbd" in engines and NODE and not (BENCH / "adapters" / "three-avbd" / "node_modules" / "webgpu").exists():
        npm = Path(NODE).with_name("npm")
        sh([str(npm) if npm.exists() else "npm", "install", "--no-audit", "--no-fund"], cwd=BENCH / "adapters" / "three-avbd")


def export_scenes():
    if not NODE:
        sys.exit("Node ≥ 22.6 is needed to export the scenes (bench/scenes/export.ts)")
    if sh([NODE, *TS, str(BENCH / "scenes" / "export.ts")], cwd=ROOT).returncode:
        sys.exit("scene export failed")
    # Shardwright fracture assets (rubble's own importer; tools/fracture-export)
    if sh(["cargo", "run", "--release", "-q", "-p", "bench-fracture-export"], cwd=BENCH).returncode:
        print("warning: fracture scene export failed; fracture tier skipped")
    # rubble's D6 joint scenes (engine-scenes builders; tools/joint-export; needs Metal)
    if platform.system() == "Darwin" and sh(["cargo", "run", "--release", "-q", "-p", "bench-joint-export"], cwd=BENCH, env=run_env()).returncode:
        print("warning: joint scene export failed; joints tier skipped")


def scene_list(spec):
    metas = [json.loads(p.read_text()) for p in sorted(SCENES.glob("*.scene.json"))]
    want = spec.split(",") if spec else ["small", "steady", "fracture"]
    picked = [m for m in metas if m["tier"] in want or m["id"] in want]
    order = {t: i for i, t in enumerate(TIERS)}
    return sorted(picked, key=lambda m: (order.get(m["tier"], 9), m["counts"]["bodies"]))


def iteration_arg(value, meta):
    if value is None:
        return []
    if value == "half":
        return ["--iterations", str(max(1, meta["iterations"] // 2))]
    if value == "double":
        return ["--iterations", str(meta["iterations"] * 2)]
    return ["--iterations", str(value)]


def plan(engines, metas, args):
    """Every run, in execution order: scene-major, engines interleaved, so slow drift
    (thermals, background work) is shared."""
    runs = []
    for meta in metas:
        for rep in range(args.repeat):
            for name in engines:
                eng = ENGINES[name]
                if not eng.get("supports", lambda m: True)(meta):
                    runs.append({"engine": name, "scene": meta["id"], "unsupported": eng.get("unsupported", "unsupported")})
                    continue
                configs = [(None, None)]
                if args.sweep and not args.smoke:
                    configs += [(None, v) for v in eng.get("sweep", []) if v is not None]
                    configs += [(var, it) for var in eng.get("variants", {}) for it in [None] + [v for v in eng.get("sweep", []) if v is not None]]
                for variant, iters in configs:
                    modes = ["batch", "sync"] if eng.get("gpu") and not args.smoke else ["batch"]
                    for mode in modes:
                        if mode == "sync" and (iters is not None or variant):
                            continue  # sweeps compare throughput; latency is reported at defaults
                        runs.append({"engine": name, "meta": meta, "mode": mode, "iters": iters, "variant": variant, "rep": rep})
    return runs


def command(out, r, args):
    name, meta, eng = r["engine"], r["meta"], ENGINES[r["engine"]]
    tag = name + (f"-{r['variant']}" if r["variant"] else "") + (f"-it{iteration_arg(r['iters'], meta)[1]}" if r["iters"] is not None else "")
    tag += f"-{r['mode']}" + (f"-r{r['rep']}" if args.repeat > 1 else "")
    result = out / meta["id"] / f"{tag}.json"
    cmd = [*eng["cmd"](), "--scene", str(SCENES / f"{meta['id']}.scene.json"), "--out", str(result), "--mode", r["mode"], "--quiet", "--label", tag]
    cmd += iteration_arg(r["iters"], meta) + eng.get("scene_args", {}).get(meta["id"], []) + (eng["variants"][r["variant"]] if r["variant"] else [])
    if args.max_seconds:
        cmd += ["--max-seconds", str(args.max_seconds)]
    if args.smoke:
        cmd.append("--smoke")
    return tag, result, cmd


def wait_idle(args):
    """--wait-idle: hold each run until the 1-minute load average is below the threshold, so a
    scheduled benchmark neither measures nor competes with other jobs."""
    if not args.wait_idle or not hasattr(os, "getloadavg"):
        return
    start = time.time()
    while os.getloadavg()[0] >= args.wait_idle:
        if time.time() - start > args.wait_max:
            sys.exit(f"machine still busy (load {os.getloadavg()[0]:.1f} ≥ {args.wait_idle}) after {args.wait_max:.0f} s; stopping")
        print(f"  waiting: load {os.getloadavg()[0]:.1f} ≥ {args.wait_idle}", flush=True)
        time.sleep(30)


def run_one(out, r, args):
    wait_idle(args)
    tag, result, cmd = command(out, r, args)
    result.parent.mkdir(parents=True, exist_ok=True)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    t = time.time()
    p = sh(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout, env=run_env())
    status = {"engine": r["engine"], "scene": r["meta"]["id"], "mode": r["mode"], "iterations": r["iters"], "variant": r["variant"], "repeat": r["rep"], "result": str(result.relative_to(out)), "loadBefore": load, "seconds": time.time() - t}
    if p.returncode or not result.exists():
        status["error"] = (p.stderr or p.stdout).strip().splitlines()[-5:]
        print(f"  ERROR {r['engine']} {r['meta']['id']}: {' | '.join(status['error'])}")
        return status
    j = subprocess.run([str(TARGET / "bench-judge"), "--result", str(result), "--scene", str(SCENES / f"{r['meta']['id']}.scene.json")], capture_output=True, text=True)
    print("  " + (j.stderr or j.stdout).strip())
    if not args.smoke:
        print(f"  wall {json.loads(result.read_text())['timing']['wallMs']:.3f} ms/step")
    return status


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report(Path(sys.argv[2]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=list(GROUPS), help="a comparison group (default all)")
    ap.add_argument("--engines", help=f"explicit engines: {','.join(ENGINES)}")
    ap.add_argument("--scenes", help="tiers (small,steady,fracture,paper,large) and/or scene ids; default small,steady,fracture")
    ap.add_argument("--sweep", action="store_true", help="also run each engine's iteration sweep and variants")
    ap.add_argument("--repeat", type=int, default=1, help="repeat every run N times (report takes the median)")
    ap.add_argument("--smoke", action="store_true", help="a few steps per run: proves everything runs, measures nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the run plan and commands, run nothing")
    ap.add_argument("--out", type=Path, help="results directory (default bench/results/<UTC time>)")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--export", action="store_true", help="re-export the scenes first")
    ap.add_argument("--max-seconds", type=float, help="cap each run's untimed tail (later gates read 'not reached')")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--wait-idle", type=float, metavar="LOAD", help="before each run, wait until the 1-minute load average is below LOAD")
    ap.add_argument("--wait-max", type=float, default=6 * 3600, help="give up waiting for idle after this many seconds")
    ap.add_argument("--list", action="store_true", help="list scenes, engines and their availability")
    args = ap.parse_args()

    engines = args.engines.split(",") if args.engines else GROUPS[args.group or "all"]
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        sys.exit(f"unknown engines {unknown}; choose from {list(ENGINES)}")
    if args.export or not any(SCENES.glob("*.scene.json")):
        export_scenes()
    if not args.no_build and not args.dry_run and not args.list:
        build(engines)
    metas = scene_list(args.scenes)
    if args.list:
        for m in metas:
            c = m["counts"]
            extra = f" {c['hulls']:>5} hulls ({c.get('compoundBodies', 0)} compound)" if c.get("hulls") else ""
            print(f"{m['id']:22} {m['tier']:8} {c['bodies']:>7} bodies {c['joints']:>6} joints{extra}")
        for e in ENGINES:
            print(f"{e:14} {'available' if available(e) else 'unavailable: ' + ENGINES[e].get('why', 'not built / not on this OS')}")
        print("groups:", ", ".join(f"{g}={'+'.join(v)}" for g, v in GROUPS.items() if g != "all"))
        return
    skipped = [e for e in engines if not available(e)]
    engines = [e for e in engines if available(e)]
    for e in skipped:
        print(f"skipping {e}: {ENGINES[e].get('why', 'not available on this machine')}", file=sys.stderr)
    runs = plan(engines, metas, args)
    out = args.out or RESULTS / (("smoke-" if args.smoke else "") + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    todo = [r for r in runs if "meta" in r]
    if args.dry_run:
        gpu = sum(1 for r in todo if ENGINES[r["engine"]].get("gpu"))
        print(f"{len(todo)} runs ({gpu} on the GPU, {len(todo) - gpu} on the CPU) over {len(metas)} scenes into {out}")
        for r in todo:
            print(" ".join(shlex.quote(str(c)) for c in command(out, r, args)[2]))
        return
    out.mkdir(parents=True, exist_ok=True)
    statuses = [{k: v for k, v in r.items() if k != "meta"} for r in runs if "meta" not in r]
    current = None
    for r in todo:
        if r["meta"]["id"] != current:
            current = r["meta"]["id"]
            print(f"\n== {current} ({r['meta']['counts']['bodies']} bodies, {r['meta']['counts']['joints']} joints)")
        statuses.append(run_one(out, r, args))
        (out / "runs.json").write_text(json.dumps(statuses, indent=1) + "\n")
    report(out)


# --- Report ----------------------------------------------------------------------------------
def fmt_ms(v):
    return "—" if v is None else (f"{v:.2f}" if v < 100 else f"{v:.0f}")


def load_runs(out):
    rows = []
    for res in sorted(out.glob("*/*.json")):
        if res.name.endswith(".judge.json"):
            continue
        doc = json.loads(res.read_text())
        if doc.get("format") != "rubble-bench-result":
            continue
        jpath = res.with_suffix(".judge.json")
        judge = json.loads(jpath.read_text()) if jpath.exists() else None
        rows.append({"path": res, "doc": doc, "judge": judge})
    return rows


def engine_key(doc):
    name = doc.get("label") or doc["engine"]["engine"]
    head, _, tail = name.rpartition("-r")
    return head if head and tail.isdigit() else name


def family(key):
    for f in sorted(ENGINES, key=len, reverse=True):
        if key == f or key.startswith(f + "-"):
            return f
    return key


def median(v):
    s = sorted(v)
    return s[len(s) // 2] if s else None


def report(out):
    runs = load_runs(out)
    if not runs:
        print(f"no results in {out}")
        return
    status = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else []
    smoke = any(r["doc"]["protocol"].get("smoke") for r in runs)
    machine = runs[0]["doc"]["machine"]
    lines = [
        "# Physics engine benchmark" + (" — SMOKE RUN (plumbing only: no timing or quality claims)" if smoke else ""),
        "",
        f"Machine: **{machine.get('cpu')}**, {machine.get('logicalCores')} logical cores, {machine.get('os')} {machine.get('osVersion')}"
        + (f", {machine['memoryBytes'] / 2**30:.0f} GB" if machine.get("memoryBytes") else "")
        + (f", GPU {machine['gpu']}" if machine.get("gpu") else ""),
        f"Results: `{out}`",
        "",
        "Protocol: identical scene files for every engine (three-avbd's builders, Y-up), same lead-in, "
        "2 rounds × 5 batches × 10 timed steps; **wall** = min over rounds of the median batch "
        "(ms per step, submission to GPU completion); **sync** = per-step latency p50 with a drain after every step. "
        "Quality is judged from pose dumps by one engine-neutral judge; a run's time only counts when it **passes** "
        "the scene's gates. See bench/README.md.",
        "",
    ]
    preamble = len(lines)
    by_scene = {}
    for r in runs:
        by_scene.setdefault(r["doc"]["scene"]["id"], []).append(r)
    order = {t: i for i, t in enumerate(TIERS)}
    summary, headline = {}, []
    for sid, rs in sorted(by_scene.items(), key=lambda kv: (order.get(kv[1][0]["doc"]["scene"]["tier"], 9), kv[1][0]["doc"]["scene"]["bodies"])):
        sc = rs[0]["doc"]["scene"]
        groups = {}
        for r in rs:
            groups.setdefault(engine_key(r["doc"]), []).append(r)
        lines += [f"## {sid}", "", f"{sc['bodies']:,} bodies ({sc['dynamic']:,} dynamic), {sc['joints']:,} joints, dt {sc['dt']:.5f}, AVBD reference {sc['referenceIterations']} iterations.", ""]
        lines += ["| run | backend | iters | wall ms | sync p50 ms | quality | failed gates | pen p99 (last) | notes |", "|---|---|---:|---:|---:|---|---|---:|---|"]
        best = {}
        for key, grp in sorted(groups.items(), key=lambda kv: median([g["doc"]["timing"]["wallMs"] for g in kv[1]])):
            doc = grp[0]["doc"]
            walls = [g["doc"]["timing"]["wallMs"] for g in grp]
            wall = median(walls)
            spread = (max(walls) - min(walls)) if len(grp) > 1 else None
            step = doc["timing"].get("step")
            judges = [g["judge"] for g in grp if g["judge"]]
            passed = bool(judges) and all(j["passed"] for j in judges)
            failed = sorted({f"{g['metric']}@{g['at']}" + (f"[{g['group']}]" if g.get("group") else "") for j in judges for g in j["gates"] if g["status"] != "pass"})
            if judges and not judges[0]["initialMatches"]:
                failed.insert(0, "INITIAL STATE MISMATCH")
            last = judges[0]["checkpoints"][-1]["metrics"] if judges else {}
            e = doc["engine"]
            notes = []
            if e.get("threads", 1) > 1:
                notes.append(f"{e['threads']} threads")
            if doc["protocol"]["truncated"]:
                notes.append("truncated")
            if spread is not None:
                notes.append(f"±{spread / 2:.2f} over {len(grp)}")
            loads = [s.get("loadBefore") for s in status if s.get("result", "").endswith(grp[0]["path"].name)]
            if loads and loads[0] and loads[0] > 2.0:
                notes.append(f"load {loads[0]:.1f}")
            verdict = ("SMOKE OK" if passed else "SMOKE FAIL") if smoke else ("PASS" if passed else ("FAIL" if judges else "unjudged"))
            lines.append(
                f"| {key} | {e['backend']} | {e['iterations']} | {fmt_ms(wall)} | {fmt_ms(step['p50Ms']) if step else '—'} | "
                f"{verdict} | {', '.join(failed[:4])}{' …' if len(failed) > 4 else ''} | "
                f"{last.get('pen_p99', float('nan')) * 1000:.1f} mm | {'; '.join(notes)} |"
            )
            fam = family(key)
            if passed and not smoke and doc["protocol"]["mode"] == "batch" and (fam not in best or wall < best[fam][0]):
                best[fam] = (wall, key)
        for s in status:
            if s.get("scene") == sid and s.get("unsupported"):
                lines.append(f"| {s['engine']} | | | | | n/a | {s['unsupported']} | | |")
            elif s.get("scene") == sid and s.get("error"):
                lines.append(f"| {s['engine']} | | | | | ERROR | {' '.join(s['error'])[-120:]} | | |")
        lines.append("")
        summary[sid] = {fam: {"wallMs": w, "run": k} for fam, (w, k) in best.items()}
        headline.append((sid, best))
    head = []
    if not smoke:
        head = ["## Headlines: fastest passing configuration per comparison group (wall ms/step)", "",
                "Only runs that pass every quality gate count; `fail` = no configuration of that engine passed, `—` = not run.", ""]
        present = {family(engine_key(r["doc"])) for r in runs}
        for g, members in GROUPS.items():
            if g == "all":
                continue
            cols = [m for m in members if m in present]
            if not any(c.startswith("rubble") for c in cols) or len(cols) < 2:
                continue
            head += [f"### {g}: " + " vs ".join(cols), "", "| scene | " + " | ".join(cols) + " | rubble vs best other |", "|---|" + "---:|" * len(cols) + "---:|"]
            for sid, best in headline:
                ran = {family(engine_key(r["doc"])) for r in by_scene[sid]}
                cells = [fmt_ms(best[c][0]) if c in best else ("fail" if c in ran else "—") for c in cols]
                ours = min([best[c][0] for c in cols if c.startswith("rubble") and c in best], default=None)
                other = min([best[c][0] for c in cols if not c.startswith("rubble") and c in best], default=None)
                ratio = "—"
                if ours and other:
                    ratio = f"{other / ours:.2f}× faster" if ours < other else f"{ours / other:.2f}× slower"
                elif ours:
                    ratio = "only rubble passes"
                head.append(f"| {sid} | " + " | ".join(cells) + f" | {ratio} |")
            head.append("")
    text = "\n".join(lines[:preamble] + head + lines[preamble:]) + "\n"
    (out / "report.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\nwrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
