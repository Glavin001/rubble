"""LMGC90 oracle pipeline: scene -> deck -> run -> observation + provenance.

    python -I oracles/lmgc90/pipeline.py scenes/<scene>.json [--seed N] [--work DIR]

Writes ``golden/<scene>/lmgc90[_seedN].json`` (compact) and
``golden/<scene>/provenance_lmgc90[_seedN].json``. Each stage runs as its own
``python -I`` process (export.py, run.py, observe.py). Scratch decks go under
``--work`` (default /home/user/oracle-runs/lmgc90).
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage(args, log):
    t0 = time.time()
    with open(log, "w") as f:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise SystemExit("stage failed (%s): %s\nsee %s" % (r.returncode, " ".join(map(shlex.quote, map(str, args))), log))
    return time.time() - t0


def versions():
    py = sys.executable
    code = ("import json, importlib.metadata as m, numpy, pylmgc90;"
            "print(json.dumps({'pylmgc90_dist': m.version('pylmgc90'), 'numpy': numpy.__version__,"
            "'pylmgc90_path': pylmgc90.__file__}))")
    out = subprocess.run([py, "-I", "-c", code], capture_output=True, text=True)
    v = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-500:]}
    v["python"] = platform.python_version()
    v["lmgc90_source"] = "lmgc90_user_2026.rc1.zip (see oracles/lmgc90/PINNED.md)"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--work", default="/home/user/oracle-runs/lmgc90")
    ap.add_argument("--golden", default=str(ROOT / "golden"))
    a = ap.parse_args()

    scene_path = pathlib.Path(a.scene).resolve()
    scene = json.loads(scene_path.read_text())
    seed = a.seed if a.seed is not None else int(scene["sim"].get("seed", 0))
    suffix = "" if seed == 0 else "_seed%d" % seed
    work = pathlib.Path(a.work) / (scene["name"] + suffix)
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    run_scene = scene_path
    if a.seed is not None and seed != int(scene["sim"].get("seed", 0)):
        run_scene = work / "scene.json"
        subprocess.run([str(ROOT / "target/release/stress-ref"), "with-seed", str(scene_path), str(seed), str(run_scene)], check=True)

    deck = work / "deck"
    py = sys.executable
    t_export = stage([py, "-I", str(HERE / "export.py"), str(run_scene), str(deck)], work / "export.log")
    t_run = stage([py, "-I", str(HERE / "run.py"), str(deck)], work / "run.log")
    golden_dir = pathlib.Path(a.golden) / scene["name"]
    golden_dir.mkdir(parents=True, exist_ok=True)
    out = golden_dir / ("lmgc90%s.json" % suffix)
    t_obs = stage([py, "-I", str(HERE / "observe.py"), str(deck), str(out)], work / "observe.log")

    man = json.loads((deck / "manifest.json").read_text())
    run = json.loads((deck / "run.json").read_text())
    prov = {
        "tool": "lmgc90",
        "scene": scene["name"],
        "seed": seed,
        "scene_file": str(scene_path),
        "scene_sha256": sha256(scene_path),
        "run_scene_sha256": sha256(run_scene),
        "deck_sha256": man["deck_sha256"],
        "observation_sha256": sha256(out),
        "command": " ".join(shlex.quote(x) for x in [py, "-I"] + sys.argv),
        "versions": versions(),
        "solver_version": run["solver_version"],
        "params": man["params"],
        "dt": man["dt"],
        "n_settle": man["n_settle"],
        "n_steps": man["n_steps"],
        "nlgs": {k: v for k, v in run["nlgs"].items() if k not in ("impact_iterations", "impact_converged")},
        "wall_seconds": {"export": t_export, "run": t_run, "observe": t_obs, "total": time.time() - t_start,
                         "solver_settle": run["settle_seconds"], "solver_impact": run["impact_seconds"]},
        "threads": 1,
        "host": platform.platform(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "work_dir": str(work),
    }
    (golden_dir / ("provenance_lmgc90%s.json" % suffix)).write_text(json.dumps(prov, indent=1) + "\n")
    print("wrote %s (%.0f s)" % (out, prov["wall_seconds"]["total"]))


if __name__ == "__main__":
    main()
