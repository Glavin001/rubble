"""OpenCourant oracle pipeline: export -> run -> observe.

    python -I oracles/opencourant/pipeline.py scenes/<scene>.json [--seed N]
           [--elements-per-chunk K] [--cohesive-thickness F] [--threads T] [--rundir DIR]

Writes golden/<scene>/opencourant[_seedN].json (compact) and
golden/<scene>/provenance_opencourant[_seedN].json.  For --seed N != the scene's seed the
scene is first re-derived with `stress-ref with-seed` (Weibull strengths), and the
re-derived scene file is what gets exported and hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # stress-oracle/
sys.path.insert(0, HERE)

import export  # noqa: E402
import observe  # noqa: E402
import run as runner  # noqa: E402

RUNS = os.environ.get("OPENCOURANT_RUNS", "/home/user/oracle-runs/opencourant")
STRESS_REF = os.environ.get("STRESS_REF", os.path.join(ROOT, "target", "release", "stress-ref"))
PINNED = {
    "tool": "OpenCourant (OpenRadioss fork)",
    "release_tag": "latest-20261006",
    "commit": "33e685176cccf0c539a3ce07aa2096985a284e2a",
    "asset": "OpenCourant_linux64.zip",
    "asset_sha256": "9d67531de156dd9beba05fbfe710dcdc2bcecbdf3dc3a12642cf85dcece80081",
    "url": "https://github.com/OpenCourant/OpenCourant/releases/download/latest-20261006/OpenCourant_linux64.zip",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--elements-per-chunk", type=int)
    ap.add_argument("--cohesive-thickness", type=float)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--chunk-cdpm2", action="store_true",
                    help="chunk solids as concrete (CDPM2, /MAT/LAW124): written as oracle `opencourant_cdpm2`")
    ap.add_argument("--chunk-crushing", action="store_true",
                    help="chunk solids crush at f_c (J2, eroded at G_c per area): written as oracle `opencourant_crushing`")
    ap.add_argument("--rundir")
    ap.add_argument("--golden", default=os.path.join(ROOT, "golden"))
    ap.add_argument("--stress-ref", default=STRESS_REF, help="stress-ref binary used for `with-seed`")
    a = ap.parse_args(argv)

    t_start = time.time()
    derived_cmd = None
    scene_path = os.path.abspath(a.scene)
    scene = json.load(open(scene_path))
    name = scene["name"]
    seed = scene["sim"].get("seed", 0) if a.seed is None else a.seed
    suffix = "" if seed == 0 else f"_seed{seed}"
    rundir = os.path.abspath(a.rundir or os.path.join(RUNS, name + suffix))
    os.makedirs(rundir, exist_ok=True)
    for f in os.listdir(rundir):
        os.remove(os.path.join(rundir, f))
    if seed != scene["sim"].get("seed", 0):
        derived = os.path.join(rundir, f"{name}{suffix}.scene.json")
        subprocess.run([a.stress_ref, "with-seed", scene_path, str(seed), derived], check=True)
        derived_cmd = f"{a.stress_ref} with-seed {a.scene} {seed} {derived}"
        scene_path = derived
        scene = json.load(open(scene_path))

    cli = {"elements_per_chunk": a.elements_per_chunk, "cohesive_thickness": a.cohesive_thickness,
           "threads": a.threads, "chunk_crushing": True if a.chunk_crushing else None,
           "chunk_cdpm2": True if a.chunk_cdpm2 else None}
    opts = export.options_from(scene, cli)
    meta = export.export(scene, rundir, opts)
    info = runner.run(rundir, name, threads=int(opts["threads"]))
    obs = observe.observe(scene, rundir, seed=seed)
    obs["notes"].append(f"mesh: {opts['elements_per_chunk']} elements per smallest chunk edge, cohesive layer "
                        f"thickness {opts['cohesive_thickness']} x element size; {meta['n_elements']} elements "
                        f"({meta['n_cohesive']} cohesive), {meta['n_nodes']} nodes")

    out_dir = os.path.join(a.golden, name)
    os.makedirs(out_dir, exist_ok=True)
    oracle = "opencourant_cdpm2" if opts.get("chunk_cdpm2") else "opencourant_crushing" if opts.get("chunk_crushing") else "opencourant"
    if oracle != "opencourant":
        obs["solver"] = oracle
    out = os.path.join(out_dir, f"{oracle}{suffix}.json")
    with open(out, "w") as f:
        json.dump(obs, f, separators=(",", ":"))
        f.write("\n")
    eng_out = open(os.path.join(rundir, f"{name}_0001.out")).read()
    cycles = None
    for line in eng_out.splitlines():
        if "TOTAL NUMBER OF CYCLES" in line:
            cycles = int(line.split(":")[1])
    prov = {
        "oracle": oracle,
        "scene": name,
        "seed": seed,
        "scene_file": os.path.relpath(scene_path, ROOT) if scene_path.startswith(ROOT) else scene_path,
        "scene_sha256": sha256_file(scene_path),
        "base_scene_sha256": sha256_file(os.path.abspath(a.scene)),
        "derive_command": derived_cmd,
        "tool": PINNED,
        "starter_deck_sha256": meta["deck_sha256"]["starter"],
        "engine_deck_sha256": meta["deck_sha256"]["engine"],
        "options": opts,
        "command": "python -I oracles/opencourant/pipeline.py " + " ".join(argv if argv is not None else sys.argv[1:]),
        "solver_commands": info["commands"],
        "threads": info["threads"],
        "starter_wall_s": round(info["starter_s"], 2),
        "engine_wall_s": round(info["engine_s"], 2),
        "pipeline_wall_s": round(time.time() - t_start, 2),
        "engine_cycles": cycles,
        "model_size": {"nodes": meta["n_nodes"], "elements": meta["n_elements"], "cohesive": meta["n_cohesive"]},
        "python": platform.python_version(),
        "numpy": __import__("numpy").__version__,
        "host": platform.platform(),
        "rundir": rundir,
    }
    with open(os.path.join(out_dir, f"provenance_{oracle}{suffix}.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"wrote {out} (engine {info['engine_s']:.1f} s, {cycles} cycles)")


if __name__ == "__main__":
    main()
