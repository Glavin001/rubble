"""OpenSeesPy oracle pipeline: scene -> deck (export) -> run -> observation + provenance.

    python -I oracles/opensees/pipeline.py scenes/<scene>.json [--seed N] [--work DIR] [--golden DIR]

Writes `golden/<scene>/opensees[_seedN].json` and `golden/<scene>/provenance_opensees[_seedN].json`.
The deck (model JSON) and the raw response arrays go to `--work` (default
`$OPENSEES_ORACLE_WORK` or `oracle-runs/opensees/<scene>[_seedN]/` next to the PhysX
checkout). For `--seed N` different from the scene's
seed, the scene is re-derived with `stress-ref with-seed` (Weibull strengths) first.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # destruction/stress-oracle
sys.path.insert(0, HERE)  # -I drops the script directory from sys.path
# Scratch (deck + raw arrays): $OPENSEES_ORACLE_WORK, else oracle-runs/opensees next to the
# PhysX checkout (/home/user/oracle-runs/opensees for /home/user/PhysX).
DEFAULT_WORK = os.environ.get("OPENSEES_ORACLE_WORK") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), "oracle-runs", "opensees")

import numpy as np  # noqa: E402

import export  # noqa: E402
import observe  # noqa: E402
import run  # noqa: E402

PINNED_WHEELS = {
    "openseespy-3.8.0.0-py3-none-any.whl": "ce6a43503b737025f763a87bebacb1cc2fbd82a156982647e9e014a11b85fdbd",
    "openseespylinux-3.8.0.0-py3-none-any.whl": "2307e40213f1d7128a5a5151d5c8beee8244b8b988528a58f5e44394a045ba64",
}
PINNED_SO_SHA256 = "b1bfa97ceb8f0f5d3c3c49eb498ac4e9dc57f86971f0f54c03074968a0f830a3"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def opensees_identity():
    import importlib.metadata as md

    import openseespy.opensees as ops
    import openseespylinux

    so = os.path.join(os.path.dirname(openseespylinux.__file__), "opensees.so")
    so_hash = sha256_file(so)
    return ops, {
        "openseespy": md.version("openseespy"),
        "openseespylinux": md.version("openseespylinux"),
        "opensees_core": ops.version(),
        "opensees_so_sha256": so_hash,
        "opensees_so_matches_pinned_wheel": so_hash == PINNED_SO_SHA256,
        "wheels_sha256": PINNED_WHEELS,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--golden", default=os.path.join(ROOT, "golden"))
    ap.add_argument("--stress-ref", default=os.path.join(ROOT, "target", "release", "stress-ref"))
    args = ap.parse_args(argv)
    wall0 = time.perf_counter()

    scene_path = os.path.abspath(args.scene)
    with open(scene_path, "rb") as f:
        scene_bytes = f.read()
    scene = json.loads(scene_bytes)
    seed = scene["sim"].get("seed", 0) if args.seed is None else args.seed
    suffix = "" if seed == 0 else f"_seed{seed}"
    work = args.work or os.path.join(DEFAULT_WORK, scene["name"] + suffix)
    os.makedirs(work, exist_ok=True)

    provenance_scene = {"path": scene_path, "sha256": sha256_bytes(scene_bytes)}
    if seed != scene["sim"].get("seed", 0):
        derived = os.path.join(work, "scene.json")
        subprocess.run([args.stress_ref, "with-seed", scene_path, str(seed), derived], check=True)
        with open(derived, "rb") as f:
            b = f.read()
        scene = json.loads(b)
        provenance_scene["derived_with_seed"] = {"path": derived, "sha256": sha256_bytes(b),
                                                 "command": [args.stress_ref, "with-seed", scene_path, str(seed), derived]}

    ops, ident = opensees_identity()
    version = f"OpenSeesPy {ident['openseespy']} (OpenSees {ident['opensees_core']}, openseespylinux {ident['openseespylinux']})"

    # 1. Export (pure function of the scene) -> deck.
    model = export.export_model(scene)
    deck = json.dumps(model, indent=1, sort_keys=True).encode()
    deck_path = os.path.join(work, "model.json")
    with open(deck_path, "wb") as f:
        f.write(deck)
    print(f"{scene['name']}: {len(model['nodes'])} nodes, {len(model['elements'])} elements, "
          f"dt={model['analysis']['dt']:.3g} s to {model['analysis']['duration']} s")

    # 2. Run.
    raw = run.run_model(model, ops)
    np.savez(os.path.join(work, "raw.npz"), **{k: v for k, v in raw.items() if isinstance(v, np.ndarray)})
    with open(os.path.join(work, "raw_meta.json"), "w") as f:
        json.dump({k: v for k, v in raw.items() if not isinstance(v, np.ndarray)}, f, indent=1)
    print(f"  timing {raw['timing']}")

    # 3. Observe.
    a = model["analysis"]
    notes = [
        "frame model: one 6-DOF node per level-0 chunk (lumped box mass + rotary inertia), one elasticBeamColumn "
        "(Euler-Bernoulli, no shear deformation) per bond between chunk centres; E, G of the bond material, A = bond area, "
        "I about t1/t2 = derived.i_t1/i_t2, J = derived.torsion_constant",
        f"time integration: static prestress (gravity + t=0 loads) then Newmark average acceleration, dt = {a['dt']:.4g} s "
        f"(sample_interval / {a['substeps']}); probes sampled every {a['sample_interval']:.4g} s with lo/hi envelopes over the substeps",
        "damping: per-element stiffness-proportional beta_e = 2 zeta / sqrt(kn / m_red), i.e. zeta at each bond's own axial "
        "frequency like the reference solver's dashpots (much less than zeta for the slow global modes); reported forces "
        "exclude the damping force",
        "elastic analysis: no damage, nothing detaches (collapse = false by construction); any_bond_cracked = any_bond_broken = some bond's "
        "joint.rs failure index (strengths x Weibull, at the bond centroid) reached 1 at some analysis step",
        "removed chunks are reported with removed = true and zero velocity/displacement",
    ]
    if scene.get("events"):
        notes.append("remove_chunks: elements touching removed chunks and the removed nodes (mass, loads) are deleted at the "
                     "event time; the forces the deleted elements exerted on the kept nodes (measured at that instant) are "
                     "re-applied as nodal loads ramping linearly to zero over `duration` (none when duration = 0)")
    obs = observe.observe(scene, model, raw, version, seed, notes)
    wall = time.perf_counter() - wall0
    obs["values"]["wall_seconds"] = wall
    obs["values"]["analysis_steps"] = float(raw["timing"]["steps"])

    out_dir = os.path.join(args.golden, scene["name"])
    os.makedirs(out_dir, exist_ok=True)
    obs_path = os.path.join(out_dir, f"opensees{suffix}.json")
    with open(obs_path, "w") as f:
        json.dump(obs, f, separators=(",", ":"))
        f.write("\n")
    prov = {
        "tool": "opensees",
        "solver_version": version,
        "versions": ident,
        "license": "OpenSeesPy: free for research/education/internal use, NOT OSI open source (UC Berkeley / OpenSeesPy license); commercial redistribution needs a license",
        "scene": provenance_scene,
        "seed": seed,
        "deck": {"path": deck_path, "sha256": sha256_bytes(deck), "format": model["format"]},
        "command": [sys.executable, "-I"] + [os.path.abspath(sys.argv[0])] + sys.argv[1:],
        "cwd": os.getcwd(),
        "wall_seconds": wall,
        "timing": raw["timing"],
        "host": {"platform": platform.platform(), "cpus": os.cpu_count()},
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "observation": {"path": obs_path, "sha256": sha256_file(obs_path)},
    }
    with open(os.path.join(out_dir, f"provenance_opensees{suffix}.json"), "w") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    print(f"  wrote {obs_path} ({wall:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
