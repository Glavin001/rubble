"""Runner: starter + engine of the pinned OpenCourant build on an exported deck.

    python -I run.py <rundir> <scene_name> [--threads N] [--timeout S]

Writes starter/engine logs into <rundir>, converts the time-history file (T01) to CSV
with th_to_csv and the animation files (A###) to VTK with anim_to_vtk.  Fails loudly if
either program does not report a normal termination.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = "/home/user/oracle-env/opencourant/OpenCourant"


def env_for(root):
    env = dict(os.environ)
    env["OPENCOURANT_PATH"] = root
    env["LD_LIBRARY_PATH"] = ":".join([f"{root}/extlib/hm_reader/linux64", f"{root}/extlib/h3d/lib/linux64",
                                       env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    env["RAD_CFG_PATH"] = f"{root}/hm_cfg_files"
    env["KMP_STACKSIZE"] = "400m"
    env["OMP_STACKSIZE"] = "400m"
    return env


def opencourant_root():
    return os.environ.get("OPENCOURANT_PATH", DEFAULT_ROOT)


def run(rundir, name, threads=1, timeout=3600):
    rundir = os.path.abspath(rundir)
    root = opencourant_root()
    env = env_for(root)
    env["OMP_NUM_THREADS"] = str(threads)
    exe = f"{root}/exec"
    t0 = time.time()
    cmds = []
    with open(os.path.join(rundir, "starter.log"), "w") as log:
        cmd = [f"{exe}/starter_linux64_gf", "-i", f"{name}_0000.rad", "-np", "1", "-nt", str(threads)]
        cmds.append(" ".join(cmd))
        subprocess.run(cmd, cwd=rundir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    slog = open(os.path.join(rundir, "starter.log")).read()
    if "ERROR TERMINATION" in slog or "TERMINATION" not in slog:
        tail = "\n".join(slog.splitlines()[-40:])
        raise RuntimeError(f"starter failed, see {rundir}/starter.log and {name}_0000.out\n{tail}")
    t1 = time.time()
    with open(os.path.join(rundir, "engine.log"), "w") as log:
        cmd = [f"{exe}/engine_linux64_gf", "-i", f"{name}_0001.rad", "-nt", str(threads)]
        cmds.append(" ".join(cmd))
        subprocess.run(cmd, cwd=rundir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    elog = open(os.path.join(rundir, "engine.log")).read()
    if "NORMAL TERMINATION" not in elog:
        tail = "\n".join(elog.splitlines()[-40:])
        raise RuntimeError(f"engine failed, see {rundir}/engine.log\n{tail}")
    t2 = time.time()
    # post conversion
    th = os.path.join(rundir, f"{name}T01")
    if os.path.exists(th):
        cmd = [f"{exe}/th_to_csv_linux64_gf", th]
        cmds.append(" ".join(cmd))
        subprocess.run(cmd, cwd=rundir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    for a in sorted(glob.glob(os.path.join(rundir, f"{name}A[0-9][0-9][0-9]"))):
        with open(a + ".vtk", "w") as out:
            subprocess.run([f"{exe}/anim_to_vtk_linux64_gf", a], cwd=rundir, env=env, stdout=out,
                           stderr=subprocess.DEVNULL, check=True)
    cmds.append(f"{exe}/anim_to_vtk_linux64_gf <A-files>")
    return {"starter_s": t1 - t0, "engine_s": t2 - t1, "commands": cmds, "threads": threads}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("name")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args(argv)
    print(run(a.rundir, a.name, a.threads, a.timeout))


if __name__ == "__main__":
    main()
