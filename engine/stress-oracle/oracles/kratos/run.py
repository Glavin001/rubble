"""Runner: executes an exported Kratos DEM deck and stores raw results.

Reads ``<deck>/manifest.json`` and the Kratos input files; writes
``<deck>/raw.npz`` and ``<deck>/run.json``. No scene access.

Schedule: ``n_settle`` steps under gravity with the fixed (footing) particles
held and the impactor held at its scene position, then the impactor is
released with its scene velocity (t = 0 of the reported time axis) and
``n_steps`` steps follow. Bond (ParticleContactElement) failure ids are
refreshed by Kratos every step (ContactMeshOption) and accumulated at every sample, so a
bond that failed is remembered even if its particles later separate.

Usage: python -I run.py <deck_dir>
"""

import json
import os
import pathlib
import sys
import time

import numpy as np


def main(deck_dir):
    deck = pathlib.Path(deck_dir).resolve()
    man = json.loads((deck / "manifest.json").read_text())
    os.chdir(deck)

    import KratosMultiphysics as K
    import KratosMultiphysics.DEMApplication as DEM
    from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

    K.Logger.GetDefaultOutput().SetSeverity(K.Logger.Severity.WARNING)

    class Stage(DEMAnalysisStage):
        def GetMainPath(self):
            return str(deck)

        def IsTimeToPrintPostProcess(self):
            # Bond (continuum contact) elements are refreshed every step whenever
            # ContactMeshOption is on; IS_TIME_TO_PRINT would additionally touch
            # discontinuum contact elements that were never created (segfault).
            return False

        def PrintResultsForGid(self, time):
            pass

        def RunAnalytics(self, time):
            pass

        def OutputSolutionStep(self):
            pass

    with open(deck / "ProjectParametersDEM.json") as f:
        params = K.Parameters(f.read())
    model = K.Model()
    stage = Stage(model, params)
    t0 = time.time()
    stage.Initialize()
    t_init = time.time() - t0
    spheres = stage.spheres_model_part
    contact = stage.contact_model_part

    def fix(node, value):
        for v in (K.VELOCITY_X, K.VELOCITY_Y, K.VELOCITY_Z, K.ANGULAR_VELOCITY_X, K.ANGULAR_VELOCITY_Y, K.ANGULAR_VELOCITY_Z):
            node.Fix(v)
        node.SetSolutionStepValue(K.VELOCITY, value)
        node.SetSolutionStepValue(K.ANGULAR_VELOCITY, [0.0, 0.0, 0.0])

    for nid in man["fixed_nodes"]:
        fix(spheres.GetNode(nid), [0.0, 0.0, 0.0])
    ball = spheres.GetNode(man["impactor"]["node"])
    fix(ball, [0.0, 0.0, 0.0])

    n_wall = len(man["particle_chunk"])
    coor0 = np.array([[n.X0, n.Y0, n.Z0] for n in (spheres.GetNode(i + 1) for i in range(n_wall))])

    def step():
        stage.time = stage._AdvanceTime()
        stage.InitializeSolutionStep()
        stage._GetSolver().Predict()
        stage._GetSolver().SolveSolutionStep()
        stage.FinalizeSolutionStep()

    bond_failed = {}

    def census():
        stage._GetSolver().PrepareContactElementsForPrinting()
        out = {}
        for e in contact.Elements:
            g = e.GetGeometry()
            a, b = g[0].Id, g[1].Id
            if a > n_wall or b > n_wall:
                continue
            key = (min(a, b), max(a, b))
            f = int(round(e.GetValue(DEM.CONTACT_FAILURE)))
            out[key] = f
            if f != 0:
                bond_failed[key] = bond_failed.get(key) or f
            else:
                bond_failed.setdefault(key, 0)
        return out

    t1 = time.time()
    for k in range(man["n_settle"]):
        step()
    t_settle = time.time() - t1
    first = census()
    vel_settle = np.array([list(spheres.GetNode(i + 1).GetSolutionStepValue(K.VELOCITY)) for i in range(n_wall)])
    mass = np.array([spheres.GetNode(i + 1).GetSolutionStepValue(K.NODAL_MASS) for i in range(n_wall)])
    ke_settle = float(0.5 * np.sum(mass[:, None] * vel_settle ** 2))

    # Release the impactor.
    for v in (K.VELOCITY_X, K.VELOCITY_Y, K.VELOCITY_Z, K.ANGULAR_VELOCITY_X, K.ANGULAR_VELOCITY_Y, K.ANGULAR_VELOCITY_Z):
        ball.Free(v)
    v0 = np.array(man["impactor"]["velocity"], dtype=float)
    ball.SetSolutionStepValue(K.VELOCITY, list(v0))
    ball.SetSolutionStepValue(K.ANGULAR_VELOCITY, list(man["impactor"]["angular_velocity"]))
    ball_pos0 = np.array([ball.X, ball.Y, ball.Z])

    axes = np.array([p["axis"] for p in man["probe_axes"]], dtype=float).reshape(-1, 3)
    axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-300)
    pv0 = axes @ v0
    times, pv, lo, hi = [0.0], [pv0], [pv0], [pv0]
    run_lo, run_hi = np.full(len(axes), np.inf), np.full(len(axes), -np.inf)
    every = man["sample_every"]
    t2 = time.time()
    for k in range(1, man["n_steps"] + 1):
        step()
        vb = np.array(list(ball.GetSolutionStepValue(K.VELOCITY)))
        p = axes @ vb
        run_lo, run_hi = np.minimum(run_lo, p), np.maximum(run_hi, p)
        if k % every == 0 or k == man["n_steps"]:
            times.append(k * man["dt"])
            pv.append(p)
            lo.append(run_lo)
            hi.append(run_hi)
            run_lo, run_hi = np.full(len(axes), np.inf), np.full(len(axes), -np.inf)
            census()
        if k % 2000 == 0:
            print("step %d/%d  %.1f s" % (k, man["n_steps"], time.time() - t2), flush=True)
    t_run = time.time() - t2
    last = census()

    pos = np.array([[n.X, n.Y, n.Z] for n in (spheres.GetNode(i + 1) for i in range(n_wall))])
    vel = np.array([list(spheres.GetNode(i + 1).GetSolutionStepValue(K.VELOCITY)) for i in range(n_wall)])
    keys = sorted(bond_failed)
    np.savez_compressed(
        deck / "raw.npz", coor0=coor0, pos=pos, vel=vel, mass=mass,
        bond_nodes=np.array(keys, dtype=int).reshape(-1, 2),
        bond_failed=np.array([bond_failed[k] for k in keys], dtype=int),
        bond_first=np.array([first.get(k, -1) for k in keys], dtype=int),
        bond_last_seen=np.array([1 if k in last else 0 for k in keys], dtype=int),
        t=np.array(times), probe_v=np.array(pv).reshape(len(times), -1),
        probe_lo=np.array(lo).reshape(len(times), -1), probe_hi=np.array(hi).reshape(len(times), -1),
        ball_pos0=ball_pos0, ball_pos=np.array([ball.X, ball.Y, ball.Z]),
    )
    import importlib.metadata as md
    info = {
        "solver_version": "KratosMultiphysics %s + KratosDEMApplication %s (PyPI wheels)" % (
            md.version("KratosMultiphysics"), md.version("KratosDEMApplication")),
        "wall_seconds": time.time() - t0, "init_seconds": t_init, "settle_seconds": t_settle, "impact_seconds": t_run,
        "n_particles": n_wall + 1, "n_bonds_initial": len(first),
        "n_bonds_initially_failed": int(sum(1 for v in first.values() if v != 0)),
        "ke_after_settle": ke_settle, "wall_mass": float(mass.sum()),
        "n_settle": man["n_settle"], "n_steps": man["n_steps"], "dt": man["dt"],
    }
    stage.Finalize()
    (deck / "run.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(main(sys.argv[1]), indent=1))
