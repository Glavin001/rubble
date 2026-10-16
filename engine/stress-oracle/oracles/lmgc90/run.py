"""Runner: executes an exported LMGC90 deck with pylmgc90.chipy and stores raw results.

Reads ``<deck>/manifest.json`` + ``<deck>/DATBOX``; writes ``<deck>/raw.npz`` and
``<deck>/run.json`` (timings, solver version, step counts). No scene access.

Schedule: ``n_settle`` steps under gravity with the impactor invisible (the wall
settles on its footing and the cohesive joints take the self-weight), then the
impactor is made visible at its scene position with its scene velocity; that
instant is t = 0 of the reported time axis, followed by ``n_steps`` steps.

Usage: python -I run.py <deck_dir>
"""

import json
import os
import pathlib
import sys
import time

import numpy as np


def main(deck_dir):
    deck_dir = pathlib.Path(deck_dir).resolve()
    man = json.loads((deck_dir / "manifest.json").read_text())
    p = man["params"]
    os.chdir(deck_dir)  # LMGC90 reads ./DATBOX and writes ./OUTBOX
    for d in ("OUTBOX", "DISPLAY", "POSTPRO"):
        (deck_dir / d).mkdir(exist_ok=True)

    from pylmgc90 import chipy

    t_wall0 = time.time()
    chipy.Initialize()
    chipy.checkDirectories()
    chipy.utilities_DisableLogMes()
    chipy.SetDimension(3)
    chipy.TimeEvolution_SetTimeStep(man["dt"])
    chipy.Integrator_InitTheta(p["theta"])
    # Intact joints keep Coulomb friction on top of cohesion: mu(beta) = (1-beta) mu_d + beta mu_s
    # (mu_s = mu_d = joint friction), i.e. a Mohr-Coulomb envelope c + mu sigma.
    chipy.tact_behav_SetCZMwithInitialFriction(float(p.get("czm_friction_power", 0.0)))
    if p.get("detection", "f2f") == "sto":
        chipy.PRPRx_UseStoDetection(True, -1.0, p["f2f_tol"])
    else:
        chipy.PRPRx_ShrinkPolyrFaces(p["shrink"])
        chipy.PRPRx_UseCpF2fExplicitDetection(p["f2f_tol"])
    chipy.PRPRx_LowSizeArrayPolyr(10)
    chipy.nlgs_3D_DiagonalResolution()

    chipy.ReadBehaviours()
    chipy.ReadBodies()
    chipy.LoadBehaviours()
    chipy.ReadIniDof()
    chipy.ReadDrivenDof()
    chipy.LoadTactors()
    chipy.ReadIniVlocRloc()
    chipy.ComputeMass()

    nb = chipy.RBDY3_GetNbRBDY3()
    ball = int(man["impactor"]["body"])
    v0 = np.array(man["impactor"]["velocity"] + man["impactor"]["angular_velocity"], dtype=float)
    coor0 = np.array([chipy.RBDY3_GetBodyVector("Coor0", i)[:3] for i in range(1, nb + 1)])
    mass = np.array([chipy.RBDY3_GetMass(i) for i in range(1, nb + 1)])

    solver_type = "Stored_Delassus_Loops         "
    norm = p["nlgs_norm"]

    def step():
        chipy.IncrementStep()
        chipy.ComputeFext()
        chipy.ComputeBulk()
        chipy.ComputeFreeVelocity()
        chipy.SelectProxTactors()
        chipy.RecupRloc()
        # Same algorithm as chipy.ExSolver, unrolled to record iterations/convergence.
        chipy.nlgs_3D_SetCheckType(norm, p["nlgs_tol"], p["nlgs_relax"])
        chipy.nlgs_3D_ExPrep(solver_type)
        iters, conv = 0, 1
        for _ in range(p["gs_it2"]):
            if p.get("quick_scramble", False):
                chipy.nlgs_3D_QuickScrambleContactOrder()
            chipy.nlgs_3D_ExIter(p["gs_it1"])
            iters += p["gs_it1"]
            conv = chipy.nlgs_3D_AfterIterCheck()
            if conv in (0, -1):
                break
        chipy.nlgs_3D_ExPost()
        if conv == -1:
            raise RuntimeError("NLGS produced NaN reactions")
        chipy.UpdateTactBehav()
        chipy.StockRloc()
        chipy.ComputeDof()
        chipy.UpdateStep()
        return iters, conv == 0

    def interactions():
        it = chipy.getInteractions()
        return {
            "cd": np.array(it["icdbdy"], dtype=int), "an": np.array(it["ianbdy"], dtype=int),
            "law": np.array([x.decode() for x in it["behav"]]), "status": np.array([x.decode() for x in it["status"]]),
            "internal": np.array(it["internals"], dtype=float), "rl": np.array(it["rl"], dtype=float),
            "gap": np.array(it["gapTT"], dtype=float),
        }

    # Settling phase: impactor out of the simulation.
    chipy.RBDY3_SetInvisible(ball)
    t_settle0 = time.time()
    first = None
    settle_iters = []
    for k in range(man["n_settle"]):
        settle_iters.append(step())
        if k == 0:
            first = interactions()
    t_settle = time.time() - t_settle0
    if first is None:
        first = interactions()
    vel_settle = np.array([chipy.RBDY3_GetBodyVector("V____", i) for i in range(1, nb + 1)])
    ke_settle = 0.5 * float(np.sum(mass[:-1, None] * vel_settle[:-1, :3] ** 2))
    pos_settle = np.array([chipy.RBDY3_GetBodyVector("Coor_", i)[:3] for i in range(1, nb + 1)])

    # Release the impactor at its scene state.
    chipy.RBDY3_SetVisible(ball)
    xb = chipy.RBDY3_GetBodyVector("Coor_", ball)
    xbeg = chipy.RBDY3_GetBodyVector("Xbeg_", ball)
    shift = np.zeros(6)
    shift[:3] = np.array(man["impactor"]["position"]) - np.array(xb[:3])
    chipy.RBDY3_PutBodyVector("Xbeg_", ball, xbeg + shift)
    chipy.RBDY3_PutBodyVector("Vbeg_", ball, v0)
    chipy.RBDY3_PutBodyVector("V____", ball, v0)

    every = man["sample_every"]
    # Impactor-velocity probes are projected every step so the lo/hi envelope is exact.
    axes = np.array([pr["axis"] for pr in man["probe_axes"]], dtype=float).reshape(-1, 3)
    axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-300)
    pv0 = axes @ v0[:3]
    times, ball_v, lo, hi = [0.0], [pv0], [pv0], [pv0]
    run_lo, run_hi = np.full(len(axes), np.inf), np.full(len(axes), -np.inf)
    ball_vec = [v0[:3].tolist()]
    t_run0 = time.time()
    impact_iters = []
    for k in range(1, man["n_steps"] + 1):
        impact_iters.append(step())
        if k % 100 == 0:
            print("step %d/%d  %.1f s  nlgs %d" % (k, man["n_steps"], time.time() - t_run0, impact_iters[-1][0]), flush=True)
        vball = np.array(chipy.RBDY3_GetBodyVector("V____", ball)[:3])
        pv = axes @ vball
        run_lo, run_hi = np.minimum(run_lo, pv), np.maximum(run_hi, pv)
        if k % every == 0 or k == man["n_steps"]:
            times.append(k * man["dt"])
            ball_v.append(pv)
            lo.append(run_lo)
            hi.append(run_hi)
            ball_vec.append(vball.tolist())
            run_lo, run_hi = np.full(len(axes), np.inf), np.full(len(axes), -np.inf)
    t_run = time.time() - t_run0

    vel = np.array([chipy.RBDY3_GetBodyVector("V____", i) for i in range(1, nb + 1)])
    pos = np.array([chipy.RBDY3_GetBodyVector("Coor_", i)[:3] for i in range(1, nb + 1)])
    last = interactions()
    law_comment = {}
    for name in sorted(set(last["law"]) | set(first["law"])):
        try:
            law_comment[name] = chipy.tact_behav_GetLawInternalComment(chipy.tact_behav_GetTactBehavRankFromName(name))
        except Exception:  # comment retrieval is informational only
            law_comment[name] = ""
    chipy.Finalize()

    np.savez_compressed(
        deck_dir / "raw.npz",
        coor0=coor0, pos_settle=pos_settle, pos=pos, vel=vel, mass=mass,
        t=np.array(times), probe_v=np.array(ball_v).reshape(len(times), -1), probe_lo=np.array(lo).reshape(len(times), -1),
        probe_hi=np.array(hi).reshape(len(times), -1), ball_vec=np.array(ball_vec),
        **{"first_" + k: v for k, v in first.items()}, **{"last_" + k: v for k, v in last.items()},
    )
    import importlib.metadata as md
    info = {
        "solver_version": "LMGC90 user 2026.rc1 (pylmgc90 built from lmgc90_user_2026.rc1.zip, sha256 a3190240...; dist metadata %s)"
        % md.version("pylmgc90"),
        "wall_seconds": time.time() - t_wall0,
        "settle_seconds": t_settle,
        "impact_seconds": t_run,
        "n_bodies": nb,
        "n_settle": man["n_settle"],
        "n_steps": man["n_steps"],
        "ke_after_settle": ke_settle,
        "nlgs": {
            "norm": norm, "tol": p["nlgs_tol"], "max_iterations": p["gs_it1"] * p["gs_it2"],
            "settle_unconverged_steps": int(sum(1 for _, c in settle_iters if not c)),
            "impact_unconverged_steps": int(sum(1 for _, c in impact_iters if not c)),
            "impact_iterations_mean": float(np.mean([i for i, _ in impact_iters])) if impact_iters else 0.0,
            "impact_iterations_max": int(max([i for i, _ in impact_iters], default=0)),
            "impact_iterations": [i for i, _ in impact_iters],
            "impact_converged": [bool(c) for _, c in impact_iters],
        },
        "law_internal_comment": law_comment,
    }
    (deck_dir / "run.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(main(sys.argv[1]), indent=1))
