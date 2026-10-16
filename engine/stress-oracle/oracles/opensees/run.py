"""Run an exported OpenSees frame model; record raw responses (no post-processing).

`run_model(model)` builds the domain (export.build), computes the intact eigenvalues,
solves the static state (gravity prestress: gravity + the t = 0 loads), then integrates
in time with Newmark average acceleration (gamma = 1/2, beta = 1/4) on a uniform grid
of `analysis.dt` (event times inserted exactly). Scripted removals are processed at the
start of the step that begins at the event time, like the reference solver.

Raw output (numpy arrays, NaN where an element/node no longer exists):
`t[k]`, `ele_force[k, e, 12]` (global resisting forces, `ops.eleForce`),
`disp[k, n, 6]`, `vel[k, n, 6]` and `reaction[k, n, 6]` (held nodes only), plus
`eigen_hz`, replacement loads and timings.
"""

from __future__ import annotations

import math
import time

import numpy as np

import export


def _analysis(ops, kind):
    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandSPD")
    ops.test("NormDispIncr", 1e-12, 10)
    ops.algorithm("Linear")
    if kind == "static":
        ops.integrator("LoadControl", 0.0)
        ops.analysis("Static")
    else:
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")


def time_grid(model):
    a = model["analysis"]
    dt, duration = a["dt"], a["duration"]
    n = int(round(duration / dt))
    grid = [k * dt for k in range(n + 1)]
    if grid[-1] < duration - 1e-12:
        grid.append(duration)
    for ev in model["events"]:
        if 0.0 < ev["time"] < duration and min(abs(ev["time"] - g) for g in grid) > 1e-9 * dt:
            grid.append(ev["time"])
            if ev["duration"] > 0 and ev["time"] + ev["duration"] < duration:
                grid.append(ev["time"] + ev["duration"])
    return sorted(grid)


def run_model(model, ops, log=print):
    t_start = time.perf_counter()
    export.build(model, ops)
    nodes = [nd["tag"] for nd in model["nodes"]]
    elems = [el["tag"] for el in model["elements"]]
    held = [nd["tag"] for nd in model["nodes"] if any(nd["fix"])]
    n_index = {t: i for i, t in enumerate(nodes)}
    e_index = {t: i for i, t in enumerate(elems)}

    # Intact eigenvalues (before any loading; linear elastic, so state-independent).
    n_modes = min(model["analysis"]["eigen_modes"], 6 * len(nodes) - 6 * len(held) - 1)
    eigen_hz = []
    if n_modes > 0:
        # ARPACK on the constrained system (fullGenLapack is slow and returns spurious
        # zero modes for the held DOFs).
        lam = ops.eigen("-genBandArpack", n_modes)
        eigen_hz = [math.sqrt(max(x, 0.0)) / (2.0 * math.pi) for x in lam]
    t_built = time.perf_counter()

    if model["analysis"]["static_first"]:
        _analysis(ops, "static")
        ops.setTime(0.0)
        if ops.analyze(1) != 0:
            raise RuntimeError("static (prestress) analysis failed")
    t_static = time.perf_counter()
    _analysis(ops, "transient")
    ops.setTime(0.0)

    grid = time_grid(model)
    K, NE, NN = len(grid), len(elems), len(nodes)
    t_arr = np.array(grid)
    ele_force = np.full((K, NE, 12), np.nan)
    disp = np.full((K, NN, 6), np.nan)
    vel = np.full((K, NN, 6), np.nan)
    reaction = np.full((K, NN, 6), np.nan)
    alive_e = set(elems)
    alive_n = set(nodes)
    want_reactions = any(p["type"] == "reaction" for p in model["probes"])

    def record(k):
        for e in alive_e:
            ele_force[k, e_index[e]] = ops.eleForce(e)
        for n in alive_n:
            i = n_index[n]
            disp[k, i] = ops.nodeDisp(n)
            vel[k, i] = ops.nodeVel(n)
        if want_reactions:
            ops.reactions()
            for n in held:
                if n in alive_n:
                    reaction[k, n_index[n]] = ops.nodeReaction(n)

    replacements = []
    events_done = [False] * len(model["events"])
    next_tag = 1 + max([p["tag"] for p in model["patterns"]] + [0])
    record(0)
    for k in range(1, K):
        t_cur = grid[k - 1]
        for ei, ev in enumerate(model["events"]):
            if events_done[ei] or t_cur + 1e-12 < ev["time"]:
                continue
            events_done[ei] = True
            # Interface forces the removed elements exert on the kept nodes (now).
            loads = {}
            for itf in ev["interface"]:
                f = ops.eleForce(itf["element"])
                r = f[6:12] if itf["kept_end"] == "j" else f[0:6]
                acc = loads.setdefault(itf["kept_node"], [0.0] * 6)
                for d in range(6):
                    acc[d] -= r[d]
            for p in model["patterns"]:
                if p["removed_by_event"] == ei:
                    ops.remove("loadPattern", p["tag"])
            for e in ev["removed_elements"]:
                ops.remove("ele", e)
                alive_e.discard(e)
            for n in ev["removed_nodes"]:
                ops.remove("node", n)
                alive_n.discard(n)
            if ev["duration"] > 0:
                ops.timeSeries("Path", next_tag, "-time", ev["time"], ev["time"] + ev["duration"], "-values", 1.0, 0.0)
                ops.pattern("Plain", next_tag, next_tag)
                for n, v in sorted(loads.items()):
                    if n in alive_n:
                        ops.load(n, *v)
                next_tag += 1
            replacements.append({"event": ei, "time": t_cur, "duration": ev["duration"],
                                 "loads": {str(n): v for n, v in sorted(loads.items())}})
            log(f"  event {ei} at t={t_cur:.6f}: removed {len(ev['removed_elements'])} elements, "
                f"{len(ev['removed_nodes'])} nodes; {len(loads)} replacement loads")
        if ops.analyze(1, grid[k] - t_cur) != 0:
            raise RuntimeError(f"transient step failed at t={grid[k]:.6g}")
        record(k)
    t_end = time.perf_counter()
    return {
        "t": t_arr, "ele_force": ele_force, "disp": disp, "vel": vel, "reaction": reaction,
        "node_tags": np.array(nodes), "ele_tags": np.array(elems),
        "eigen_hz": eigen_hz, "replacements": replacements,
        "timing": {"build_and_eigen_s": t_built - t_start, "static_s": t_static - t_built,
                   "transient_s": t_end - t_static, "steps": K - 1},
    }
