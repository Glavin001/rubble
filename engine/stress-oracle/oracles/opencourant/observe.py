"""Post-processor: OpenCourant outputs -> stress-observation/1 JSON.

    python -I observe.py <scene.json> <rundir> <out.json> [--seed N] [--version STR]

Reads only the scene, the exporter's map.json, the time-history CSV (th_to_csv of T01)
and the last animation state (anim_to_vtk).  Probes:
  * chunk_displacement / chunk_velocity: lumped-mass weighted mean of the chunk's nodes
    (/TH/NODE DX..VZ);
  * reaction: sum of /TH/NODE REACX..REACZ (accumulated reaction impulse) over the nodes of
    the selected supported chunks, differentiated over each sampling interval;
  * impactor_velocity / impactor_position: /TH/RBODY of the impactor's rigid body;
  * section_force: the crossing bonds' cohesive elements (/TH/BRIC mean stress, global
    frame) integrated over each element face: force on the +normal side = -/+ traction x
    area, moments about the probe point from element centroids (within-element moment
    neglected).
Final chunk states come from the last animation frame: mean nodal velocity and
displacement (lumped-mass weights); a bond is intact while at least one of its cohesive
elements is not deleted; `detached` = not connected through intact bonds to a supported
chunk (or to the heaviest piece of an unsupported body).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

SOLVER = "opencourant"
VERSION = ("OpenCourant latest-20261006 (OpenRadioss fork, commit 33e685176cccf0c539a3ce07aa2096985a284e2a, "
           "OpenCourant_linux64.zip sha256 9d67531d...0080081)")


def unit_vec(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def read_th(path):
    with open(path) as f:
        header = next(csv.reader(f))
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return [h.strip() for h in header], data


def th_columns(header, meta):
    """Map (kind, object id) -> {var: column}, using the TH group titles."""
    titles = {"node": None}
    out = {}
    # group titles used by the exporter, in a fixed mapping to the meta entries
    groups = []
    for g in meta["th"]:
        groups.append(g)
    name_of = {"probe nodes": ("node", ["DX", "DY", "DZ", "VX", "VY", "VZ"]),
               "reaction nodes": ("reac", ["REACX", "REACY", "REACZ"]),
               "impactors": ("rbody", ["X", "Y", "Z", "VX", "VY", "VZ"]),
               "section cohesive elements": ("bric", ["OFF", "SX", "SY", "SZ", "SXY", "SYZ", "SXZ"])}
    counters = defaultdict(int)
    for col, h in enumerate(header):
        parts = re.split(r"\s{2,}", h)
        if len(parts) >= 3 and parts[0] in name_of:
            kind, vars_ = name_of[parts[0]]
            oid = int(parts[1].split()[0])
            k = (kind, oid)
            i = counters[k]
            counters[k] += 1
            if i < len(vars_):
                out.setdefault(k, {})[vars_[i]] = col
        elif len(parts) == 2 and parts[1] in ("IE", "KE") and parts[0].startswith("cohesive"):
            out.setdefault(("part", parts[0]), {})[parts[1]] = col
    return out


def read_vtk(path):
    """Minimal legacy-ASCII reader for anim_to_vtk output."""
    with open(path) as f:
        tok = f.read().split("\n")
    i = 0
    res = {"point": {}, "cell": {}}
    mode = None
    n = 0
    while i < len(tok):
        line = tok[i].strip()
        if line.startswith("TIME"):
            res["time"] = float(tok[i + 1])
            i += 2
            continue
        if line.startswith("POINTS"):
            n = int(line.split()[1])
            res["npoints"] = n
            i += 1 + n
            continue
        if line.startswith("CELLS"):
            i += 1 + int(line.split()[1])
            continue
        if line.startswith("CELL_TYPES"):
            i += 1 + int(line.split()[1])
            continue
        if line.startswith("POINT_DATA"):
            mode, n = "point", int(line.split()[1])
            i += 1
            continue
        if line.startswith("CELL_DATA"):
            mode, n = "cell", int(line.split()[1])
            i += 1
            continue
        if line.startswith("SCALARS"):
            name = line.split()[1]
            vals = np.array([float(x) for x in tok[i + 2:i + 2 + n]])
            res[mode][name] = vals
            i += 2 + n
            continue
        if line.startswith("VECTORS"):
            name = line.split()[1]
            vals = np.array([[float(x) for x in t.split()] for t in tok[i + 1:i + 1 + n]])
            res[mode][name] = vals
            i += 1 + n
            continue
        i += 1
    return res


def observe(scene, rundir, seed=None, version=VERSION, extra_notes=()):
    meta = json.load(open(os.path.join(rundir, "map.json")))
    name = scene["name"]
    u = meta["units"]
    lm, fn, en, sp = u["length_to_m"], u["force_to_N"], u["energy_to_J"], u["stress_to_Pa"]
    header, data = read_th(os.path.join(rundir, f"{name}T01.csv"))
    t = data[:, 0]
    cols = th_columns(header, meta)
    obs = {"format": "stress-observation/1", "scene": name, "solver": SOLVER, "solver_version": version,
           "seed": int(scene["sim"].get("seed", 0) if seed is None else seed), "end_time": float(t[-1]),
           "probes": {}, "bodies": {}, "flags": {}, "values": {}, "notes": []}

    # ------------------------------------------------------------- probes
    def node_series(nodes, weights, var):
        acc = np.zeros_like(t)
        wsum = 0.0
        for n_, w in zip(nodes, weights):
            acc += w * data[:, cols[("node", n_)][var]]
            wsum += w
        return acc / wsum

    for p in scene.get("probes", []):
        pm = meta["probes"].get(p["name"])
        if pm is None:
            continue
        typ = pm["type"]
        if typ in ("chunk_displacement", "chunk_velocity"):
            ch = meta["bodies"][pm["body"]]["chunks"][pm["chunk"]]
            a = unit_vec(pm["axis"])
            pre = "D" if typ == "chunk_displacement" else "V"
            v = sum(a[i] * node_series(ch["nodes"], ch["weights"], pre + "XYZ"[i]) for i in range(3) if a[i] != 0)
            v = v * lm
        elif typ == "reaction":
            a = unit_vec(pm["axis"])
            v = np.zeros_like(t)
            for n_ in pm["nodes"]:
                c = cols[("reac", n_)]
                for i in range(3):
                    if a[i] != 0:
                        v += a[i] * data[:, c["REAC" + "XYZ"[i]]]
            # /TH/NODE REAC* is the reaction impulse accumulated since t = 0 (BCS1TH adds
            # -A m dt every cycle); the force is its rate over each sampling interval.
            imp = v * fn
            v = np.zeros_like(imp)
            v[1:] = np.diff(imp) / np.maximum(np.diff(t), 1e-300)
            v[0] = v[1] if len(v) > 1 else 0.0
        elif typ in ("impactor_velocity", "impactor_position"):
            a = unit_vec(pm["axis"])
            c = cols[("rbody", pm["rbody"])]
            pre = "V" if typ == "impactor_velocity" else ""
            v = sum(a[i] * data[:, c[pre + "XYZ"[i]]] for i in range(3) if a[i] != 0) * lm
        elif typ == "section_force":
            f = np.zeros((len(t), 3))
            mom = np.zeros((len(t), 3))
            pt = np.asarray(pm["point"], float)
            for e in pm["elements"]:
                c = cols[("bric", e["eid"])]
                sx, sy, sz = (data[:, c[k]] for k in ("SX", "SY", "SZ"))
                sxy, syz, sxz = (data[:, c[k]] for k in ("SXY", "SYZ", "SXZ"))
                n_ = np.asarray(e["normal"])
                tr = np.stack([sx * n_[0] + sxy * n_[1] + sxz * n_[2],
                               sxy * n_[0] + sy * n_[1] + syz * n_[2],
                               sxz * n_[0] + syz * n_[1] + sz * n_[2]], axis=1) * sp
                fe = (-1.0 if e["plus"] == "b" else 1.0) * tr * e["area"]
                f += fe
                mom += np.cross(np.asarray(e["centroid"]) - pt, fe)
            comp = pm["component"]
            if comp == "normal":
                v = -f @ unit_vec(pm["normal"])
            elif "force" in comp:
                v = f @ unit_vec(comp["force"])
            else:
                v = mom @ unit_vec(comp["moment"])
        else:
            continue
        obs["probes"][p["name"]] = {"t": [float(x) for x in t], "v": [float(x) for x in v]}

    # ------------------------------------------------------------- final state
    vtks = sorted(glob.glob(os.path.join(rundir, f"{name}A[0-9][0-9][0-9].vtk")))
    last = read_vtk(vtks[-1])
    if abs(last["time"] - scene["sim"]["duration"]) > 0.02 * scene["sim"]["duration"]:
        obs["notes"].append(f"last animation frame at t = {last['time']:.6g} s (duration {scene['sim']['duration']})")
    nid = last["point"]["NODE_ID"].astype(int)
    vel = dict(zip(nid, last["point"]["Velocity"]))
    disp = dict(zip(nid, last["point"]["Displacement"]))
    eid = last["cell"]["ELEMENT_ID"].astype(int)
    alive = dict(zip(eid, last["cell"]["EROSION_STATUS"].astype(int)))
    broken_total = 0
    eroded_total = 0
    collapse = False
    fragments = 0
    for body in scene["bodies"]:
        bm = meta["bodies"][body["name"]]
        nch = len(body["chunks"])
        mass = []
        for c in body["chunks"]:
            h = c["half_extents"]
            mass.append(8 * h[0] * h[1] * h[2] * scene["materials"][c["material"]]["density"])
        parent = list(range(nch))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for b, els in zip(body["bonds"], bm["bonds"]):
            n_alive = sum(1 for e in els if alive.get(e, 1) == 1)
            eroded_total += len(els) - n_alive
            if n_alive > 0:
                ra, rb = find(b["a"]), find(b["b"])
                if ra != rb:
                    parent[ra] = rb
            else:
                broken_total += 1
        comps = defaultdict(list)
        for i in range(nch):
            comps[find(i)].append(i)
        fragments += len(comps)
        supported = set(bm["supported"])
        if supported:
            main = {r for r, mem in comps.items() if any(i in supported for i in mem)}
        else:
            heaviest = max(comps, key=lambda r: sum(mass[i] for i in comps[r]))
            main = {heaviest}
        frag_id = {r: k for k, r in enumerate(sorted(comps))}
        states = []
        for i, ch in enumerate(bm["chunks"]):
            w = np.asarray(ch["weights"], float)
            vv = np.array([vel[n_] for n_ in ch["nodes"]])
            dd = np.array([disp[n_] for n_ in ch["nodes"]])
            v = (w[:, None] * vv).sum(0) / w.sum() * lm
            d = (w[:, None] * dd).sum(0) / w.sum() * lm
            det = find(i) not in main
            if det and supported:
                collapse = True
            states.append({"detached": bool(det), "removed": False, "fragment": frag_id[find(i)],
                           "velocity": [float(x) for x in v], "displacement": [float(x) for x in d]})
        obs["bodies"][body["name"]] = states
    obs["flags"]["any_bond_broken"] = broken_total > 0
    obs["flags"]["collapse"] = collapse
    obs["values"]["broken_bonds"] = float(broken_total)
    obs["values"]["deleted_cohesive_elements"] = float(eroded_total)
    obs["values"]["fragments"] = float(fragments)
    ie_cols = [c["IE"] for k, c in cols.items() if k[0] == "part" and "IE" in c]
    if ie_cols:
        ie = sum(data[:, c] for c in ie_cols) * en
        obs["values"]["bond_dissipation"] = float(ie[-1])
        obs["values"]["cohesive_internal_energy_peak"] = float(ie.max())
    gl = {h: i for i, h in enumerate(header)}
    for key, label in (("INTERNAL ENERGY", "total_internal_energy"), ("KINETIC ENERGY", "total_kinetic_energy"),
                       ("EXTERNAL WORK", "external_work"), ("CONTACT ENERGY", "contact_energy")):
        if key in gl:
            obs["values"][label] = float(data[-1, gl[key]] * en)
    obs["values"]["elements"] = float(meta["n_elements"])
    obs["values"]["cohesive_elements"] = float(meta["n_cohesive"])
    obs["notes"] = list(dict.fromkeys(MODEL_NOTES + meta.get("notes", []) + list(extra_notes) + obs["notes"]))
    return obs


MODEL_NOTES = [
    "continuum oracle: each chunk is a block of linear-elastic hexahedra (/PROP/SOLID Isolid=24, /MAT/LAW1)",
    "each bond is a layer of /PROP/TYPE43 cohesive elements with /MAT/LAW169 (ARUP adhesive): tension cutoff "
    "f_t*weibull, Mohr-Coulomb shear c*weibull + friction*compression (SHT_SL), linear softening with the scene's "
    "G_f (tension, shear); no crushing (compression elastic) and no shear cap",
    "the cohesive layer has a finite thickness (bonded chunk faces are pulled in by t_c/2, layer modulus matched "
    "to the bond's series compliance); a gap contact (TYPE7, gap ~t_c) between all chunks closes cracked joints "
    "at their original position, with the joint friction",
    "bond_dissipation = final internal energy of all cohesive layers (includes elastic energy still stored in "
    "intact layers)",
    "a bond counts as broken when all of its cohesive elements are deleted; detached = not connected to a "
    "supported chunk through intact bonds",
    "no rate effects (DIF), no bond damping (only the elements' numerical damping)",
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("rundir")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    scene = json.load(open(a.scene))
    obs = observe(scene, a.rundir, a.seed)
    with open(a.out, "w") as f:
        json.dump(obs, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
