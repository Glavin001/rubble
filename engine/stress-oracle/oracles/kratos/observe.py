"""Post-processor: Kratos DEM raw results -> stress-observation/1 JSON.

Reads ``<deck>/manifest.json``, ``<deck>/raw.npz``, ``<deck>/run.json``; never the
scene. A scene bond (mortar joint) is *intact* while at least one of the particle
bonds crossing it has never failed (CONTACT_FAILURE == 0 at every census);
bonds between two fixed chunks count as intact. Chunks connected to a fixed chunk
through intact joints form the main piece; all others are ``detached``.
``velocity``/``displacement`` = mass-weighted mean over the chunk's particles.

Usage: python -I observe.py <deck_dir> <out.json>
"""

import json
import pathlib
import sys

import numpy as np


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(n)]


def observe(deck_dir):
    deck = pathlib.Path(deck_dir)
    man = json.loads((deck / "manifest.json").read_text())
    run = json.loads((deck / "run.json").read_text())
    raw = np.load(deck / "raw.npz")
    pc = np.array(man["particle_chunk"])
    n = man["n_chunks"]
    fixed = [s == "fixed" for s in man["chunk_support"]]
    notes = []

    joint_bonds = {}  # (chunk a, chunk b) -> [n intact, n total]
    intra_failed = 0
    for (na, nb), f in zip(raw["bond_nodes"], raw["bond_failed"]):
        ca, cb = int(pc[na - 1]), int(pc[nb - 1])
        if ca == cb:
            intra_failed += int(f != 0)
            continue
        key = (min(ca, cb), max(ca, cb))
        st = joint_bonds.setdefault(key, [0, 0])
        st[1] += 1
        st[0] += int(f == 0)
    if intra_failed:
        notes.append("WARNING: %d intra-chunk (unbreakable) particle bonds report failure" % intra_failed)

    edges, broken, partial, modelled = [], 0, 0, 0
    for b in man["bonds"]:
        key = (min(b["a"], b["b"]), max(b["a"], b["b"]))
        if fixed[b["a"]] and fixed[b["b"]]:
            edges.append(key)
            continue
        modelled += 1
        intact, total = joint_bonds.get(key, [0, 0])
        if total == 0:
            notes.append("scene bond %s has no particle bonds" % (key,))
        if intact > 0:
            edges.append(key)
            if intact < total:
                partial += 1
        else:
            broken += 1
    expected = sum(man["particle_bonds_per_scene_bond"].values())
    found = sum(v[1] for v in joint_bonds.values())
    if found != expected:
        notes.append("particle bonds across joints: %d found vs %d expected from geometry" % (found, expected))

    comp = components(n, edges)
    main_roots = {comp[i] for i in range(n) if fixed[i]}
    roots = sorted(set(comp))
    frag = {r: k for k, r in enumerate(roots)}

    mass, vel, pos, coor0 = raw["mass"], raw["vel"], raw["pos"], raw["coor0"]
    states = []
    for i in range(n):
        sel = pc == i
        w = mass[sel] / mass[sel].sum()
        states.append({
            "detached": comp[i] not in main_roots,
            "removed": False,
            "fragment": frag[comp[i]],
            "velocity": [float(x) for x in w @ vel[sel]],
            "displacement": [float(x) for x in w @ (pos[sel] - coor0[sel])],
        })

    t = raw["t"].tolist()
    probes = {}
    for j, pr in enumerate(man["probe_axes"]):
        probes[pr["name"]] = {"t": t, "v": raw["probe_v"][:, j].tolist(),
                              "lo": raw["probe_lo"][:, j].tolist(), "hi": raw["probe_hi"][:, j].tolist()}
    measured = {p["name"] for p in man["probe_axes"]}
    for p in man["probes"]:
        if p["name"] not in measured:
            notes.append("probe '%s' (%s) not measured by this oracle" % (p["name"], p["type"]))

    P = man["params"]
    notes = [
        "Kratos DEM continuum (SphericContinuumParticle3D, explicit symplectic Euler, dt=%.3g s = %.2f x critical estimate); "
        "each chunk = cubic pack of spheres (spacing %g m, radius %g m) bonded by stiff unbreakable bonds (brick_stiffness_ratio %g)"
        % (man["dt"], man["dt"] / man["dt_critical_estimate"], man["particle_size"], man["particle_radius"], P["brick_stiffness_ratio"]),
        "mortar joints: DEM_parallel_bond_bilinear_damage_Linear on every particle bond crossing a scene bond; per joint area: "
        "tension cutoff f_t, Mohr-Coulomb tau0=c plus mu*sigma, linear softening with G_f (tension/shear), coupled damage; "
        "chunk-centre-to-centre normal/shear compliance equals the scene bond's kn/ks (joint bond in series with brick bonds)",
        "unbonded contacts: DEM_D_Linear_classic, E = chunk material E, friction = joint (cracked) or min chunk friction, "
        "restitution %g (not in the scene; same as the reference solver's contact constant); impactor: one rigid-sized sphere, "
        "mass from the scene, friction min(impactor, brick)" % P["impact_restitution"],
        "gravity settling %d steps (%.3g s) with the impactor held; kinetic energy after settling %.3e J; t=0 = impactor release"
        % (man["n_settle"], man["n_settle"] * man["dt"], run["ke_after_settle"]),
        "joint intact = at least one particle bond across it never failed; %d joints partially cracked at end_time" % partial,
    ] + man.get("notes", []) + notes

    return {
        "format": "stress-observation/1",
        "scene": man["scene"],
        "solver": "kratos",
        "solver_version": run["solver_version"],
        "seed": man["seed"],
        "end_time": man["end_time"],
        "probes": probes,
        "bodies": {man["body"]: states},
        "flags": {"any_bond_broken": broken > 0, "collapse": any(s["detached"] for s in states)},
        "values": {
            "broken_bonds": float(broken), "partially_cracked_bonds": float(partial), "modelled_bonds": float(modelled),
            "fragments": float(len(roots)), "wall_seconds": float(run["wall_seconds"]),
            "particles": float(run["n_particles"]),
        },
        "notes": notes,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    obs = observe(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(obs, f, separators=(",", ":"))
        f.write("\n")
