"""Post-processor: LMGC90 raw results -> stress-observation/1 JSON.

Reads ``<deck>/manifest.json``, ``<deck>/raw.npz`` and ``<deck>/run.json``; never
the scene. Connectivity uses the cohesive state of every scene bond at end_time:

* a bond is *intact* when at least one LMGC90 contact point between its two
  bodies still carries a cohesive law with beta > 0 (EXPO_CZM internal 5);
  a bond whose bodies lost contact entirely is broken; bonds between two fixed
  chunks (not modelled, both driven) count as intact;
* chunks are grouped by intact bonds; the group containing a fixed chunk is the
  main piece, every other chunk is ``detached``; ``fragment`` = group id;
* ``velocity`` = rigid-body (centre of mass = box centre) velocity at end_time,
  ``displacement`` = centre displacement from the scene position (includes the
  gravity settling, which is ~1e-6 m).

Usage: python -I observe.py <deck_dir> <out.json>
"""

import json
import pathlib
import sys

import numpy as np

BETA_INDEX = 4  # internal(5) of IQS_EXPO_CZM / IQS_MAL_CZM = beta (1 intact, 0 broken)


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
    deck_dir = pathlib.Path(deck_dir)
    man = json.loads((deck_dir / "manifest.json").read_text())
    run = json.loads((deck_dir / "run.json").read_text())
    raw = np.load(deck_dir / "raw.npz")
    czm_laws = {l["name"] for l in man["laws"] if l["law"].endswith("_CZM")}

    chunk_body = man["chunk_body"]
    body_chunk = {b: i for i, b in enumerate(chunk_body)}
    n = len(chunk_body)
    fixed = [s == "fixed" for s in man["chunk_support"]]

    def cohesive_pairs(prefix):
        law, cd, an, internal = raw[prefix + "law"], raw[prefix + "cd"], raw[prefix + "an"], raw[prefix + "internal"]
        out = {}
        for k in range(len(law)):
            if law[k] not in czm_laws or cd[k] not in body_chunk or an[k] not in body_chunk:
                continue
            key = tuple(sorted((body_chunk[cd[k]], body_chunk[an[k]])))
            out[key] = max(out.get(key, 0.0), float(internal[k, BETA_INDEX]))
        return out

    first = cohesive_pairs("first_")
    last = cohesive_pairs("last_")
    notes = []
    intact_edges, broken, modelled = [], 0, 0
    unmatched_initial = 0
    for b in man["bonds"]:
        key = tuple(sorted((b["a"], b["b"])))
        if fixed[b["a"]] and fixed[b["b"]]:
            intact_edges.append(key)
            continue
        modelled += 1
        if first.get(key, 0.0) < 1.0:
            unmatched_initial += 1
        if last.get(key, 0.0) > 0.0:
            intact_edges.append(key)
        else:
            broken += 1
    extra_initial = [k for k in first if not any(tuple(sorted((b["a"], b["b"]))) == k for b in man["bonds"])]
    if unmatched_initial:
        notes.append("%d scene bonds had no intact cohesive contact at the first step" % unmatched_initial)
    if extra_initial:
        notes.append("%d cohesive contacts at the first step do not correspond to scene bonds" % len(extra_initial))

    comp = components(n, intact_edges)
    main_roots = {comp[i] for i in range(n) if fixed[i]}
    roots = sorted(set(comp))
    frag_id = {r: k for k, r in enumerate(roots)}

    vel, pos = raw["vel"], raw["pos"]
    center0 = np.array(man["chunk_center"])
    states = []
    for i in range(n):
        bi = chunk_body[i] - 1
        states.append({
            "detached": comp[i] not in main_roots,
            "removed": False,
            "fragment": frag_id[comp[i]],
            "velocity": [float(x) for x in vel[bi, :3]],
            "displacement": [float(x) for x in pos[bi] - center0[i]],
        })
    any_detached = any(s["detached"] for s in states)

    probes = {}
    t = raw["t"].tolist()
    for j, pr in enumerate(man["probe_axes"]):
        v = raw["probe_v"][:, j].tolist()
        lo = raw["probe_lo"][:, j].tolist()
        hi = raw["probe_hi"][:, j].tolist()
        probes[pr["name"]] = {"t": t, "v": v, "lo": lo, "hi": hi}
    measured = {p["name"] for p in man["probe_axes"]}
    for p in man["probes"]:
        if p["name"] not in measured:
            notes.append("probe '%s' (%s) not measured by this oracle" % (p["name"], p["type"]))

    nl = run["nlgs"]
    notes += [
        "LMGC90 NSCD (Moreau-Jean theta=%.2f, dt=%g s, NLGS %s tol %g, <= %d iterations/step); rigid POLYR bricks, rigid SPHER impactor"
        % (man["params"]["theta"], man["dt"], nl["norm"], nl["tol"], nl["max_iterations"]),
        "mortar joints: %s cohesive zone per bond (cn=E/L, ct=G/L per area from the bond's derived kn/ks; s1=f_t, s2=c, G1=G_f,t, G2=G_f,s; "
        "intact joints keep Coulomb friction mu (SetCZMwithInitialFriction(0)) so shear capacity ~ c + mu sigma); contacts appearing later are frictional only"
        % man["params"]["joint_law"],
        "impactor contact: %s with normal restitution %g (not in the scene; same value as the reference solver's contact constant) "
        "and mu = min(impactor, brick) friction" % ("RST_CLB" if man["params"]["impact_restitution"] > 0 else "IQS_CLB", man["params"]["impact_restitution"]),
        "gravity settling %d steps (%.3g s) with the impactor invisible; kinetic energy after settling %.3e J; t=0 = impactor release"
        % (man["n_settle"], man["n_settle"] * man["dt"], run["ke_after_settle"]),
        "NLGS unconverged steps: settle %d, impact %d of %d (mean %.0f, max %d iterations)"
        % (nl["settle_unconverged_steps"], nl["impact_unconverged_steps"], man["n_steps"], nl["impact_iterations_mean"], nl["impact_iterations_max"]),
        "detached = not connected to a fixed chunk through bonds with an intact (beta>0) cohesive contact point",
    ] + man.get("notes", [])

    obs = {
        "format": "stress-observation/1",
        "scene": man["scene"],
        "solver": "lmgc90",
        "solver_version": run["solver_version"],
        "seed": man["seed"],
        "end_time": man["end_time"],
        "probes": probes,
        "bodies": {man["body"]: states},
        "flags": {"any_bond_broken": broken > 0, "collapse": any_detached},
        "values": {
            "broken_bonds": float(broken),
            "modelled_bonds": float(modelled),
            "fragments": float(len(roots)),
            "wall_seconds": float(run["wall_seconds"]),
            "nlgs_unconverged_steps": float(nl["impact_unconverged_steps"]),
        },
        "notes": notes,
    }
    return obs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    obs = observe(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(obs, f, separators=(",", ":"))
        f.write("\n")
