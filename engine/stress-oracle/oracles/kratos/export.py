"""Exporter: stress-scene/1 JSON -> Kratos DEM (DEMApplication) bonded-particle deck.

Reads only the scene file. Writes into <deck>:

* ``wallDEM.mdpa``        spheres model part: every chunk is a regular cubic packing
                          of SphericContinuumParticle3D (spacing s, radius s/2), the
                          impactor is one sphere particle;
* ``wallDEM_FEM_boundary.mdpa``  (empty; no rigid walls);
* ``MaterialsDEM.json``   one material per chunk colour + material relations
                          (bond laws), see README.md;
* ``ProjectParametersDEM.json``;
* ``manifest.json``       particle <-> chunk map, fixed particles, impactor, schedule.

Model: bricks are bonded sphere packs whose internal bonds are unbreakable and
stiff; every scene bond (mortar joint) becomes the set of particle bonds that
cross it, with ``DEM_parallel_bond_bilinear_damage_Linear`` (tension cutoff,
Mohr-Coulomb shear tau0 + mu sigma, linear softening by fracture energy, Coulomb
friction after failure). Parameters are scaled per particle bond so that, per unit
joint area, strength, fracture energy and (in series with the brick bonds along
the chunk-centre path) the normal/shear stiffness equal the scene bond's.

Usage: python -I export.py <scene.json> <deck_dir>
"""

import hashlib
import itertools
import json
import math
import pathlib
import sys

import numpy as np

DEFAULTS = {
    "particle_size": None,       # lattice spacing s (m); default = half the smallest chunk edge
    "brick_stiffness_ratio": 5.0,  # intra-brick bond stiffness / stiffest joint stiffness (per area)
    "dt": None,                  # default: safety * critical estimate
    "dt_safety": 0.12,           # x the translational estimate below; particle rotation makes the scheme
                                 # unstable at 5e-6 s here (empirical), 3e-6 s is stable
    "settle_time": 0.01,         # gravity settling with the impactor held (s)
    "search_tolerance": 2.0e-3,  # neighbour search gap tolerance (m); also the initial-bond tolerance
    "impact_restitution": 0.2,   # not in the scene: same value the reference solver uses for contacts
    "contact_young_modulus": None,  # unbonded particle contact E; default = chunk material E
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def course_colours(chunks):
    """Colour = (support, size, course parity, position parity within the course).

    Two chunks of one colour never touch, so a material relation (i, i) only ever
    holds intra-chunk bonds and (i, j), i != j, holds joints; verified by the caller.
    """
    sig = []
    for c in chunks:
        hz = c["half_extents"][2]
        course = int(math.floor(c["center"][2] / (2.0 * hz) + 1e-9))
        sig.append((c["support"], tuple(round(h, 6) for h in c["half_extents"]), course % 2, course))
    order = {}
    for i, s in enumerate(sig):
        order.setdefault(s[3], []).append(i)
    xpar = [0] * len(chunks)
    for course, idx in order.items():
        idx.sort(key=lambda i: chunks[i]["center"][0])
        for k, i in enumerate(idx):
            xpar[i] = k % 2
    keys = [(s[0], s[1], s[2], xpar[i], chunks[i]["material"]) for i, s in enumerate(sig)]
    uniq = sorted(set(keys))
    return [uniq.index(k) + 1 for k in keys], {str(i + 1): list(map(str, k)) for i, k in enumerate(uniq)}


def export(scene_path, deck_dir):
    scene_path = pathlib.Path(scene_path)
    deck = pathlib.Path(deck_dir)
    deck.mkdir(parents=True, exist_ok=True)
    scene = json.loads(scene_path.read_text())
    if scene.get("format") != "stress-scene/1":
        raise SystemExit("not a stress-scene/1 file")
    P = dict(DEFAULTS)
    P.update(scene.get("oracle", {}).get("kratos", {}) or {})
    notes = []

    if len(scene["bodies"]) != 1 or scene.get("loads") or scene.get("events") or scene.get("ground") is not None:
        raise SystemExit("kratos exporter: one body, no loads/events/ground supported")
    body = scene["bodies"][0]
    if body.get("orientation", [1, 0, 0, 0]) != [1.0, 0.0, 0.0, 0.0] or any(body.get("position", [0, 0, 0])):
        raise SystemExit("kratos exporter: body pose must be identity")
    chunks = body["chunks"]
    for c in chunks:
        if c.get("orientation", [1, 0, 0, 0]) != [1.0, 0.0, 0.0, 0.0] or c.get("level", 0) != 0 or c["support"] == "pinned":
            raise SystemExit("kratos exporter: axis-aligned level-0 chunks with none/fixed support only")
    if any(b.get("rebar") for b in body["bonds"]):
        raise SystemExit("kratos exporter: rebar not supported")
    if len(scene["impactors"]) != 1 or scene["impactors"][0]["shape"]["type"] != "sphere":
        raise SystemExit("kratos exporter: one sphere impactor expected")
    imp = scene["impactors"][0]
    mats = scene["materials"]

    s = P["particle_size"] or min(min(2 * h for h in c["half_extents"]) for c in chunks) / 2.0
    r = 0.5 * s
    A_b = math.pi * r * r  # BOND_RADIUS_FACTOR = 1

    # Particles.
    nodes, part_chunk = [], []
    for i, c in enumerate(chunks):
        n = [int(round(2 * h / s)) for h in c["half_extents"]]
        if any(abs(k * s - 2 * h) > 1e-9 for k, h in zip(n, c["half_extents"])):
            raise SystemExit("kratos exporter: chunk %d is not a multiple of the particle size %g" % (i, s))
        lo = np.array(c["center"]) - np.array(c["half_extents"])
        for a, b, d in itertools.product(range(n[0]), range(n[1]), range(n[2])):
            nodes.append(lo + s * (np.array([a, b, d]) + 0.5))
            part_chunk.append(i)
    nodes = np.array(nodes)
    part_chunk = np.array(part_chunk)

    colours, colour_sig = course_colours(chunks)
    # Bond classes per colour pair.
    pair_class = {}
    kn_area_max = 0.0
    for b in body["bonds"]:
        ca, cb = colours[b["a"]], colours[b["b"]]
        if ca == cb:
            raise SystemExit("kratos exporter: bonded chunks %d, %d share a colour" % (b["a"], b["b"]))
        m = mats[b["material"]]
        d = b["derived"]
        cls = {
            "cn": d["kn"] / b["area"], "ct": d["ks"] / b["area"], "L": d["length"],
            "ft": m["tensile_strength"] * d["weibull"], "c": m["cohesion"] * d["weibull"],
            "Gt": m["fracture_energy"]["tension"], "Gs": m["fracture_energy"]["shear"],
            "mu": m["friction"], "damping": m["damping_ratio"],
            "fixed": chunks[b["a"]]["support"] == "fixed" and chunks[b["b"]]["support"] == "fixed",
        }
        key = tuple(sorted((ca, cb)))
        rounded = {k: (float("%.6g" % v) if isinstance(v, float) else v) for k, v in cls.items()}
        if key in pair_class and pair_class[key][0] != rounded:
            raise SystemExit("kratos exporter: colour pair %s carries different joints (per-bond scatter not representable)" % (key,))
        pair_class[key] = (rounded, cls)
        kn_area_max = max(kn_area_max, cls["cn"])

    # Intra-chunk ("brick") bonds: stiff, unbreakable.
    R = float(P["brick_stiffness_ratio"])
    k_in_n = R * kn_area_max * s * s
    knks_in = 2.0 * (1.0 + mats[chunks[0]["material"]]["poisson_ratio"])
    k_in_t = k_in_n / knks_in

    def joint_constants(cls):
        n_intra = cls["L"] / s - 1.0
        if abs(n_intra - round(n_intra)) > 1e-6:
            raise SystemExit("kratos exporter: bond length %g not a multiple of s" % cls["L"])
        cn_res = 1.0 / cls["cn"] - n_intra * s * s / k_in_n
        ct_res = 1.0 / cls["ct"] - n_intra * s * s / k_in_t
        if cn_res <= 0 or ct_res <= 0:
            raise SystemExit("kratos exporter: brick_stiffness_ratio too small for the joint stiffness")
        kn = s * s / cn_res
        kt = s * s / ct_res
        sig = cls["ft"] * s * s / A_b
        tau = cls["c"] * s * s / A_b
        Gn = cls["Gt"] * s * s / A_b
        Gt = cls["Gs"] * s * s / A_b
        coeff_n = 2.0 * Gn * kn / (A_b * sig * sig) - 1.0
        coeff_t = 2.0 * Gt * kt / (A_b * tau * tau) - 1.0
        if coeff_n > 30 or coeff_t > 30:
            raise SystemExit("kratos exporter: fracture energy too large for DEM_parallel_bond_bilinear_damage (coeff %.1f/%.1f)" % (coeff_n, coeff_t))
        return {"kn": kn, "kt": kt, "BOND_YOUNG_MODULUS": kn * s / A_b, "BOND_KNKS_RATIO": kn / kt,
                "BOND_SIGMA_MAX": sig, "BOND_TAU_ZERO": tau, "FRACTURE_ENERGY_NORMAL": Gn,
                "FRACTURE_ENERGY_TANGENTIAL": Gt, "softening_coeff_n": coeff_n, "softening_coeff_t": coeff_t,
                "n_intra": int(round(n_intra))}

    # Materials and relations.
    ncol = max(colours)
    ball_id = ncol + 1
    colour_mat = {}
    for i, c in enumerate(chunks):
        colour_mat.setdefault(colours[i], c["material"])
        if colour_mat[colours[i]] != c["material"]:
            raise SystemExit("kratos exporter: colour mixes chunk materials")
    rho_eff = {}
    materials = []
    for k in range(1, ncol + 1):
        m = mats[colour_mat[k]]
        rho = m["density"] * s ** 3 / (4.0 / 3.0 * math.pi * r ** 3)  # lattice cell mass in one sphere
        rho_eff[k] = rho
        materials.append({"material_name": "C%d" % k, "material_id": k, "Variables": {
            "PARTICLE_DENSITY": rho, "YOUNG_MODULUS": P["contact_young_modulus"] or m["youngs_modulus"],
            "POISSON_RATIO": m["poisson_ratio"], "PARTICLE_SPHERICITY": 1.0}})
    rb = imp["shape"]["radius"]
    im = mats[imp["material"]]
    ball_rho = imp["mass"] / (4.0 / 3.0 * math.pi * rb ** 3)
    materials.append({"material_name": "IMPACTOR", "material_id": ball_id, "Variables": {
        "PARTICLE_DENSITY": ball_rho, "YOUNG_MODULUS": im["youngs_modulus"], "POISSON_RATIO": im["poisson_ratio"],
        "PARTICLE_SPHERICITY": 1.0}})

    def contact_vars(mu):
        return {"DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME": "DEM_D_Linear_classic",
                "STATIC_FRICTION": mu, "DYNAMIC_FRICTION": mu, "FRICTION_DECAY": 500.0,
                "COEFFICIENT_OF_RESTITUTION": P["impact_restitution"], "K_ALPHA": 45.0,
                "DEM_ROLLING_FRICTION_MODEL_NAME": "DEMRollingFrictionModelConstantTorque",
                "ROLLING_FRICTION": 0.0, "ROLLING_FRICTION_WITH_WALLS": 0.0}

    def bond_vars(young, knks, sig, tau, mu_int, gn, gt, damping, unbreakable):
        return {"DEM_CONTINUUM_CONSTITUTIVE_LAW_NAME": "DEM_parallel_bond_bilinear_damage_Linear",
                "BOND_YOUNG_MODULUS": young, "BOND_KNKS_RATIO": knks,
                "BOND_SIGMA_MAX": sig, "BOND_SIGMA_MAX_DEVIATION": 0.0,
                "BOND_TAU_ZERO": tau, "BOND_TAU_ZERO_DEVIATION": 0.0,
                "BOND_INTERNAL_FRICC": mu_int,
                "BOND_ROTATIONAL_MOMENT_COEFFICIENT_NORMAL": 0.0,
                "BOND_ROTATIONAL_MOMENT_COEFFICIENT_TANGENTIAL": 0.0,
                "BOND_RADIUS_FACTOR": 1.0, "IS_UNBREAKABLE": unbreakable,
                "FRACTURE_ENERGY_NORMAL": gn, "FRACTURE_ENERGY_TANGENTIAL": gt,
                "DAMPING_GAMMA": damping}

    relations, law_table = [], []
    # Intra-chunk bonds are IS_UNBREAKABLE; their nominal strength only sets Kratos'
    # neighbour-search extension (u_max = sigma A / kn), so keep it modest.
    big = 10.0 * max([c["ft"] for _, c in pair_class.values()] + [c["c"] for _, c in pair_class.values()]) * s * s / A_b
    for k in range(1, ncol + 1):
        m = mats[colour_mat[k]]
        v = contact_vars(m["friction"])
        v.update(bond_vars(k_in_n * s / A_b, knks_in, big, big, m["friction"], 0.0, 0.0, m["damping_ratio"], True))
        relations.append({"material_names_list": ["C%d" % k, "C%d" % k], "material_ids_list": [k, k], "Variables": v})
    for ka, kb in itertools.combinations(range(1, ncol + 1), 2):
        ma, mb = mats[colour_mat[ka]], mats[colour_mat[kb]]
        if (ka, kb) in pair_class:
            _, cls = pair_class[(ka, kb)]
            j = joint_constants(cls)
            v = contact_vars(cls["mu"])
            v.update(bond_vars(j["BOND_YOUNG_MODULUS"], j["BOND_KNKS_RATIO"], j["BOND_SIGMA_MAX"], j["BOND_TAU_ZERO"],
                               cls["mu"], j["FRACTURE_ENERGY_NORMAL"], j["FRACTURE_ENERGY_TANGENTIAL"], cls["damping"],
                               bool(cls["fixed"])))
            law_table.append({"colours": [ka, kb], "joint": cls, "particle_bond": j})
        else:
            mu = min(ma["friction"], mb["friction"])
            v = contact_vars(mu)
            # Never bonded (no initial neighbours, checked below); law present for completeness.
            v.update(bond_vars(k_in_n * s / A_b, knks_in, 0.0, 0.0, mu, 0.0, 0.0, ma["damping_ratio"], False))
        relations.append({"material_names_list": ["C%d" % ka, "C%d" % kb], "material_ids_list": [ka, kb], "Variables": v})
    for k in range(1, ncol + 1):
        mu = min(im["friction"], mats[colour_mat[k]]["friction"])
        relations.append({"material_names_list": ["IMPACTOR", "C%d" % k], "material_ids_list": [ball_id, k],
                          "Variables": contact_vars(mu)})
    relations.append({"material_names_list": ["IMPACTOR", "IMPACTOR"], "material_ids_list": [ball_id, ball_id],
                      "Variables": contact_vars(im["friction"])})

    # Initial neighbours must exist only inside chunks and across scene bonds.
    bonded = {tuple(sorted((b["a"], b["b"]))) for b in body["bonds"]}
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    pairs = tree.query_pairs(2 * r + P["search_tolerance"])
    bond_count = {}
    for i, j in pairs:
        ci, cj = part_chunk[i], part_chunk[j]
        if ci == cj:
            continue
        key = tuple(sorted((int(ci), int(cj))))
        if key not in bonded:
            raise SystemExit("kratos exporter: particles of unbonded chunks %s are initial neighbours" % (key,))
        bond_count[key] = bond_count.get(key, 0) + 1
    missing = [k for k in bonded if k not in bond_count]
    if missing:
        raise SystemExit("kratos exporter: %d scene bonds have no particle bond" % len(missing))
    for b in body["bonds"]:
        key = tuple(sorted((b["a"], b["b"])))
        if abs(bond_count[key] * s * s - b["area"]) > 1e-9:
            notes.append("bond %s: particle-bond area %g != scene area %g" % (key, bond_count[key] * s * s, b["area"]))

    # mdpa.
    n_wall = len(nodes)
    ball_node = n_wall + 1
    lines = ["Begin ModelPartData", "End ModelPartData", "", "Begin Properties 0", "End Properties", "", "Begin Nodes"]
    for i, p in enumerate(nodes):
        lines.append("%d %.12g %.12g %.12g" % (i + 1, p[0], p[1], p[2]))
    bp = imp["position"]
    lines.append("%d %.12g %.12g %.12g" % (ball_node, bp[0], bp[1], bp[2]))
    lines += ["End Nodes", "", "Begin Elements SphericContinuumParticle3D"]
    for i in range(n_wall + 1):
        lines.append("%d 0 %d" % (i + 1, i + 1))
    lines += ["End Elements", "", "Begin NodalData RADIUS"]
    for i in range(n_wall):
        lines.append("%d 0 %.12g" % (i + 1, r))
    lines.append("%d 0 %.12g" % (ball_node, rb))
    lines += ["End NodalData", "", "Begin NodalData COHESIVE_GROUP"]
    for i in range(n_wall):
        lines.append("%d 0 1" % (i + 1))
    lines.append("%d 0 0" % ball_node)
    lines += ["End NodalData", ""]
    groups = {}
    for i in range(n_wall):
        groups.setdefault(colours[part_chunk[i]], []).append(i + 1)
    assign = []
    for k in sorted(groups):
        name = "DEMParts_C%d" % k
        lines += ["Begin SubModelPart %s" % name, "Begin SubModelPartNodes"] + ["%d" % x for x in groups[k]]
        lines += ["End SubModelPartNodes", "Begin SubModelPartElements"] + ["%d" % x for x in groups[k]]
        lines += ["End SubModelPartElements", "Begin SubModelPartConditions", "End SubModelPartConditions", "End SubModelPart", ""]
        assign.append(["SpheresPart.%s" % name, "C%d" % k])
    lines += ["Begin SubModelPart DEMParts_Impactor", "Begin SubModelPartNodes", "%d" % ball_node, "End SubModelPartNodes",
              "Begin SubModelPartElements", "%d" % ball_node, "End SubModelPartElements",
              "Begin SubModelPartConditions", "End SubModelPartConditions", "End SubModelPart", ""]
    assign.append(["SpheresPart.DEMParts_Impactor", "IMPACTOR"])
    (deck / "wallDEM.mdpa").write_text("\n".join(lines) + "\n")
    (deck / "wallDEM_FEM_boundary.mdpa").write_text(
        "Begin ModelPartData\nEnd ModelPartData\n\nBegin Properties 0\nEnd Properties\n\nBegin Nodes\nEnd Nodes\n")
    (deck / "MaterialsDEM.json").write_text(json.dumps(
        {"materials": materials, "material_relations": relations, "material_assignation_table": assign}, indent=1))

    # Time step: explicit stability of the stiffest (intra-chunk) bonds.
    m_p = min(rho_eff.values()) * 4.0 / 3.0 * math.pi * r ** 3
    k_max = max(k_in_n, max(joint_constants(c)["kn"] for _, c in pair_class.values()))
    dt_crit = 2.0 / (2.0 * math.sqrt(k_max / m_p))
    dt = float(P["dt"] or P["dt_safety"] * dt_crit)
    duration = float(scene["sim"]["duration"])
    sample = float(scene["sim"].get("sample_interval") or scene["sim"]["frame_dt"])
    n_settle = int(round(float(P["settle_time"]) / dt)) if scene["sim"].get("gravity_prestress", False) else 0
    n_steps = int(math.ceil(duration / dt - 1e-9))
    g = scene.get("gravity", [0.0, 0.0, -9.81])
    final_time = (n_settle + n_steps + 2) * dt

    params = {
        "Dimension": 3, "PeriodicDomainOption": False, "BoundingBoxOption": False, "AutomaticBoundingBoxOption": False,
        "dem_inlet_option": False, "GravityX": g[0], "GravityY": g[1], "GravityZ": g[2],
        "RotationOption": True, "CleanIndentationsOption": False, "VirtualMassCoefficient": 1.0,
        "RollingFrictionOption": False, "GlobalDamping": 0.0, "ContactMeshOption": True,
        "OutputFileType": "Binary", "Multifile": "multiple_files",
        "ElementType": "SphericContPartDEMElement3D",
        "TranslationalIntegrationScheme": "Symplectic_Euler", "RotationalIntegrationScheme": "Direct_Integration",
        "MaxTimeStep": dt, "FinalTime": final_time, "NeighbourSearchFrequency": 1,
        "SearchTolerance": P["search_tolerance"], "DeltaOption": "Absolute", "CoordinationNumber": 6,
        "AmplifiedSearchRadiusExtension": 1.0, "MaxAmplificationRatioOfSearchRadius": 1000,
        "PoissonEffectOption": False, "ShearStrainParallelToBondOption": False, "ComputeStressTensorOption": False,
        "GraphExportFreq": 1e9, "VelTrapGraphExportFreq": 1e9, "OutputTimeStep": 1e9,
        "PostContactFailureId": True, "do_print_results_option": False,
        "problem_name": "wall",
        "EnergyCalculationOption": False, "VelocityTrapOption": False, "ModelDataInfo": False,
        "DontSearchUntilFailure": False, "MaxNumberOfIntactBondsToConsiderASphereBroken": 0,
        "AutomaticTimestep": False, "DeltaTimeSafetyFactor": 1.0, "ControlTime": 1e9,
        "solver_settings": {"RemoveBallsInitiallyTouchingWalls": False, "strategy": "continuum_sphere_strategy",
                            "material_import_settings": {"materials_filename": "MaterialsDEM.json"}},
    }
    (deck / "ProjectParametersDEM.json").write_text(json.dumps(params, indent=1))

    deck_hash = hashlib.sha256()
    for name in ("wallDEM.mdpa", "wallDEM_FEM_boundary.mdpa", "MaterialsDEM.json", "ProjectParametersDEM.json"):
        deck_hash.update(name.encode())
        deck_hash.update((deck / name).read_bytes())

    manifest = {
        "scene": scene["name"], "scene_file": str(scene_path.resolve()), "scene_sha256": sha256_file(scene_path),
        "seed": scene["sim"].get("seed", 0), "params": P, "particle_size": s, "particle_radius": r,
        "dt": dt, "dt_critical_estimate": dt_crit, "n_settle": n_settle, "n_steps": n_steps,
        "sample_every": max(1, int(round(sample / dt))), "sample_interval": sample, "end_time": n_steps * dt,
        "body": body["name"], "n_chunks": len(chunks),
        "particle_chunk": part_chunk.tolist(),
        "fixed_nodes": [i + 1 for i in range(n_wall) if chunks[part_chunk[i]]["support"] == "fixed"],
        "chunk_center": [c["center"] for c in chunks], "chunk_support": [c["support"] for c in chunks],
        "chunk_colour": colours, "colour_signature": colour_sig,
        "bonds": [{"a": b["a"], "b": b["b"]} for b in body["bonds"]],
        "particle_bonds_per_scene_bond": {"%d-%d" % k: v for k, v in sorted(bond_count.items())},
        "intra_bond": {"kn": k_in_n, "kt": k_in_t},
        "joint_laws": law_table,
        "impactor": {"name": imp["name"], "node": ball_node, "position": imp["position"], "velocity": imp["velocity"],
                     "angular_velocity": imp.get("angular_velocity", [0, 0, 0]), "mass": imp["mass"], "radius": rb},
        "probe_axes": [{"name": p["name"], "axis": p["axis"]} for p in scene.get("probes", [])
                       if p["type"] == "impactor_velocity" and p["impactor"] == imp["name"]],
        "probes": scene.get("probes", []),
        "deck_sha256": deck_hash.hexdigest(),
        "notes": notes,
    }
    (deck / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    m = export(sys.argv[1], sys.argv[2])
    print("deck %s: %d particles, dt %.3g (crit %.3g), %d+%d steps, deck sha256 %s" % (
        sys.argv[2], len(m["particle_chunk"]) + 1, m["dt"], m["dt_critical_estimate"], m["n_settle"], m["n_steps"], m["deck_sha256"]))
