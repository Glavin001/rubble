"""Exporter: stress-scene/1 JSON -> LMGC90 (pylmgc90) DATBOX deck + run manifest.

Reads only the scene file. Writes

* ``<deck>/DATBOX/*``  (BODIES.DAT, BULK_BEHAV.DAT, TACT_BEHAV.DAT, DRV_DOF.DAT, ...)
  through ``pylmgc90.pre.writeDatbox``;
* ``<deck>/manifest.json``: the run schedule (time step, settle steps, sampling),
  the scene-chunk <-> LMGC90 body numbering, the scene bonds and the contact-law
  table, so ``run.py`` and ``observe.py`` never re-read or re-interpret the scene.

Model (see README.md): every scene chunk is a rigid POLYR box, the impactor is a
rigid SPHER, ``fixed`` chunks have all 6 velocity DOFs driven to zero. Every scene
bond (mortar joint) is an ``IQS_EXPO_CZM`` cohesive interface (unilateral contact +
mixed-mode cohesive zone with exponential softening, Coulomb friction kept after
failure); everything else is ``IQS_CLB`` Coulomb contact. LMGC90 assigns contact
laws per (candidate colour, antagonist colour) pair, so chunks are coloured such
that every colour pair maps to exactly one bond class (stiffness/strength); the
exporter verifies that mapping and refuses scenes it cannot represent.

Usage: python -I export.py <scene.json> <deck_dir>
"""

import hashlib
import itertools
import json
import math
import pathlib
import sys

import numpy as np

# Numerical settings (not physics). Overridable per scene via scene["oracle"]["lmgc90"].
DEFAULTS = {
    "dt": 1.0e-4,            # NSCD time step (s); theta-method, implicit cohesive springs
    "theta": 0.5,
    "settle_time": 0.01,     # gravity settling before the impactor is made visible (s)
    "alert": 2.0e-3,         # contact detection distance (m)
    "f2f_tol": 1.0e-3,       # face-to-face detection tolerance (cos of normal mismatch)
    "shrink": 1.0e-3,        # PRPRx_ShrinkPolyrFaces: avoids degenerate shared-corner projections
    "detection": "f2f",
    "impact_restitution": 0.2,   # not in the scene; RST_CLB normal restitution (0 -> IQS_CLB)
    "joint_law": "IQS_EXPO_CZM",  # IQS_MAL_CZM (other softening shape) / IQS_CLB (diagnostic, no cohesion)
    "eta": 0.01,             # EXPO_CZM residual ratio at which a joint is fully broken
    "nlgs_tol": 1.666e-4,    # LMGC90's customary QM/16 tolerance
    "nlgs_relax": 1.0,
    "nlgs_norm": "QM/16",
    "gs_it1": 50,            # iterations between convergence checks
    "gs_it2": 200,           # max checks -> at most 10000 NLGS iterations per step
}

COLOR_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def box_vertices_faces(half):
    """Box centred at the origin: 8 vertices, 12 outward triangles (1-based)."""
    hx, hy, hz = half
    v = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = []
    for a, b, c, d in quads:
        tris.append((a + 1, b + 1, c + 1))
        tris.append((a + 1, c + 1, d + 1))
    return v, np.array(tris, dtype=int)


def chunk_world(body, chunk):
    rb = quat_to_mat(body.get("orientation", [1, 0, 0, 0]))
    rc = quat_to_mat(chunk.get("orientation", [1, 0, 0, 0]))
    center = rb @ np.array(chunk["center"]) + np.array(body.get("position", [0, 0, 0]))
    return center, rb @ rc


def bond_class(scene, bond):
    """Joint parameters of one scene bond, per unit area (what one LMGC90 law can hold)."""
    m = scene["materials"][bond["material"]]
    d = bond["derived"]
    a = bond["area"]
    w = d["weibull"]
    return {
        "cn": d["kn"] / a,                     # N/m per m^2 (= E / L)
        "ct": d["ks"] / a,                     # N/m per m^2 (= G / L)
        "s1": m["tensile_strength"] * w,       # Pa
        "s2": m["cohesion"] * w,               # Pa
        "G1": m["fracture_energy"]["tension"],  # J/m^2
        "G2": m["fracture_energy"]["shear"],    # J/m^2
        "mu": m["friction"],
    }


def class_key(c):
    return tuple(float("%.6g" % c[k]) for k in ("cn", "ct", "s1", "s2", "G1", "G2", "mu"))


def colour_chunks(chunks):
    """Colour = geometric signature of the chunk (support, size, course parity).

    For the running-bond wall this separates bed joints (different parity), head
    joints between full bricks and head joints next to half bricks; the caller
    verifies that every colour pair carries a single bond class.
    """
    sig = []
    for c in chunks:
        hz = c["half_extents"][2]
        parity = int(math.floor(c["center"][2] / (2.0 * hz) + 1e-9)) % 2 if hz > 0 else 0
        sig.append((c["support"], tuple(round(h, 6) for h in c["half_extents"]), parity, c["material"]))
    uniq = sorted(set(sig))
    if len(uniq) > len(COLOR_CHARS):
        raise SystemExit("lmgc90 exporter: too many chunk colour classes (%d)" % len(uniq))
    names = {s: ("C" + COLOR_CHARS[i] + "xxx")[:5] for i, s in enumerate(uniq)}
    return [names[s] for s in sig], {names[s]: list(map(str, s)) for s in uniq}


def touching_pairs(body, tol=1e-9):
    """Pairs of axis-aligned chunks sharing a face patch of positive area."""
    out = []
    ch = body["chunks"]
    for i, j in itertools.combinations(range(len(ch)), 2):
        ci, cj = np.array(ch[i]["center"]), np.array(ch[j]["center"])
        hi, hj = np.array(ch[i]["half_extents"]), np.array(ch[j]["half_extents"])
        gap = np.abs(ci - cj) - (hi + hj)
        if np.all(gap <= tol):
            overlap = np.minimum(ci + hi, cj + hj) - np.maximum(ci - hi, cj - hj)
            touching_axes = int(np.sum(np.abs(gap) <= tol))
            if touching_axes == 1 and np.sum(overlap > tol) == 2:
                out.append((i, j))
    return out


def export(scene_path, deck_dir):
    scene_path = pathlib.Path(scene_path)
    deck_dir = pathlib.Path(deck_dir)
    scene = json.loads(scene_path.read_text())
    if scene.get("format") != "stress-scene/1":
        raise SystemExit("not a stress-scene/1 file")
    params = dict(DEFAULTS)
    params.update(scene.get("oracle", {}).get("lmgc90", {}) or {})
    notes = []

    # Scope checks: refuse what this deck cannot represent rather than silently dropping it.
    if len(scene["bodies"]) != 1:
        raise SystemExit("lmgc90 exporter supports exactly one structural body")
    if scene.get("loads") or scene.get("events"):
        raise SystemExit("lmgc90 exporter: loads/events are not exported (none expected for b7)")
    if scene.get("ground") is not None:
        raise SystemExit("lmgc90 exporter: ground plane not supported")
    body = scene["bodies"][0]
    if any(abs(v) > 0 for v in body.get("linear_velocity", [0, 0, 0]) + body.get("angular_velocity", [0, 0, 0])):
        raise SystemExit("lmgc90 exporter: initial body velocity not supported")
    for c in body["chunks"]:
        if c.get("orientation", [1, 0, 0, 0]) != [1.0, 0.0, 0.0, 0.0] or body.get("orientation", [1, 0, 0, 0]) != [1.0, 0.0, 0.0, 0.0]:
            raise SystemExit("lmgc90 exporter: colouring/touching checks assume axis-aligned chunks")
        if c.get("level", 0) != 0:
            raise SystemExit("lmgc90 exporter: refined (level > 0) chunks not supported")
        if c["support"] == "pinned":
            raise SystemExit("lmgc90 exporter: pinned supports not supported")
    for b in body["bonds"]:
        if b.get("rebar"):
            raise SystemExit("lmgc90 exporter: rebar not representable with a CZM joint")
    if len(scene["impactors"]) != 1 or scene["impactors"][0]["shape"]["type"] != "sphere":
        raise SystemExit("lmgc90 exporter: exactly one sphere impactor expected")
    imp = scene["impactors"][0]

    from pylmgc90 import pre

    chunks = body["chunks"]
    colours, colour_sig = colour_chunks(chunks)

    # Bond classes per colour pair.
    pair_class = {}
    # Colours made only of fixed chunks never interact with each other (both sides
    # are driven; LMGC90's cohesive local solver divides by a zero Delassus term there).
    fixed_colours = {col for col in set(colours)
                     if all(chunks[i]["support"] == "fixed" for i in range(len(chunks)) if colours[i] == col)}
    skipped_fixed_bonds = 0
    for k, b in enumerate(body["bonds"]):
        if colours[b["a"]] in fixed_colours and colours[b["b"]] in fixed_colours:
            skipped_fixed_bonds += 1
            continue
        cls = bond_class(scene, b)
        key = class_key(cls)
        pair = tuple(sorted((colours[b["a"]], colours[b["b"]])))
        if pair in pair_class and pair_class[pair][0] != key:
            raise SystemExit("lmgc90 exporter: colour pair %s carries two different bond classes; "
                             "per-bond parameters (e.g. Weibull scatter) cannot be expressed with LMGC90 see tables" % (pair,))
        pair_class[pair] = (key, cls)

    # Every initially touching chunk pair must be a scene bond: LMGC90 makes every
    # contact that exists at the first step cohesive when its colour pair has a CZM law.
    bonded = {tuple(sorted((b["a"], b["b"]))) for b in body["bonds"]}
    touching = touching_pairs(body)
    unbonded_touching = [p for p in touching if p not in bonded
                         and not (chunks[p[0]]["support"] == "fixed" and chunks[p[1]]["support"] == "fixed")]
    for p in unbonded_touching:
        pair = tuple(sorted((colours[p[0]], colours[p[1]])))
        if pair in pair_class:
            raise SystemExit("lmgc90 exporter: touching unbonded chunks %s share a cohesive colour pair" % (p,))
    if skipped_fixed_bonds:
        notes.append("%d bonds between two fixed chunks are not modelled (both sides driven)" % skipped_fixed_bonds)
    missing = [p for p in bonded if p not in set(touching)]
    if missing:
        notes.append("%d scene bonds join chunks that are not face-touching boxes" % len(missing))

    # Materials (rigid bulk, density only).
    mats = pre.materials()
    mods = pre.models()
    mod = pre.model(name="rigid", physics="MECAx", element="Rxx3D", dimension=3)
    mods.addModel(mod)
    mat_of = {}
    for i, name in enumerate(sorted({c["material"] for c in chunks})):
        lname = ("M%03d" % i) + "x"
        mat_of[name] = pre.material(name=lname, materialType="RIGID", density=scene["materials"][name]["density"])
        mats.addMaterial(mat_of[name])
    r = imp["shape"]["radius"]
    ball_density = imp["mass"] / (4.0 / 3.0 * math.pi * r ** 3)
    ball_mat = pre.material(name="IMPxx", materialType="RIGID", density=ball_density)
    mats.addMaterial(ball_mat)

    bodies = pre.avatars()
    lmgc_index = []  # scene chunk i -> LMGC90 RBDY3 number (1-based)
    for i, c in enumerate(chunks):
        center, rot = chunk_world(body, c)
        v, f = box_vertices_faces(c["half_extents"])
        v = (rot @ v.T).T + center
        av = pre.rigidPolyhedron(model=mod, material=mat_of[c["material"]], center=center, color=colours[i],
                                 generation_type="full", vertices=v, faces=f)
        if c["support"] == "fixed":
            av.imposeDrivenDof(component=[1, 2, 3, 4, 5, 6], dofty="vlocy")
        bodies += av
        lmgc_index.append(len(bodies))
    ball = pre.rigidSphere(r=r, center=np.array(imp["position"], dtype=float), model=mod, material=ball_mat, color="IMPAC")
    bodies += ball
    ball_index = len(bodies)
    # The impactor velocity is set by run.py when the ball is released after settling.

    # Contact laws.
    tacts = pre.tact_behavs()
    svs = pre.see_tables()
    laws = []
    for n, (pair, (key, cls)) in enumerate(sorted(pair_class.items())):
        name = ("J%03d" % n) + "x"
        if params["joint_law"] == "IQS_EXPO_CZM":
            law = pre.tact_behav(name=name, law="IQS_EXPO_CZM", dyfr=cls["mu"], stfr=cls["mu"],
                                 cn=cls["cn"], ct=cls["ct"], s1=cls["s1"], s2=cls["s2"],
                                 G1=cls["G1"], G2=cls["G2"], eta=params["eta"])
        elif params["joint_law"] == "IQS_MAL_CZM":
            law = pre.tact_behav(name=name, law="IQS_MAL_CZM", dyfr=cls["mu"], stfr=cls["mu"],
                                 cn=cls["cn"], ct=cls["ct"], s1=cls["s1"], s2=cls["s2"], G1=cls["G1"], G2=cls["G2"])
        elif params["joint_law"] == "IQS_CLB":  # diagnostic only: no cohesion at all
            law = pre.tact_behav(name=name, law="IQS_CLB", fric=cls["mu"])
        else:
            raise SystemExit("unknown joint_law %s" % params["joint_law"])
        tacts += law
        svs += pre.see_table(CorpsCandidat="RBDY3", candidat="POLYR", colorCandidat=pair[0], behav=law,
                             CorpsAntagoniste="RBDY3", antagoniste="POLYR", colorAntagoniste=pair[1], alert=params["alert"])
        if pair[0] != pair[1]:
            svs += pre.see_table(CorpsCandidat="RBDY3", candidat="POLYR", colorCandidat=pair[1], behav=law,
                                 CorpsAntagoniste="RBDY3", antagoniste="POLYR", colorAntagoniste=pair[0], alert=params["alert"])
        laws.append({"name": name, "law": params["joint_law"], "colours": list(pair), **cls, "eta": params["eta"]})
    # Unbonded chunk pairs: Coulomb contact with the smaller chunk-material friction.
    colour_mats = {}
    for i, c in enumerate(chunks):
        colour_mats.setdefault(colours[i], set()).add(c["material"])
    frict_laws = {}
    allc = sorted(colour_mats)
    for ca, cb in itertools.combinations_with_replacement(allc, 2):
        if (ca, cb) in pair_class or (ca in fixed_colours and cb in fixed_colours):
            continue
        mu = min(scene["materials"][m]["friction"] for m in colour_mats[ca] | colour_mats[cb])
        if mu not in frict_laws:
            nm = ("F%03d" % len(frict_laws)) + "x"
            frict_laws[mu] = pre.tact_behav(name=nm, law="IQS_CLB", fric=mu)
            tacts += frict_laws[mu]
            laws.append({"name": nm, "law": "IQS_CLB", "mu": mu, "colours": []})
        for x, y in ((ca, cb), (cb, ca)) if ca != cb else ((ca, cb),):
            svs += pre.see_table(CorpsCandidat="RBDY3", candidat="POLYR", colorCandidat=x, behav=frict_laws[mu],
                                 CorpsAntagoniste="RBDY3", antagoniste="POLYR", colorAntagoniste=y, alert=params["alert"])
        next(l for l in laws if l["name"] == frict_laws[mu].nom)["colours"].append([ca, cb])
    # Impactor: sphere (candidate) against chunk polyhedra.
    for col in allc:
        mu = min([scene["materials"][imp["material"]]["friction"]] + [scene["materials"][m]["friction"] for m in colour_mats[col]])
        nm = ("B%03d" % allc.index(col)) + "x"
        e = params["impact_restitution"]
        if e > 0.0:
            law = pre.tact_behav(name=nm, law="RST_CLB", rstn=e, rstt=0.0, fric=mu)
        else:
            law = pre.tact_behav(name=nm, law="IQS_CLB", fric=mu)
        tacts += law
        svs += pre.see_table(CorpsCandidat="RBDY3", candidat="SPHER", colorCandidat="IMPAC", behav=law,
                             CorpsAntagoniste="RBDY3", antagoniste="POLYR", colorAntagoniste=col, alert=params["alert"])
        laws.append({"name": nm, "law": "RST_CLB" if e > 0 else "IQS_CLB", "mu": mu, "restitution": e, "colours": [["IMPAC", col]]})

    datbox = deck_dir / "DATBOX"
    datbox.mkdir(parents=True, exist_ok=True)
    g = scene.get("gravity", [0.0, 0.0, -9.81])
    pre.writeDatbox(3, mats, mods, bodies, tacts, svs, datbox_dir=datbox, gravy=list(g), echo=False)

    dt = float(params["dt"])
    duration = float(scene["sim"]["duration"])
    sample = float(scene["sim"].get("sample_interval") or scene["sim"]["frame_dt"])
    sample_every = max(1, int(round(sample / dt)))
    n_steps = int(round(duration / dt))
    n_settle = int(round(float(params["settle_time"]) / dt))
    if not scene["sim"].get("gravity_prestress", False):
        n_settle = 0

    deck_files = sorted(p for p in datbox.iterdir() if p.is_file())
    deck_hash = hashlib.sha256()
    for p in deck_files:
        deck_hash.update(p.name.encode())
        deck_hash.update(p.read_bytes())

    manifest = {
        "scene": scene["name"],
        "scene_file": str(scene_path.resolve()),
        "scene_sha256": sha256_file(scene_path),
        "seed": scene["sim"].get("seed", 0),
        "params": params,
        "dt": dt,
        "n_settle": n_settle,
        "n_steps": n_steps,
        "sample_every": sample_every,
        "end_time": n_steps * dt,
        "body": body["name"],
        "chunk_body": lmgc_index,
        "chunk_colour": colours,
        "colour_signature": colour_sig,
        "chunk_support": [c["support"] for c in chunks],
        "chunk_center": [list(map(float, chunk_world(body, c)[0])) for c in chunks],
        "bonds": [{"a": b["a"], "b": b["b"]} for b in body["bonds"]],
        "impactor": {"name": imp["name"], "body": ball_index, "position": imp["position"],
                     "velocity": imp["velocity"], "angular_velocity": imp.get("angular_velocity", [0, 0, 0]),
                     "mass": imp["mass"], "radius": r, "density": ball_density},
        "probes": [p for p in scene.get("probes", [])],
        # Probes this oracle measures: impactor velocity along an axis.
        "probe_axes": [{"name": p["name"], "axis": p["axis"]} for p in scene.get("probes", [])
                       if p["type"] == "impactor_velocity" and p["impactor"] == imp["name"]],
        "laws": laws,
        "deck_sha256": deck_hash.hexdigest(),
        "notes": notes,
    }
    (deck_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    m = export(sys.argv[1], sys.argv[2])
    print("deck %s: %d bodies, %d laws, deck sha256 %s" % (sys.argv[2], len(m["chunk_body"]) + 1, len(m["laws"]), m["deck_sha256"]))
