"""Plain-assert checks of the Kratos DEM exporter (no solver run).

    python -I oracles/kratos/test_export.py [scenes/b7_masonry_v04.json]
"""

import json
import math
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import export  # noqa: E402

ROOT = HERE.parents[1]


def main(scene_path):
    scene = json.loads(pathlib.Path(scene_path).read_text())
    body = scene["bodies"][0]
    chunks = body["chunks"]
    with tempfile.TemporaryDirectory() as tmp:
        man = export.export(scene_path, tmp)
        mats = json.loads((pathlib.Path(tmp) / "MaterialsDEM.json").read_text())
        mdpa = (pathlib.Path(tmp) / "wallDEM.mdpa").read_text()
        params = json.loads((pathlib.Path(tmp) / "ProjectParametersDEM.json").read_text())

    s, r = man["particle_size"], man["particle_radius"]
    pc = np.array(man["particle_chunk"])
    # Every chunk is filled exactly: particle count x cell volume = chunk volume; mass preserved.
    by_id = {m["material_id"]: m for m in mats["materials"]}
    for i, c in enumerate(chunks):
        vol = 8 * np.prod(c["half_extents"])
        cnt = int(np.sum(pc == i))
        assert abs(cnt * s ** 3 - vol) < 1e-12, (i, cnt)
        rho = by_id[man["chunk_colour"][i]]["Variables"]["PARTICLE_DENSITY"]
        m_particles = cnt * rho * 4.0 / 3.0 * math.pi * r ** 3
        m_chunk = vol * scene["materials"][c["material"]]["density"]
        assert abs(m_particles - m_chunk) < 1e-9 * m_chunk
    # Impactor mass.
    imp = scene["impactors"][0]
    ball = by_id[max(by_id)]["Variables"]
    assert abs(ball["PARTICLE_DENSITY"] * 4.0 / 3.0 * math.pi * imp["shape"]["radius"] ** 3 - imp["mass"]) < 1e-9
    # Fixed particles are exactly the particles of fixed chunks.
    fixed_chunks = {i for i, c in enumerate(chunks) if c["support"] == "fixed"}
    assert set(man["fixed_nodes"]) == {k + 1 for k in range(len(pc)) if pc[k] in fixed_chunks}
    # Each scene bond is represented by area / s^2 particle bonds.
    for b in body["bonds"]:
        key = "%d-%d" % (min(b["a"], b["b"]), max(b["a"], b["b"]))
        assert abs(man["particle_bonds_per_scene_bond"][key] * s * s - b["area"]) < 1e-12
    # Joint laws: per unit area, strength/energy equal the scene's; series compliance equals 1/(kn/A).
    A_b = math.pi * r * r
    k_in = man["intra_bond"]
    for law in man["joint_laws"]:
        j, pb = law["joint"], law["particle_bond"]
        if j["fixed"]:
            continue
        assert abs(pb["BOND_SIGMA_MAX"] * A_b / s ** 2 - j["ft"]) < 1e-9 * j["ft"]
        assert abs(pb["BOND_TAU_ZERO"] * A_b / s ** 2 - j["c"]) < 1e-9 * j["c"]
        assert abs(pb["FRACTURE_ENERGY_NORMAL"] * A_b / s ** 2 - j["Gt"]) < 1e-9 * j["Gt"]
        assert abs(pb["FRACTURE_ENERGY_TANGENTIAL"] * A_b / s ** 2 - j["Gs"]) < 1e-9 * j["Gs"]
        comp_n = pb["n_intra"] * s * s / k_in["kn"] + s * s / pb["kn"]
        comp_t = pb["n_intra"] * s * s / k_in["kt"] + s * s / pb["kt"]
        assert abs(comp_n * j["cn"] - 1.0) < 1e-9 and abs(comp_t * j["ct"] - 1.0) < 1e-9
        assert pb["softening_coeff_n"] <= 30 and pb["softening_coeff_t"] <= 30
    # Relations: (k, k) unbreakable, every pair of materials present.
    rel = {tuple(sorted(x["material_ids_list"])): x["Variables"] for x in mats["material_relations"]}
    ids = sorted(by_id)
    for a in ids:
        for b in ids:
            assert (min(a, b), max(a, b)) in rel
    for k in ids[:-1]:
        assert rel[(k, k)]["IS_UNBREAKABLE"] is True
    # Impactor relations: friction = min(impactor, chunk).
    for k in ids[:-1]:
        mu = rel[(k, ids[-1])]["STATIC_FRICTION"]
        assert mu == min(scene["materials"][imp["material"]]["friction"], scene["materials"]["brick"]["friction"])
    # Deck consistency.
    assert mdpa.count("\n") > len(pc)
    assert params["GravityZ"] == scene["gravity"][2]
    assert man["dt"] < man["dt_critical_estimate"]
    assert abs(man["n_steps"] * man["dt"] - scene["sim"]["duration"]) < man["dt"]
    print("kratos exporter OK: %d particles, %d joint laws, dt %.3g" % (len(pc) + 1, len(man["joint_laws"]), man["dt"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "scenes" / "b7_masonry_v04.json"))
