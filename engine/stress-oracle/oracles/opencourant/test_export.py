"""Smoke test of the exporter on b1_bond_tension (no solver needed).

    python -I oracles/opencourant/test_export.py [scenes/b1_bond_tension.json]

Plain asserts, no pytest.  Checks the mesh/bond bookkeeping, the LAW169 card values
(strength, fracture energy, friction slope in CGS units), the shrunk cohesive gap,
supports, the distributed load and the probe mapping.
"""

import json
import math
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "scenes", "b1_bond_tension.json")
    scene = json.load(open(path))
    out = tempfile.mkdtemp(prefix="oc_export_test_")
    meta = export.export(scene, out, export.options_from(scene, {}))
    deck = open(os.path.join(out, "b1_bond_tension_0000.rad")).read()
    eng = open(os.path.join(out, "b1_bond_tension_0001.rad")).read()

    # 2 chunks x 2x2x2 hexes + 2x2 cohesive elements on the single bond
    assert meta["n_elements"] == 2 * 8 + 4, meta["n_elements"]
    assert meta["n_cohesive"] == 4
    body = meta["bodies"]["pair"]
    assert len(body["chunks"]) == 2 and len(body["bonds"]) == 1 and len(body["bonds"][0]) == 4
    assert all(len(c["nodes"]) == 27 for c in body["chunks"])
    assert sum(body["chunks"][0]["weights"]) == 64  # 8 hexes x 8 nodes
    assert body["supported"] == [0]

    # LAW169 card: f_t = 1 MPa = 1e7 dyn/cm^2, G_f = 100 J/m^2 = 1e5 erg/cm^2,
    # cohesion 1.5 MPa, G_II = 400 J/m^2, friction slope 0.6
    m = re.search(r"/MAT/LAW169/\d+\n.*\n.*\n(.*)\n.*\n(.*)\n.*\n(.*)\n", deck)
    assert m, "LAW169 card missing"
    rho = float(m.group(1))
    e, nu, sht, ten, gct = (float(x) for x in m.group(2).split())
    shr, gcs = (float(x) for x in m.group(3).split()[:2])
    assert math.isclose(ten, 1e6 * export.P, rel_tol=1e-9), ten
    assert math.isclose(gct, 100.0 * export.GF, rel_tol=1e-9), gct
    assert math.isclose(shr, 1.5e6 * export.P, rel_tol=1e-9), shr
    assert math.isclose(gcs, 400.0 * export.GF, rel_tol=1e-9), gcs
    assert math.isclose(sht, 0.6) and math.isclose(nu, 0.25)
    assert math.isclose(rho, 2000.0 * export.RHO, rel_tol=1e-9)
    # equal materials: the layer modulus equals the chunk modulus (series compliance)
    assert math.isclose(e, 1e9 * export.P, rel_tol=1e-9), e

    # cohesive gap: bonded faces pulled in by t_c/2 on both sides
    tc = body["cohesive_thickness"]
    assert math.isclose(tc, 0.2 * 0.05)
    nodes = {}
    for line in deck.split("/NODE\n", 1)[1].split("\n/")[0].splitlines():
        i, x, y, z = line.split()
        nodes[int(i)] = (float(x), float(y), float(z))
    xs_a = max(nodes[n][0] for n in body["chunks"][0]["nodes"]) / export.L
    xs_b = min(nodes[n][0] for n in body["chunks"][1]["nodes"]) / export.L
    assert math.isclose(xs_a, 0.05 - tc / 2) and math.isclose(xs_b, 0.05 + tc / 2), (xs_a, xs_b)
    # unbonded faces stay at their true positions
    assert math.isclose(min(nodes[n][0] for n in body["chunks"][0]["nodes"]) / export.L, -0.05)

    # supports, load, probes, engine
    assert "/BCS/" in deck and "   111 111" in deck
    groups = {}
    for gid, body_txt in re.findall(r"/GRNOD/NODE/(\d+)\n.*\n((?:[ \d]+\n)+)", deck):
        groups[int(gid)] = len(body_txt.split())
    cl = re.findall(r"/CLOAD/\d+\n.*\n.*\n(.*)", deck)
    total = sum(float(l.split()[-1]) * groups[int(l.split()[4])] for l in cl)
    assert math.isclose(total, 1.0 * export.F, rel_tol=1e-9), total  # nodal shares of a unit force
    assert meta["probes"]["axial"]["type"] == "section_force"
    assert len(meta["probes"]["axial"]["elements"]) == 4
    assert all(e["plus"] == "b" for e in meta["probes"]["axial"]["elements"])
    assert "/TH/BRIC/" in deck and "/TH/PART/" in deck
    assert "/RUN/b1_bond_tension/1" in eng and "/DTIX" in eng
    assert "/END" in deck
    shutil.rmtree(out)
    print(f"test_export: OK ({meta['n_nodes']} nodes, {meta['n_elements']} elements)")


if __name__ == "__main__":
    main()
