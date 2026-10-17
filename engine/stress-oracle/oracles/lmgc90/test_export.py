"""Plain-assert checks of the LMGC90 exporter (no solver run).

    python -I oracles/lmgc90/test_export.py [scenes/b7_masonry_v04.json]
"""

import json
import math
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import export  # noqa: E402

ROOT = HERE.parents[1]


def main(scene_path):
    scene = json.loads(pathlib.Path(scene_path).read_text())
    body = scene["bodies"][0]
    with tempfile.TemporaryDirectory() as tmp:
        man = export.export(scene_path, tmp)
        datbox = pathlib.Path(tmp) / "DATBOX"
        bodies_dat = (datbox / "BODIES.DAT").read_text()
        tact = (datbox / "TACT_BEHAV.DAT").read_text()
        drv = (datbox / "DRV_DOF.DAT").read_text()

    n = len(body["chunks"])
    # One rigid body per chunk plus the impactor, in scene order.
    assert bodies_dat.count("RBDY3") == n + 1
    assert bodies_dat.count("POLYR") == n and bodies_dat.count("SPHER") == 1
    assert man["chunk_body"] == list(range(1, n + 1)) and man["impactor"]["body"] == n + 1
    # Impactor mass from the sphere volume and the exported density.
    imp = scene["impactors"][0]
    r = imp["shape"]["radius"]
    assert abs(man["impactor"]["density"] * 4.0 / 3.0 * math.pi * r ** 3 - imp["mass"]) < 1e-9
    # Fixed chunks: all six velocity DOFs driven.
    n_fixed = sum(c["support"] == "fixed" for c in body["chunks"])
    assert drv.count("$bdyty") == n_fixed, drv.count("$bdyty")

    laws = {l["name"]: l for l in man["laws"]}
    by_pair = {}
    for l in man["laws"]:
        if l["law"].endswith("CZM"):
            by_pair[tuple(sorted(l["colours"]))] = l
    col = man["chunk_colour"]
    fixed = [c["support"] == "fixed" for c in body["chunks"]]
    for b in body["bonds"]:
        pair = tuple(sorted((col[b["a"]], col[b["b"]])))
        if fixed[b["a"]] and fixed[b["b"]]:
            assert pair not in by_pair  # driven on both sides: no interaction at all
            continue
        l = by_pair[pair]
        m = scene["materials"][b["material"]]
        w = b["derived"]["weibull"]
        assert abs(l["cn"] - b["derived"]["kn"] / b["area"]) < 1e-6 * l["cn"]
        assert abs(l["ct"] - b["derived"]["ks"] / b["area"]) < 1e-6 * l["ct"]
        assert abs(l["s1"] - m["tensile_strength"] * w) < 1e-6 * l["s1"]
        assert abs(l["s2"] - m["cohesion"] * w) < 1e-6 * l["s2"]
        assert l["G1"] == m["fracture_energy"]["tension"] and l["G2"] == m["fracture_energy"]["shear"]
        assert l["mu"] == m["friction"]
    # Every law written to the deck, every bonded pair has a see table.
    for name in laws:
        assert name in tact
    assert tact.count("$seety") >= len(by_pair)
    # Fixed-fixed colour pairs never get a see table.
    fixed_cols = {c for i, c in enumerate(col) if fixed[i]} - {c for i, c in enumerate(col) if not fixed[i]}
    see_rows = [ln.split() for ln in tact.splitlines() if ln.strip().startswith("RBDY3")]
    assert see_rows
    for row in see_rows:
        assert not (row[2] in fixed_cols and row[6] in fixed_cols), row
    # Impactor laws: friction = min(impactor, chunk) friction.
    mu_imp = min(scene["materials"][imp["material"]]["friction"], min(scene["materials"][c["material"]]["friction"] for c in body["chunks"]))
    for l in man["laws"]:
        if l["colours"] and l["colours"][0][0] == "IMPAC":
            assert l["mu"] == mu_imp
    # Schedule.
    assert abs(man["n_steps"] * man["dt"] - scene["sim"]["duration"]) < 1e-9
    assert man["sample_every"] * man["dt"] <= (scene["sim"].get("sample_interval") or scene["sim"]["frame_dt"]) + 1e-12
    print("lmgc90 exporter OK: %d bodies, %d laws, %d bonds" % (n + 1, len(laws), len(body["bonds"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "scenes" / "b7_masonry_v04.json"))
