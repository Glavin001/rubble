"""The decode's energy envelope against the host's parity fixture (P0: stress_learned::kin_envelope,
runs/fixtures/kin_envelope_v*.json in its worktree: per call E and every node's radii with their
f64 bits). Bit for bit, or it prints the mismatches.

    python ../scripts/envelope_parity.py <fixture.json>
"""
import json, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import decode as D, schema as S
from learned_impact.outcome.batch import HierarchyCache, collate
fx = json.load(open(sys.argv[1])); assert fx["tol"] == D.ENERGY_ENVELOPE_TOL
bits = lambda x: struct.pack(">d", x).hex()
ev = {}; bad = n = 0
for c in fx["calls"]:
    if c["dir"] not in ev: ev[c["dir"]] = S.load_event(c["dir"])
    e = ev[c["dir"]]; call = next(x for x in e.calls if x.call_index == c["call_index"])
    b = collate([call], {e.manifest["event_id"]: e.structures}, HierarchyCache(2))
    E, rx, rr = D.envelope_radii(b)
    if c["energy_bits"] is not None and bits(float(E[0])) != c["energy_bits"]: bad += 1; print("E", c["event"], c["call_index"], bits(float(E[0])), c["energy_bits"])
    radii = c["chunk_radii"] + c["impactor_radii"]
    for i, r in enumerate(radii):
        n += 1
        if r is None:
            if rx[i] == rx[i]: bad += 1; print("expected None", i)
            continue
        if bits(float(rx[i])) != r["dx_bits"] or bits(float(rr[i])) != r["drot_bits"]:
            bad += 1
            if bad < 5: print("radius", c["event"], c["call_index"], i, float(rx[i]), r["dx"], float(rr[i]), r["drot"])
print(f"{len(fx['calls'])} calls, {n} nodes, {bad} mismatches")
