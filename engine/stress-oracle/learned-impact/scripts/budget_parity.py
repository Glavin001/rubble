"""The event energy budget (schema.energy_budget) against the host's fixture (P0:
stress_learned::energy_budget, runs/fixtures/energy_budget_v1.json in its worktree: E_ref,
gain and remaining per call with their f64 bits). Bit for bit, or it prints the mismatches.

    python ../scripts/budget_parity.py <fixture.json>
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import schema as S  # noqa: E402

fx = json.load(open(sys.argv[1]))
assert fx["tol"] == S.ENERGY_BUDGET_TOL
bits = lambda x: struct.pack(">d", x).hex()  # noqa: E731
ev, bad = {}, 0
for c in fx["calls"]:
    if c["dir"] not in ev:
        ev[c["dir"]] = S.load_event(c["dir"])
    call = next(x for x in ev[c["dir"]].calls if x.call_index == c["call_index"])
    for k in ("E_budget_ref", "E_budget_gain", "E_budget_remaining"):
        if bits(call.energy[k]) != c[f"{k}_bits"]:
            bad += 1
            if bad < 6:
                print(c["event"], c["call_index"], k, call.energy[k], c[k])
print(f"{len(fx['calls'])} calls, {bad} mismatches")
