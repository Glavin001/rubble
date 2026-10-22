"""The event energy budget (schema.energy_budget) against the host's fixture (P0:
stress_learned::energy_budget, runs/fixtures/energy_budget_v1.json in its worktree: E_ref,
gain and remaining per call with their f64 bits). Bit for bit, or it prints the mismatches.

    python ../scripts/budget_parity.py <fixture.json>            (energy_budget_v1.json)
    python ../scripts/budget_parity.py <fixture.jsonl>           (contact_bounds_v1.jsonl: the
        ledger with contact energy, from each line's energy scalars; events start at frame 0)
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import schema as S  # noqa: E402

if sys.argv[1].endswith(".jsonl"):
    from types import SimpleNamespace as N

    unb = lambda h: struct.unpack(">d", bytes.fromhex(h))[0]  # noqa: E731
    lines = [json.loads(x) for x in open(sys.argv[1])]
    evs, bad = [], 0
    for ln in lines:
        if ln["frame"] == 0:
            evs.append([])
        evs[-1].append(ln)
    for ev in evs:
        calls = [N(routed=True, call_index=i, energy={k: unb(v) for k, v in ln["energy_bits"].items()})
                 for i, ln in enumerate(ev)]
        S.energy_budget(calls)
        for c, ln in zip(calls, ev):
            e = c.energy
            avail = S.budget_start(e) + max(e["E_budget_remaining"], 0.0)
            bad += struct.pack(">d", e["E_budget_remaining"]).hex() != ln["E_budget_remaining_bits"]
            bad += struct.pack(">d", avail).hex() != ln["e_avail_bits"]
    print(f"{len(lines)} calls in {len(evs)} events, {bad} mismatches")
    sys.exit(0)
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
