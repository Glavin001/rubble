"""Reference Python of the event energy budget (`stress_learned::energy_budget`): over an event's
routed calls in order, E_ref = the largest frame-start KE + E_elastic so far, gain = the signed
sum of the calls' changes of their island's KE + E_elastic + PE
(KE_out + E_elastic_end - KE_in - E_elastic_start - W_gravity); a call may add at most
remaining = max(0, TOL E_ref - gain) to its island's KE + E_elastic + PE. Bit-for-bit with the
Rust on `runs/fixtures/energy_budget_v1.json`."""
from __future__ import annotations

TOL = 0.05


def call_delta(e: dict) -> float:
    return e["KE_out"] + e["E_elastic_end"] - e["KE_in"] - e["E_elastic_start"] - e["W_gravity"]


def ledger_of_calls(energies: list[dict]) -> list[tuple[float, float, float]]:
    """(E_ref, gain, remaining) before each call (after its open), from the calls' energy scalars."""
    e_ref, gain, out = 0.0, 0.0, []
    for e in energies:
        e_ref = max(e_ref, e["KE_in"] + e["E_elastic_start"])
        out.append((e_ref, gain, max(TOL * e_ref - gain, 0.0)))
        gain += call_delta(e)
    return out
