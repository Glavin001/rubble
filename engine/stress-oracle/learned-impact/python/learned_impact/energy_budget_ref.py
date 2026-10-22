"""Reference Python of the event energy budget (`stress_learned::energy_budget`): over an event's
routed calls in order, E_ref = the largest frame-start KE + E_elastic so far, gain = the signed
sum of the calls' changes of their island's KE + E_elastic + PE
(KE_out + E_elastic_end - KE_in - E_elastic_start - W_gravity, + E_contact_end - E_contact_start
when the call carries the contact scalars, and E_ref then reads + E_contact_start); a call may add at most
remaining = max(0, TOL E_ref - gain) to its island's KE + E_elastic + PE. Bit-for-bit with the
Rust on `runs/fixtures/energy_budget_v1.json`."""
from __future__ import annotations

import math

TOL = 0.05


def call_delta(e: dict) -> float:
    d = e["KE_out"] + e["E_elastic_end"] - e["KE_in"] - e["E_elastic_start"] - e["W_gravity"]
    if ("E_contact_start" in e) != ("E_contact_end" in e):
        raise ValueError("call energy has one of E_contact_start, E_contact_end")
    return d + e["E_contact_end"] - e["E_contact_start"] if "E_contact_start" in e else d


def call_start(e: dict) -> float:
    s = e["KE_in"] + e["E_elastic_start"]
    return s + e["E_contact_start"] if "E_contact_start" in e else s


def ledger_of_calls(energies: list[dict]) -> list[tuple[float, float, float]]:
    """(E_ref, gain, remaining) before each call (after its open), from the calls' energy scalars."""
    e_ref, gain, out = 0.0, 0.0, []
    for e in energies:
        e_ref = max(e_ref, call_start(e))
        out.append((e_ref, gain, max(TOL * e_ref - gain, 0.0)))
        gain += call_delta(e)
    return out


def energy_depth(e: float, k: float) -> float:
    """The depth at which a contact point of stiffness k holds the energy e: sqrt(2 e / k)
    (`contact_bound::energy_depth`); a contact may end at most max(depth_start, this) deep,
    e the island's available energy (KE + E_elastic + E_contact at the frame's start, in that
    order, + the remaining budget)."""
    return math.sqrt(2.0 * max(e, 0.0) / k)
