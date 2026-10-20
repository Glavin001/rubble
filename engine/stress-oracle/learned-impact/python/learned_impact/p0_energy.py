"""P0 step 0's ledger check on the pinned configuration: `python -m learned_impact.p0_energy OUT_DIR [JSON]`.

Per scene (the reference run and the kin ensemble): the balance residual
(E_mech + D - E0) / E0 at the run's end and its worst frame, the signed softening
overshoot (its negative part separately), the fracture-work floor (sum over broken bonds of
min(G_t, G_s) A) against what those bonds booked, and the dissipation's shares.
"""
import json
import sys

import numpy as np

from .p0_metrics import load_ensembles


def ledger(run):
    e0 = run["e0"]
    en = run["end"]["energy"]
    worst = max((abs(f["mechanical"] + f["dissipated"] - e0) for f in run["frames"]), default=0.0) / e0
    d = en["dissipated"]
    return {
        "residual_end": (en["mechanical"] + en["dissipated"] - e0) / e0,
        "residual_worst_frame": worst,
        "overshoot": en["softening_overshoot"],
        "overshoot_negative": en["softening_overshoot_negative"],
        "overshoot_negative_over_e0": en["softening_overshoot_negative"] / e0,
        "fracture_floor": en["fracture_floor_gf_area"],
        "broken_booked": en["broken_bonds_booked"],
        "floor_holds": en["broken_bonds_booked"] + 1e-9 >= en["fracture_floor_gf_area"],
        "floor_over_e0": en["fracture_floor_gf_area"] / e0,
        "share_contact": en["contact_dissipation"] / d if d else 0.0,
        "share_bond": en["bond_dissipation"] / d if d else 0.0,
        "share_overshoot": en["softening_overshoot"] / d if d else 0.0,
        "share_damping": en["damping_dissipation"] / d if d else 0.0,
        "external_work_over_e0": en["external_work"] / e0,
    }


def main():
    runs = load_ensembles(sys.argv[1])
    out = {}
    for scene, by in sorted(runs.items()):
        ref = ledger(by["ref"]) if "ref" in by else None
        mem = [ledger(r) for r in by.get("kin", [])]
        agg = {k: [float(np.min([m[k] for m in mem])), float(np.median([m[k] for m in mem])), float(np.max([m[k] for m in mem]))] for k in mem[0] if not isinstance(mem[0][k], bool)}
        agg["floor_holds_all"] = all(m["floor_holds"] for m in mem)
        out[scene] = {"reference": ref, "kin_min_median_max": agg}
        print(f"{scene:15s} resid ref {ref['residual_end']:+.4f} kin [{agg['residual_end'][0]:+.4f}, {agg['residual_end'][2]:+.4f}] | overshoot- ref {ref['overshoot_negative']:.3e} J kin min {agg['overshoot_negative'][0]:.3e} J ({agg['overshoot_negative_over_e0'][0]:.1e} E0) | floor {ref['fracture_floor']:.1f} J vs booked {ref['broken_booked']:.0f} J (floor {agg['floor_over_e0'][1]:.1e} E0, holds {agg['floor_holds_all']}) | shares contact {ref['share_contact']:.2f} bond {ref['share_bond']:.2f} overshoot {ref['share_overshoot']:.2f}")
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
