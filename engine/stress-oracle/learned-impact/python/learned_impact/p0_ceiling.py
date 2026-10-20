"""Perfect-model ceiling report: each handoff variant's ensemble against the oracle's ensemble
on the same perturbed members (`p0_runs.py` with `ceiling_ens_spec.json`).

    python -m learned_impact.p0_ceiling OUT_DIR [JSON]

Per scene, frame length and variant:
- the variant's ensemble summary (ram exit speed, fragments, largest fraction);
- the ensemble-level gate: the variant's median ram speed within max(IQR, 0.05 v0) of the
  oracle's median, its median fragment count and largest fraction inside the oracle's
  [min, max], and skill ratios (mean distance of a variant member to the oracle members
  over a typical oracle member's distance to the others) <= 1.25 for ram speed, partition
  (1 - ARI) and fragment sizes (W1 of log mass);
- the share of members that individually pass the P2 rule (`candidate_vs`);
- paired distances (variant member k against oracle member k: the same inputs).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from .p0_metrics import V0, candidate_vs, distances, ensemble, load, public, summary

SKILL_MAX = 1.25


def main():
    out = Path(sys.argv[1])
    groups = {}
    for p in sorted(out.glob("*.json")):
        m = re.match(r"(" + "|".join(sorted(V0, key=len, reverse=True)) + r")_(.+)_(1_\d+)_(\d+)$", p.stem)
        if not m:
            continue
        scene, variant, fdt, k = m.groups()
        groups.setdefault((scene, fdt), {}).setdefault(variant, {})[int(k)] = load(p)
    report = {}
    for (scene, fdt), by in sorted(groups.items()):
        oracle = [by["oracle"][k] for k in sorted(by["oracle"])]
        ens = ensemble(oracle)
        v0 = V0[scene]
        key = f"{scene}@{fdt}"
        report[key] = {"oracle": public(ens), "variants": {}}
        r = ens["ram_vy"]
        print(f"\n{key}: oracle ram {r['median']:.2f} [{r['min']:.2f}, {r['max']:.2f}] frag {ens['fragments']['median']:.0f} [{ens['fragments']['min']:.0f}, {ens['fragments']['max']:.0f}] largest {ens['largest_fraction']['median']:.3f} | typical 1-ARI {ens['typical_member_distance']['partition']:.3f} W1 {ens['typical_member_distance']['sizes']:.3f} dv {ens['typical_member_distance']['ram']:.2f}")
        for variant, runs in by.items():
            if variant == "oracle":
                continue
            ks = sorted(set(runs) & set(by["oracle"]))
            cand = [runs[k] for k in ks]
            per = [candidate_vs(ens, oracle, c, v0) for c in cand]
            skill = {m: float(np.mean([x["skill_ratio"][m] for x in per])) for m in ("ram", "partition", "sizes")}
            vs = {
                "ram_vy": summary([c["end"]["ram_vy"] for c in cand]),
                "fragments": summary([c["end"]["fragments"] for c in cand]),
                "largest_fraction": summary([c["end"]["largest_fraction"] for c in cand]),
            }
            ram_ok = abs(vs["ram_vy"]["median"] - r["median"]) <= max(r["iqr"], 0.05 * v0)
            frag_ok = ens["fragments"]["min"] <= vs["fragments"]["median"] <= ens["fragments"]["max"]
            largest_ok = ens["largest_fraction"]["min"] - 1e-9 <= vs["largest_fraction"]["median"] <= ens["largest_fraction"]["max"] + 1e-9
            skill_ok = all(v <= SKILL_MAX for v in skill.values())
            paired = [distances(runs[k], by["oracle"][k]) for k in ks]
            pd = {m: float(np.median([x[m] for x in paired])) for m in paired[0]}
            calls = float(np.mean([c.get("calls", 0) for c in cand]))
            entry = {
                "n": len(cand), "summary": vs, "ram_ok": ram_ok, "fragments_ok": frag_ok, "largest_ok": largest_ok, "skill": skill, "skill_ok": skill_ok,
                "within_spread": bool(ram_ok and frag_ok and largest_ok and skill_ok),
                "member_pass_rate": float(np.mean([x["within_spread"] for x in per])),
                "bond_f1_vs_consensus": float(np.mean([x["bond_f1"] for x in per])),
                "paired_median_distance": pd, "calls_per_run": calls,
            }
            report[key]["variants"][variant] = entry
            print(f"  {variant:15s} ram {vs['ram_vy']['median']:6.2f} [{vs['ram_vy']['min']:.2f}, {vs['ram_vy']['max']:.2f}] frag {vs['fragments']['median']:5.0f} [{vs['fragments']['min']:.0f}, {vs['fragments']['max']:.0f}] largest {vs['largest_fraction']['median']:.3f} | skill ram {skill['ram']:.2f} part {skill['partition']:.2f} size {skill['sizes']:.2f} | {'PASS' if entry['within_spread'] else 'fail'} (members {entry['member_pass_rate']:.0%}) | F1 {entry['bond_f1_vs_consensus']:.2f} | paired dv {pd['ram']:.2f} 1-ARI {pd['partition']:.2f} | calls {calls:.1f}")
    if len(sys.argv) > 2:
        json.dump(report, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
