"""Chaos-floor report over `p0_runs.py --chaos` output: `python -m learned_impact.p0_chaos OUT_DIR [JSON]`."""
import json
import sys

from .p0_metrics import candidate_vs, ensemble, load_ensembles, public, v0_of


def main():
    runs = load_ensembles(sys.argv[1])
    report = {}
    for scene, by in runs.items():
        report[scene] = {}
        for mode in ("kin", "kin_weibull"):
            if mode not in by:
                continue
            ens = ensemble(by[mode])
            entry = public(ens)
            if "ref" in by:
                entry["reference_run"] = candidate_vs(ens, by[mode], by["ref"], v0_of(by["ref"]))
            report[scene][mode] = entry
            r, f, l = ens["ram_vy"], ens["fragments"], ens["largest_fraction"]
            t = ens["typical_member_distance"]
            print(f"{scene:15s} {mode:12s} ram {r['median']:7.3f} [{r['min']:.3f}, {r['max']:.3f}] IQR {r['iqr']:.3f} | frag {f['median']:.0f} [{f['min']:.0f}, {f['max']:.0f}] | largest {l['median']:.3f} [{l['min']:.3f}, {l['max']:.3f}] | bonds ever {ens['bonds_ever_broken']} uncertain {ens['bonds_uncertain_0.1_0.9']} | typical 1-ARI {t['partition']:.3f} W1 {t['sizes']:.3f} dv {t['ram']:.3f}")
    if len(sys.argv) > 2:
        json.dump(report, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
