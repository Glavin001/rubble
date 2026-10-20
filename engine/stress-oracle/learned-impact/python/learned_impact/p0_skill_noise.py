"""How far an oracle member itself strays on the skill ratio: leave each member out, score it
against the other 15 as a candidate. `python -m learned_impact.p0_skill_noise OUT_DIR... [JSON]`."""
import json
import re
import sys
from pathlib import Path

import numpy as np

from .p0_metrics import V0, candidate_vs, ensemble, load


def main():
    out = {}
    dirs = [a for a in sys.argv[1:] if not a.endswith(".json")]
    for d in dirs:
        for p in sorted(Path(d).glob("*_oracle_1_60_00.json")):
            m = re.match(r"(.+)_oracle_1_60_00$", p.stem)
            scene = m.group(1)
            runs = [load(q) for q in sorted(Path(d).glob(f"{scene}_oracle_1_60_[0-9][0-9].json"))]
            if len(runs) < 4 or scene in out:
                continue
            ratios = {"ram": [], "partition": [], "sizes": []}
            passes = 0
            for i in range(len(runs)):
                rest = runs[:i] + runs[i + 1:]
                e = ensemble(rest)
                c = candidate_vs(e, rest, runs[i], V0[scene])
                for k in ratios:
                    ratios[k].append(c["skill_ratio"][k])
                passes += c["within_spread"]
            out[scene] = {k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v))} for k, v in ratios.items()}
            out[scene]["member_pass_rate"] = passes / len(runs)
            print(f"{scene:16s} " + " | ".join(f"{k} mean {v["mean"]:.2f} median {v["median"]:.2f} p90 {v['p90']:.2f} max {v['max']:.2f}" for k, v in out[scene].items() if k != "member_pass_rate") + f" | members passing the per-member rule {out[scene]['member_pass_rate']:.0%}")
    js = [a for a in sys.argv[1:] if a.endswith(".json")]
    if js:
        json.dump(out, open(js[0], "w"), indent=1)


if __name__ == "__main__":
    main()
