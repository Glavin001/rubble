"""The decode's energy envelope against the oracle's own frame outcomes: every recorded
(dx, drot) must lie inside it (0 violations), since it bounds what the oracle can do.

    python ../scripts/envelope_check.py <calls dir> [<calls dir> …]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import decode as D, schema as S  # noqa: E402
from learned_impact.outcome.batch import HierarchyCache, collate  # noqa: E402

cache = HierarchyCache(2)
for root in sys.argv[1:]:
    n = {"calls": 0, "unbounded": 0, "nodes": 0, "viol_dx": 0, "viol_drot": 0}
    worst = [0.0, 0.0]
    for p in S.find_events(root):
        e = S.load_event(p)
        calls = [c for c in e.calls if c.routed]
        if not calls:
            continue
        b = collate(calls, {e.manifest["event_id"]: e.structures}, cache)
        t = b.t
        nd = D.nodes(b)
        ph = dict(nd)
        ph.update(v_end=torch.cat([t["y_chunk_v_end"], t["y_imp_v_end"]]).double(),
                  w_end=torch.cat([t["y_chunk_w_end"], t["y_imp_w_end"]]).double(),
                  dx=torch.cat([t["y_chunk_dx"], t["y_imp_dx"]]).double(),
                  drot=torch.cat([t["y_chunk_drot"], t["y_imp_drot"]]).double())
        ext = D.causal_freedom(b)[1]
        cl, k, unb = D.energy_envelope(ph, b, ext, D.ENERGY_ENVELOPE_TOL)
        G = len(calls)
        n["calls"] += G
        n["unbounded"] += int(unb[:G].sum())
        free = ~ph["anchored"]
        n["nodes"] += int(free.sum())
        for i, key in enumerate(("dx", "drot")):
            moved = (cl[key] - ph[key]).norm(dim=-1)
            hit = moved > 0
            n[f"viol_{key}"] += int(hit.sum())
            if hit.any():
                worst[i] = max(worst[i], float((ph[key].norm(dim=-1)[hit] / cl[key].norm(dim=-1)[hit]).max()))
    print(root, n, "worst |x|/bound dx %.4g drot %.4g" % tuple(worst), flush=True)
