"""Categorise a report's `worse_than_qs` calls by cause (P2 analysis).

    cd <learned-impact/python matching the calls' schema> && \
    .venv/bin/python ../scripts/worse_than_qs.py --report <report.json> --checkpoint <best.pt> \
        --dataset <calls> --out <out.json>

For each listed call it re-runs the checkpoint and compares, bond by bond, the model's
breaks with the oracle run's, QS's and the ensemble frequency. Causes (first match):
- `extra_breaks_qs_none`: the model breaks bonds where the oracle and QS break none;
- `extra_breaks`: the model breaks bonds the oracle keeps (with oracle breaks elsewhere);
- `missed_breaks`: the model keeps bonds the oracle breaks;
- `wrong_bonds`: as many breaks as the oracle, but different bonds;
- `ambiguous`: the ensemble members disagree on those bonds (0 < frequency < 1);
- `ram`: worse on exit speed only.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome import schema as S
from learned_impact.outcome.batch import HierarchyCache, collate
from learned_impact.outcome.train import decode_config, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    want = collections.defaultdict(set)
    for w in rep["worse_than_qs"]:
        want[w["event"]].add(w["call"])
    model, ck = load_model(a.checkpoint)
    model.eval()
    dcfg = decode_config(ck)
    cache = HierarchyCache(model.cfg.levels)
    root = Path(a.dataset)
    # ensemble frequencies need all members: aggregate the ensembles that are involved
    from learned_impact.outcome.ensemble import aggregate
    ens_ids = {e.rsplit("_e", 1)[0] for e in want}
    events = [S.load_event(p) for p in sorted(root.iterdir())
              if p.is_dir() and p.name.rsplit("_e", 1)[0] in ens_ids]
    groups = collections.defaultdict(list)
    for e in events:
        for c in e.calls:
            c.arrays = dict(c.arrays)
        groups[e.manifest.get("ensemble_id")].append(e)
    for evs in groups.values():
        aggregate(evs)
    rows = []
    for e in events:
        eid = e.manifest["event_id"]
        for c in e.calls:
            if c.call_index not in want.get(eid, ()):
                continue
            b = collate([c], {eid: e.structures}, cache)
            with torch.no_grad():
                out = model(b)
            dec = D.decode(out, b, dcfg)
            mb = dec["bond_broke"].numpy().astype(bool)
            ob = c.arrays["y_bond_broke"] > 0
            qb = c.arrays.get("base_qs_bond_broke", np.zeros_like(ob)) > 0
            fr = c.arrays.get("y_bond_break_freq", ob.astype(float))
            extra, miss = mb & ~ob, ob & ~mb
            amb = (fr > 0) & (fr < 1)
            if extra.any() and not ob.any() and not qb.any():
                cause = "extra_breaks_qs_none"
            elif (extra | miss).any() and amb[extra | miss].all():
                cause = "ambiguous"
            elif extra.sum() > miss.sum():
                cause = "extra_breaks"
            elif miss.sum() > extra.sum():
                cause = "missed_breaks"
            elif extra.any():
                cause = "wrong_bonds"
            else:
                cause = "ram"
            lq = out["bond_log_q"].numpy()
            qy = c.arrays["y_bond_fail_margin"]
            rows.append({"extra_log_q_hat": [float(x) for x in lq[extra]],
                         "extra_oracle_q": [float(x) for x in qy[extra]],
                         "event": eid, "call": c.call_index, "family": c.family, "phase": c.phase,
                         "chunks": c.counts["chunks"], "bonds": int(len(ob)),
                         "oracle_breaks": int(ob.sum()), "model_breaks": int(mb.sum()),
                         "qs_breaks": int(qb.sum()), "extra": int(extra.sum()),
                         "missed": int(miss.sum()), "ambiguous_bonds": int(amb.sum()),
                         "cause": cause})
    summary = {}
    for fam in sorted({r["family"] for r in rows}):
        fr_ = [r for r in rows if r["family"] == fam]
        summary[fam] = {"n": len(fr_),
                        "by_cause": dict(collections.Counter(r["cause"] for r in fr_)),
                        "by_phase": dict(collections.Counter(r["phase"] for r in fr_)),
                        "median_extra": float(np.median([r["extra"] for r in fr_])),
                        "median_missed": float(np.median([r["missed"] for r in fr_])),
                        # how marginal the extra breaks are: the model's q̂ (τ = 1 at log 0)
                        # and the oracle's own failure margin q on those bonds
                        "extra_q_hat_quantiles": np.exp(np.quantile(
                            np.concatenate([r["extra_log_q_hat"] for r in fr_] or [[np.nan]]),
                            [0.1, 0.5, 0.9])).tolist(),
                        "extra_oracle_q_quantiles": np.quantile(
                            np.concatenate([r["extra_oracle_q"] for r in fr_] or [[np.nan]]),
                            [0.1, 0.5, 0.9]).tolist()}
    Path(a.out).write_text(json.dumps({"summary": summary, "calls": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
