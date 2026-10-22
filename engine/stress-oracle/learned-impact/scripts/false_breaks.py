"""Profile the false breaks of an overfit run on its own training events (P2 analysis).

    cd learned-impact/python && LEARNED_IMPACT_ACCEPT_V3=1 .venv/bin/python ../scripts/false_breaks.py \
        --run ../runs/overfit-real/n10 --dataset <calls-v3-eval>

For every false break (the model breaks a bond the oracle keeps, decision q̂ ≥ 1): the
model's q̂ and p_b, the oracle's q, the phase, the bond's attributes, and whether the oracle
breaks the same bond later in the event (a timing error) or never (a decision error).
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
from learned_impact.outcome.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--checkpoint", default="last.pt")
a = ap.parse_args()
run = Path(a.run)
names = json.loads((run / "events.json").read_text())
model, ck = load_model(run / a.checkpoint)
model.eval()
cache = HierarchyCache(model.cfg.levels)
dcfg = D.DecodeConfig(tau=1.0)
col = lambda n: S.col("bond_feat", n)  # noqa: E731
rows = []
for name in names:
    e = S.load_event(Path(a.dataset) / name)
    broke_at = {}  # bond key -> first call index where the oracle breaks it
    for c in e.calls:
        for k, bk in zip(c.arrays["bond_key"].tolist(), (c.arrays["y_bond_broke"] > 0).tolist()):
            if bk and k not in broke_at:
                broke_at[k] = c.call_index
    for c in e.calls:
        if not c.routed:
            continue
        b = collate([c], {e.manifest["event_id"]: e.structures}, cache)
        with torch.no_grad():
            out = model(b)
        dec = D.decode(out, b, dcfg)
        pred = dec["bond_broke"].numpy().astype(bool)
        truth = c.arrays["y_bond_broke"] > 0
        fp = np.nonzero(pred & ~truth)[0]
        if not len(fp):
            continue
        lq = out["bond_log_q"].numpy()
        pb = torch.sigmoid(out["bond_break_logit"]).numpy()
        bf = c.arrays["bond_feat"][0::2]
        for i in fp:
            k = int(c.arrays["bond_key"][i])
            later = broke_at.get(k)
            rows.append({
                "event": name, "call": c.call_index, "phase": c.phase,
                "q_hat": float(np.exp(lq[i])), "p_b": float(pb[i]),
                "oracle_q": float(c.arrays["y_bond_fail_margin"][i]),
                "oracle_peak_util": float(c.arrays["y_bond_peak_util"][i]),
                "damage0": float(bf[i, col("damage")]), "crush0": float(bf[i, col("crush")]),
                "rebar": float(bf[i, col("rebar")]), "plug": float(bf[i, col("punch_plug_bond")]),
                "oracle_breaks_later_at": later,
                "kind": ("broken earlier?" if later is not None and later < c.call_index else
                         "timing (oracle breaks it later)" if later is not None else
                         "never broken by the oracle"),
            })
qh = np.array([r["q_hat"] for r in rows])
qo = np.array([r["oracle_q"] for r in rows])
pbs = np.array([r["p_b"] for r in rows])
summary = {
    "false_breaks": len(rows),
    "kind": dict(collections.Counter(r["kind"] for r in rows)),
    "phase": dict(collections.Counter(r["phase"] for r in rows)),
    "q_hat_quantiles_10_50_90": np.quantile(qh, [0.1, 0.5, 0.9]).tolist() if len(rows) else None,
    "oracle_q_quantiles_10_50_90": np.quantile(qo, [0.1, 0.5, 0.9]).tolist() if len(rows) else None,
    "p_b_quantiles_10_50_90": np.quantile(pbs, [0.1, 0.5, 0.9]).tolist() if len(rows) else None,
    "p_b_says_break": int((pbs >= 0.5).sum()),
    "damaged_before": int(sum(r["damage0"] > 0 for r in rows)),
    "rebar": int(sum(r["rebar"] > 0 for r in rows)), "plug": int(sum(r["plug"] > 0 for r in rows)),
    "per_event": dict(collections.Counter(r["event"] for r in rows)),
    "calls_with_fp": len({(r["event"], r["call"]) for r in rows}),
}
(run / "false_breaks.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary, indent=1))
