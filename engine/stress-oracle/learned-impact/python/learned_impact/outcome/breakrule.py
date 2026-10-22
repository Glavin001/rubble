"""Break-decision analysis on validation (P2): which rule, at which operating point, and how
calibrated is p_b. Everything here is chosen on validation only.

For a checkpoint it reports, on the validation calls:
- the failure-margin threshold q̂ ≥ τ, swept over τ;
- the break-probability threshold p_b ≥ t, swept over t;
- hash-seeded Bernoulli sampling u(key, seed) < p_b (the plan's sampling ablation), several
  seeds;
- p_b's reliability (bins of p_b against the ensemble frequency where members are aligned,
  else the hard label), ECE and Brier.

Each rule is scored with bond P/R/F1 against the consensus and with the calibrated gates
on active calls (partition, fragment sizes); the impactor's speed is the model's own.

    python -m learned_impact.outcome.breakrule --checkpoint ../runs/<job>/best.pt \
        --dataset <calls> --out ../runs/<job>/breakrule.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import metrics as M
from .batch import HierarchyCache, collate
from .losses import prior_logit_shift
from .train import Data, _batches, load_model

GATE_KEYS = ("skill_over_bar_partition", "skill_over_bar_sizes", "skill_over_bar_ram",
             "pass_partition", "pass_sizes", "fragments_in_range", "bond_precision",
             "bond_recall", "bond_f1", "false_break_rate")


def _score(batches, outs, rule) -> dict:
    rows, bp, by = [], [], []
    for b, out in zip(batches, outs):
        brk = rule(b, out)
        o = dict(out)
        o["bond_log_q"] = torch.where(brk, 1.0, -1.0).to(out["bond_log_q"].dtype)
        dec = D.decode(o, b, D.DecodeConfig())
        rows.extend(M._per_call_outcomes(b, dec))
        bv = b.t["bond_valid"].numpy()
        bp.append(brk.numpy().astype(float)[bv])
        by.append(np.where(b.t["y_bond_has_freq"].numpy(), b.t["y_bond_freq"].numpy(),
                           (b.t["y_bond_broke"].numpy() > 0).astype(float))[bv])
    s = M._summarise(rows, bp, by, [np.zeros(0)], [np.zeros(0)], 0, 1)
    return {k: s.metrics.get(k) for k in GATE_KEYS}


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    for k in range(bins):
        sel = idx == k
        if sel.any():
            table.append({"bin": f"{k / bins:.1f}–{(k + 1) / bins:.1f}", "n": int(sel.sum()),
                          "mean_p": float(p[sel].mean()), "observed": float(y[sel].mean())})
    return {"ece": M.ece(p, y, bins), "brier": M.brier(p, y), "bins": table}


@torch.no_grad()
def analyse(ckpt: str | Path, data: Data, taus=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1),
            p_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5), seeds=(0, 1, 2)) -> dict:
    model, ck = load_model(ckpt, "cpu")
    model.eval()
    cache = HierarchyCache(model.cfg.levels)
    batches = [collate(bc, data.structures, cache) for bc in _batches(data.val, 20000, 64)]
    outs = [model(b) for b in batches]
    # The prior correction applies only to a model trained with prior reweighting (older
    # checkpoints stored priors regardless); the temperature is fitted here on validation,
    # since `best.pt` from older runs predates the end-of-training fit.
    reweight = ck["train_config"]["loss"].get("break_reweight", "prior")
    priors = ck.get("priors") if reweight == "prior" else None
    raw = []
    for b, o in zip(batches, outs):
        sh = prior_logit_shift(b, priors)
        lg = o["bond_break_logit"] + (sh if sh is not None else 0)
        bv = b.t["bond_valid"].numpy()
        raw.append((lg.numpy()[bv], np.where(b.t["y_bond_has_freq"].numpy(), b.t["y_bond_freq"].numpy(),
                                             (b.t["y_bond_broke"].numpy() > 0).astype(float))[bv]))
    temp = M.fit_temperature(np.concatenate([r[0] for r in raw]), np.concatenate([r[1] for r in raw]))

    def p_b(b, out):
        sh = prior_logit_shift(b, priors)
        lg = out["bond_break_logit"] + (sh if sh is not None else 0)
        return torch.sigmoid(lg / temp).double()

    res: dict = {"checkpoint": str(ckpt), "temperature": temp, "break_reweight": reweight,
                 "prior_correction": priors is not None, "rules": {}}
    for tau in taus:
        res["rules"][f"margin q̂ ≥ {tau}"] = _score(
            batches, outs, lambda b, o, tau=tau: o["bond_log_q"] >= float(np.log(tau)))
    for t in p_thresholds:
        res["rules"][f"p_b ≥ {t}"] = _score(batches, outs, lambda b, o, t=t: p_b(b, o) >= t)
    for s in seeds:
        res["rules"][f"Bernoulli on p_b (seed {s})"] = _score(
            batches, outs, lambda b, o, s=s: D.hash_uniform(b.t["bond_key"], s) < p_b(b, o))
    # reliability of p_b: against the ensemble frequency where aligned, else the hard label
    P, Y, PF, YF = [], [], [], []
    for b, o in zip(batches, outs):
        bv = b.t["bond_valid"].numpy()
        p = p_b(b, o).numpy()[bv]
        hf = b.t["y_bond_has_freq"].numpy()[bv]
        y = (b.t["y_bond_broke"].numpy() > 0).astype(float)[bv]
        f = b.t["y_bond_freq"].numpy()[bv]
        P.append(p[~hf])
        Y.append(y[~hf])
        PF.append(p[hf])
        YF.append(f[hf])
    res["reliability_hard_labels"] = reliability(np.concatenate(P), np.concatenate(Y))
    pf, yf = np.concatenate(PF), np.concatenate(YF)
    res["reliability_ensemble_freq"] = reliability(pf, yf) if len(pf) else None
    return res


def markdown(res: dict) -> str:
    lines = [f"Checkpoint `{res['checkpoint']}` (validation temperature {res['temperature']:.2f})",
             "", "| rule | " + " | ".join(GATE_KEYS) + " |",
             "|---|" + "---|" * len(GATE_KEYS)]
    for name, m in res["rules"].items():
        lines.append(f"| {name} | " + " | ".join(
            "—" if m.get(k) is None else f"{m[k]:.3f}" for k in GATE_KEYS) + " |")
    for key in ("reliability_ensemble_freq", "reliability_hard_labels"):
        r = res.get(key)
        if not r:
            continue
        lines += ["", f"**{key}**: ECE {r['ece']:.4f}, Brier {r['brier']:.4f}", "",
                  "| p_b bin | n | mean p_b | observed |", "|---|---|---|---|"]
        lines += [f"| {x['bin']} | {x['n']} | {x['mean_p']:.3f} | {x['observed']:.3f} |"
                  for x in r["bins"]]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = analyse(a.checkpoint, Data(a.dataset))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    Path(a.out).with_suffix(".md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
