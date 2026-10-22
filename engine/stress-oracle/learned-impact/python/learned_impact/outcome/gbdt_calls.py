"""Per-bond gradient-boosted tree baseline on the calls (`learned-impact-outcome/4`), the
plan's baseline ladder: does a tree over the same physics features already break the right
bonds? Features per undirected bond: both twins' bond features, both endpoints' chunk
features, the call's global features and the QS-frame prior (`bond_qs_feat`). Label: the
oracle's break. Trained on the training split (all positives, a seeded share of the
negatives), thresholds chosen on validation, scored with the same bond P/R/F1 and the same
partition / fragment-size gates as the model (breaks only; the impactor keeps the oracle's
speed, so the ram gate is not meaningful here).

    python -m learned_impact.outcome.gbdt_calls --dataset <calls> --out gbdt.json
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
from .ceiling import targets_to_outputs
from .schema import Call
from .train import Data, _batches


def bond_features(c: Call) -> np.ndarray:
    a = c.arrays
    bf = a["bond_feat"]
    nb = len(a["bond_key"])
    if nb == 0:
        return np.zeros((0, 2 * bf.shape[1] + 2 * a["chunk_feat"].shape[1]
                         + a["global_feat"].shape[1] + 2 * a["bond_qs_feat"].shape[1]), np.float32)
    s, d = a["bond_src"][0::2], a["bond_dst"][0::2]
    cf = a["chunk_feat"]
    return np.concatenate([bf[0::2], bf[1::2], cf[s], cf[d],
                           np.repeat(a["global_feat"], nb, 0),
                           a["bond_qs_feat"][0::2], a["bond_qs_feat"][1::2]], 1).astype(np.float32)


def run(data: Data, neg_share: float = 0.1, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X, Y = [], []
    for c in data.train:
        x = bond_features(c)
        y = c.arrays["y_bond_broke"] > 0
        keep = y | (rng.random(len(y)) < neg_share)
        X.append(x[keep])
        Y.append(y[keep])
    X, Y = np.concatenate(X), np.concatenate(Y)
    w = np.where(Y, 1.0, 1.0 / neg_share)  # undo the negative subsampling
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                         random_state=seed)
    clf.fit(X, Y, sample_weight=w)
    cache = HierarchyCache(2)
    batches = [collate(bc, data.structures, cache) for bc in _batches(data.val, 20000, 64)]
    probs = [torch.from_numpy(clf.predict_proba(
        np.concatenate([bond_features(c) for c in b.calls]) if b.calls else np.zeros((0, X.shape[1])))[:, 1])
        for b in batches]
    res = {"train_rows": int(len(Y)), "train_positives": int(Y.sum()), "rules": {}}
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        rows, bp, by = [], [], []
        for b, p in zip(batches, probs):
            out = targets_to_outputs(b)
            brk = p >= t
            out["bond_log_q"] = torch.where(brk, 1.0, -1.0).double()
            dec = D.decode(out, b, D.DecodeConfig())
            rows.extend(M._per_call_outcomes(b, dec))
            bp.append(brk.numpy().astype(float))
            by.append(np.where(b.t["y_bond_has_freq"].numpy(), b.t["y_bond_freq"].numpy(),
                               (b.t["y_bond_broke"].numpy() > 0).astype(float)))
        s = M._summarise(rows, bp, by, [np.zeros(0)], [np.zeros(0)], 0, 1)
        res["rules"][f"GBDT p ≥ {t}"] = {k: s.metrics.get(k) for k in (
            "skill_over_bar_partition", "skill_over_bar_sizes", "pass_partition", "pass_sizes",
            "fragments_in_range", "bond_precision", "bond_recall", "bond_f1",
            "false_break_rate")}
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Data(a.dataset))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["rules"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
