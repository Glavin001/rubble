"""Offline evaluation of a checkpoint on one split: skill metrics with the oracle-calibrated
gate, and the three-way report against `blast_parity` QS and the (CPU) punch on the same
calls. Per-call only: the closed-loop full-event gates need the rollout harness
(EVALUATION.md).

    python -m learned_impact.outcome.report --checkpoint ../runs/<job>/best.pt \
        --dataset <dataset> --split val --out ../runs/<job>/report-val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import decode as D
from . import metrics as M
from . import schema as S
from .batch import HierarchyCache, collate
from .train import _batches, load_model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "sealed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-sealed", action="store_true",
                    help="required for --split sealed: the sealed set is opened once, at P4")
    a = ap.parse_args(argv)
    if a.split == "sealed" and not a.confirm_sealed:
        ap.error("the sealed set stays closed until P4; pass --confirm-sealed to open it")
    model, ck = load_model(a.checkpoint, a.device)
    events = [S.load_event(p) for p in S.find_events(a.dataset)]
    events = [e for e in events if e.manifest["split"] == a.split
              and not e.manifest.get("truncated", False)]
    calls = [c for e in events for c in e.calls if c.routed]
    st = {e.manifest["event_id"]: e.structures for e in events}
    cache = HierarchyCache(model.cfg.levels)
    tc = ck["train_config"]
    batches = [collate(bc, st, cache) for bc in
               _batches(calls, tc.get("batch_budget", 20000), tc.get("batch_calls", 64))]
    dcfg = D.DecodeConfig(temperature=ck.get("temperature", 1.0),
                          energy_tolerance=tc["decode"]["energy_tolerance"])
    rep = M.three_way(model, batches, dcfg, ck.get("priors"))
    rep["checkpoint"] = str(a.checkpoint)
    rep["split"] = a.split
    rep["n_events"], rep["n_calls"] = len(events), len(calls)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for fam, e in sorted(rep["families"].items()):
        print(f"[report] {fam}: superiority={e.get('superiority')} "
              f"non_inferiority={e.get('non_inferiority', {}).get('pass')} model={e.get('model')}")
    print(f"[report] {len(rep['worse_than_qs'])} calls worse than QS (listed in {a.out})")


if __name__ == "__main__":
    main()
