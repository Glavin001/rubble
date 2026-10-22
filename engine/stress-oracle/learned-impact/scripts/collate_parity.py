"""Cached collate vs the per-batch reference on real calls (/5 full state included): every
batch tensor, edge, hierarchy array and loss bit-identical, over several groupings.

    python ../scripts/collate_parity.py <calls dir> [--events 40] [--groupings 4]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import losses as Lo, schema as S  # noqa: E402
from learned_impact.outcome.batch import HierarchyCache, clear_piece_cache, collate, pad_batch  # noqa: E402
from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig  # noqa: E402
from learned_impact.outcome.train import _batches  # noqa: E402
from tests.collate_reference import collate_reference  # noqa: E402
from tests.test_outcome_collate_parity import _equal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--events", type=int, default=40)
ap.add_argument("--groupings", type=int, default=4)
a = ap.parse_args()
ps = [p for p in S.find_events(a.dataset) if json.loads((Path(p) / "manifest.json").read_text())["split"] == "val"]
random.seed(0)
ps = random.sample(ps, min(a.events, len(ps)))
evs = [S.load_event(p, in_memory=True) for p in ps]
st = {e.manifest["event_id"]: e.structures for e in evs}
calls = [c for e in evs for c in e.calls if c.routed]
clear_piece_cache()
torch.manual_seed(0)
m = ImpactOutcomeModel(ModelConfig(full_state=True, break_time_head=True, hidden_head=True))
cfg = Lo.LossConfig(kinematic=True)
n = 0
for g in range(a.groupings):
    random.shuffle(calls)
    for bc in _batches(calls, 20000, 64):
        b1, b2 = collate(bc, st, HierarchyCache(2)), collate_reference(bc, st, HierarchyCache(2))
        _equal(b1, b2)
        p1, p2 = pad_batch(b1), pad_batch(b2)
        _equal(p1, p2)
        if g == 0 and n < 6:
            with torch.no_grad():
                t1, t2 = Lo.loss_terms(m(p1), p1, cfg), Lo.loss_terms(m(p2), p2, cfg)
            for k in t1:
                assert torch.equal(t1[k], t2[k]) or (torch.isnan(t1[k]) and torch.isnan(t2[k])), k
        n += 1
print(f"{len(calls)} calls ({sum(c.fs is not None for c in calls)} full-state), {n} batches over "
      f"{a.groupings} groupings: bit-identical batches, padded batches and losses")
