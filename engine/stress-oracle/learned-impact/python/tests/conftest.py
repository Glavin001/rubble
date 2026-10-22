import os

os.environ.setdefault("LEARNED_IMPACT_WANDB", "off")  # tracking off in tests (tracking test opts in)
import numpy as np
import pytest
import torch

from learned_impact.outcome import schema as S
from learned_impact.outcome.batch import HierarchyCache, collate
from learned_impact.outcome.synthetic import make_event


def make_calls(seeds, **kw):
    calls, structures = [], {}
    for s in seeds:
        m, cs, st = make_event(s, **kw)
        for c in cs:
            c.family, c.split, c.split_group = m["family"], m["split"], m["split_group"]
        calls.extend(cs)
        structures[m["event_id"]] = st
    return calls, structures


def batch_of(calls, structures, levels=2):
    return collate(calls, structures, HierarchyCache(levels))


@pytest.fixture
def synthetic_batch():
    calls, st = make_calls(range(4))
    return batch_of(calls, st), calls, st


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

__all__ = ["make_calls", "batch_of", "DEVICES", "S", "np"]
