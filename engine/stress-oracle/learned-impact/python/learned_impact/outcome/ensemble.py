"""Ensemble aggregation: writes `y_bond_break_freq`, `y_impactor_perforate_freq` and
`ens_impactor_v_end` into every member's ensemble-aligned calls (OUTCOME_SCHEMA.md,
"Ensembles"). The recorder may do the same; the arrays are identical either way.

Usage: python -m learned_impact.outcome.ensemble <dataset>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import schema as S

_ENS = ("y_bond_break_freq", "y_impactor_perforate_freq", "ens_impactor_v_end")


def _aligned(calls: list[S.Call]) -> bool:
    a = calls[0]
    for c in calls[1:]:
        if not (np.array_equal(np.sort(a.arrays["chunk_key"]), np.sort(c.arrays["chunk_key"]))
                and np.array_equal(np.sort(a.arrays["bond_key"]), np.sort(c.arrays["bond_key"]))
                and np.array_equal(np.sort(a.arrays["impactor_key"]),
                                   np.sort(c.arrays["impactor_key"]))):
            return False
    return True


def aggregate(events: list[S.Event]) -> int:
    """Fill the ensemble arrays in memory (members sorted by `ensemble_member`). Returns the
    number of aligned calls per member."""
    events = sorted(events, key=lambda e: e.manifest["ensemble_member"])
    by_index: dict[int, list[S.Call]] = defaultdict(list)
    for ev in events:
        for c in ev.calls:
            by_index[c.call_index].append(c)
    n_aligned = 0
    for _idx, calls in sorted(by_index.items()):
        if len(calls) != len(events) or not _aligned(calls):
            for c in calls:
                for k in _ENS:
                    c.arrays.pop(k, None)
            continue
        n_aligned += 1
        bkey = np.sort(calls[0].arrays["bond_key"])
        ikey = calls[0].arrays["impactor_key"]
        broke = np.zeros(len(bkey))
        perf = np.zeros(len(ikey))
        ens_v = []
        for c in calls:
            pos = np.searchsorted(bkey, c.arrays["bond_key"])
            broke[pos] += c.arrays["y_bond_broke"]
            ipos = [int(np.nonzero(c.arrays["impactor_key"] == k)[0][0]) for k in ikey]
            perf += (c.arrays["y_impactor_regime"][ipos] == S.REGIMES["perforate"])
            ens_v.append(c.arrays["y_impactor_v_end"][ipos])
        broke /= len(calls)
        perf /= len(calls)
        ens_v = np.stack(ens_v)  # [M, Ni(ref order), 3]
        for c in calls:
            pos = np.searchsorted(bkey, c.arrays["bond_key"])
            c.arrays["y_bond_break_freq"] = broke[pos].astype(np.float32)
            order = [int(np.nonzero(ikey == k)[0][0]) for k in c.arrays["impactor_key"]]
            c.arrays["y_impactor_perforate_freq"] = perf[order].astype(np.float32)
            c.arrays["ens_impactor_v_end"] = ens_v[:, order].astype(np.float32)
    return n_aligned


def aggregate_dataset(dataset: Path | str) -> dict[str, int]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in S.find_events(dataset):
        m = S.load_event(p, validate=False).manifest
        if m.get("ensemble_id"):
            groups[m["ensemble_id"]].append(p)
    out = {}
    for eid, paths in groups.items():
        events = [S.load_event(p) for p in paths]
        for ev in events:  # detach from the memmap before rewriting the files
            for c in ev.calls:
                c.arrays = {k: np.array(v) for k, v in c.arrays.items()}
            ev.structures = {k: {n: np.array(a) for n, a in s.items()}
                             for k, s in ev.structures.items()}
        out[eid] = aggregate(events)
        for ev in events:
            S.write_event(ev.root, ev.manifest, ev.calls, ev.structures)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    a = ap.parse_args()
    for eid, n in aggregate_dataset(a.dataset).items():
        print(f"[ensemble] {eid}: {n} aligned calls")


if __name__ == "__main__":
    main()
