"""Frame sampling and partitions for `learned-impact-frames/1` datasets (plan P1).

- `partition`: train / validation / test by **layout group** (whole layouts held out) and
  never splitting a `split_group` (every descendant of one physical recipe stays with its
  parent). Families named in `sealed` go to the sealed set whole.
- `sample_frames`: frames by phase with set quotas, plus **every transition frame** (a
  break, or a change of phase) kept unconditionally, the rule of `causal_state.sample_case`
  (cace5923): it raises rather than drop a transition when the budget is too small.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

# Plan P1 "Balanced sampling across the life cycle".
DEFAULT_QUOTAS = {"contact": 0.35, "secondary": 0.20, "separation+settling": 0.25, "quiet": 0.20}
QUOTA_OF_PHASE = {"contact": "contact", "secondary": "secondary", "separation": "separation+settling", "settling": "separation+settling", "pre_contact": "quiet"}


def _unit(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") / 2**64


def partition(events, fractions=(0.8, 0.1, 0.1), seed=0, sealed=("m5_reference",)):
    """{'train'|'val'|'test'|'sealed': [event, ...]} for events with `.context`.

    The unit is the layout group; a split group spanning two layout groups is an error
    (its prefixes would straddle partitions)."""
    layout_of_split = {}
    for ev in events:
        sg, lg = ev.context["split_group"], ev.context["layout_group"]
        if layout_of_split.setdefault(sg, lg) != lg:
            raise ValueError(f"split group {sg} spans layout groups {layout_of_split[sg]} and {lg}")
    out = defaultdict(list)
    cut = np.cumsum(fractions) / np.sum(fractions)
    for ev in events:
        if ev.context["family"] in sealed:
            out["sealed"].append(ev)
            continue
        # The generator's own split (procedural.rs `split_of`) when the event records one.
        if ev.context.get("split"):
            out[ev.context["split"]].append(ev)
            continue
        u = _unit(ev.context["layout_group"], seed)
        out[["train", "val", "test"][int(np.searchsorted(cut, u, side="right"))]].append(ev)
    return dict(out)


def transition_frames(phases, breaks_per_frame):
    """Indices of frames with a break or a phase different from the previous frame's."""
    keep = []
    for i, (p, b) in enumerate(zip(phases, breaks_per_frame)):
        if b > 0 or (i > 0 and p != phases[i - 1]):
            keep.append(i)
    return keep


def sample_frames(frames, budget, rng, quotas=DEFAULT_QUOTAS):
    """Choose `budget` frames from `frames` (a list of (event_key, frame_index, phase,
    breaks)). Every transition frame is kept; the rest of the budget is split by phase
    quota (a quota a phase cannot fill passes to the others). Returns (chosen, report)."""
    by_event = defaultdict(list)
    for k, f, p, b in frames:
        by_event[k].append((f, p, b))
    must = set()
    for k, rows in by_event.items():
        rows.sort()
        idx = transition_frames([p for _, p, _ in rows], [b for _, _, b in rows])
        must.update((k, rows[i][0]) for i in idx)
    if len(must) > budget:
        raise ValueError(f"frame budget {budget} is smaller than the {len(must)} transition frames")
    pools = defaultdict(list)
    for k, f, p, b in frames:
        if (k, f) not in must:
            pools[QUOTA_OF_PHASE[p]].append((k, f))
    left = budget - len(must)
    want = {q: int(round(left * w)) for q, w in quotas.items()}
    chosen = set(must)
    spare = 0
    for q in sorted(quotas, key=lambda q: len(pools[q])):
        n = min(want[q] + spare, len(pools[q]))
        spare = want[q] + spare - n
        if n:
            pick = rng.choice(len(pools[q]), size=n, replace=False)
            chosen.update(pools[q][i] for i in pick)
    phase_of = {(k, f): p for k, f, p, _ in frames}
    got = Counter(QUOTA_OF_PHASE[phase_of[c]] for c in chosen)
    report = {"budget": budget, "chosen": len(chosen), "transitions": len(must), "by_quota": dict(got),
              "share": {q: got.get(q, 0) / max(len(chosen), 1) for q in quotas}}
    return sorted(chosen), report


def frames_of(events):
    """(event_key, frame, phase, breaks) rows of loaded `OutcomeEvent`s."""
    return [(ev.path.name, i, f["phase"], f["breaks"]) for ev in events for i, f in enumerate(ev.frames)]
