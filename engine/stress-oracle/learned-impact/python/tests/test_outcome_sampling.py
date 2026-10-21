"""Partitions keep recipes and layouts whole; sampling keeps every transition frame."""
from types import SimpleNamespace

import numpy as np
import pytest

from learned_impact.outcome_sampling import partition, sample_frames, transition_frames


def ev(split, layout, family="free_wall"):
    return SimpleNamespace(context={"split_group": split, "layout_group": layout, "family": family})


def test_partition_holds_out_whole_layouts_and_seals_families():
    events = [ev(f"r{i}", f"L{i % 7}") for i in range(70)] + [ev("m5", "m5", "m5_reference")]
    parts = partition(events, seed=3)
    assert [e.context["split_group"] for e in parts["sealed"]] == ["m5"]
    layouts = {k: {e.context["layout_group"] for e in v} for k, v in parts.items()}
    names = [k for k in layouts if k != "sealed"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not layouts[a] & layouts[b], f"layout shared by {a} and {b}"
    assert sum(len(v) for v in parts.values()) == len(events)


def test_a_split_group_never_spans_layouts():
    with pytest.raises(ValueError, match="spans layout groups"):
        partition([ev("r0", "L0"), ev("r0", "L1")])


def test_transitions_are_breaks_and_phase_changes():
    phases = ["pre_contact", "contact", "contact", "separation", "settling", "settling"]
    assert transition_frames(phases, [0, 0, 3, 0, 0, 0]) == [1, 2, 3, 4]


def test_sampling_keeps_every_transition_and_fills_quotas():
    frames = []
    for e in range(5):
        frames += [(e, 0, "pre_contact", 0), (e, 1, "contact", 12), (e, 2, "contact", 4)]
        frames += [(e, 3 + i, "settling", 0) for i in range(60)]
    rng = np.random.default_rng(0)
    chosen, rep = sample_frames(frames, 40, rng)
    for e in range(5):
        assert (e, 1) in chosen and (e, 2) in chosen and (e, 3) in chosen
    assert len(chosen) == 40 and rep["transitions"] == 15
    # Not uniform: settling (92% of raw frames) is held near its quota.
    assert rep["share"]["separation+settling"] < 0.8
    with pytest.raises(ValueError, match="smaller than the 15 transition"):
        sample_frames(frames, 10, rng)
