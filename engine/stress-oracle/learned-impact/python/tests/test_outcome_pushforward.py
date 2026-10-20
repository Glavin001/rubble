"""Pushforward re-derives carried columns by the schema's direction convention: feeding the
oracle's own outcome must reproduce the recorded successor call."""

import numpy as np
import pytest

from learned_impact.outcome import pushforward as P
from learned_impact.outcome import schema as S
from learned_impact.outcome.synthetic import make_event


@pytest.mark.parametrize("seed", range(6))
def test_rewrite_from_oracle_outcome_reproduces_recorded_call(seed):
    _, (c0, c1), _ = make_event(seed, gravity=bool(seed % 2))
    r = P.rewrite_carried(c1, P.prediction_from_targets(c0))
    for arr in S.CARRIED_COLUMNS:
        np.testing.assert_allclose(r.arrays[arr], c1.arrays[arr], atol=2e-6, err_msg=arr)


def test_rewrite_contact_columns_from_start_state():
    _, (c0, _), _ = make_event(4)
    a = c0.arrays
    pred = {"chunk_v": dict(zip(a["chunk_key"].tolist(), a["chunk_v"])),
            "chunk_w": dict(zip(a["chunk_key"].tolist(), a["chunk_w"])),
            "imp_v": dict(zip(a["impactor_key"].tolist(), a["impactor_v"])),
            "imp_w": dict(zip(a["impactor_key"].tolist(), a["impactor_w"]))}
    r = P.rewrite_carried(c0, pred)
    np.testing.assert_allclose(r.arrays["impactor_contact_feat"], a["impactor_contact_feat"],
                               atol=2e-6)


def test_noise_touches_only_carried_columns():
    _, (c0, _), _ = make_event(2)
    n = P.add_noise(c0, 0.1, np.random.default_rng(0))
    for arr, cols in S.FEATURE_COLUMNS.items():
        if arr not in c0.arrays or not len(c0.arrays[arr]):
            continue
        carried = set(S.carried_indices(arr)) if arr in S.CARRIED_COLUMNS else set()
        for j in range(len(cols)):
            same = np.array_equal(n.arrays[arr][:, j], c0.arrays[arr][:, j])
            if j not in carried:
                assert same, (arr, cols[j])
