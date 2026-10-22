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


def _oracle_breaks(c):
    return {k for k, b in zip(c.arrays["bond_key"].tolist(), c.arrays["y_bond_broke"].tolist())
            if b}


@pytest.mark.parametrize("seed", range(8))
def test_topology_with_oracle_breaks_is_identity(seed):
    _, (c0, c1), _ = make_event(seed)
    out = P.apply_topology(c0, c1, _oracle_breaks(c0))
    assert set(out.arrays) == set(c1.arrays)
    for k, v in c1.arrays.items():
        np.testing.assert_allclose(out.arrays[k].astype(float), v.astype(float), atol=1e-6,
                                   err_msg=k)


@pytest.mark.parametrize("seed", [3, 4, 8])
def test_topology_follows_the_model(seed):
    _, (c0, c1), _ = make_event(seed)
    ob = _oracle_breaks(c0)
    assert ob, "seed must break something"
    keys = c0.arrays["bond_key"].tolist()
    extra = [k for k in keys if k not in ob][:2]
    out = P.apply_topology(c0, c1, set(extra))  # misses every oracle break, adds two
    S.validate_call(out)
    live = set(out.arrays["bond_key"].tolist())
    assert ob <= live and not (set(extra) & live)
    assert not (ob & set(out.arrays["cracked_key"].tolist()))
    sel = np.isin(out.arrays["bond_key"], list(ob))
    assert np.all(out.arrays["y_bond_broke"][sel] == 1)
    assert np.all(out.arrays["y_bond_peak_util"][sel] >= 1)  # censored "already failed"
    # chunk live-bond counts follow the new bond list
    nc = out.counts["chunks"]
    live_deg = np.bincount(out.arrays["bond_src"], minlength=nc)[:nc] / 6
    np.testing.assert_allclose(out.arrays["chunk_feat"][:, S.col("chunk_feat", "n_live_bonds")],
                               live_deg, atol=1e-6)


def test_chained_parity_over_settling_frames():
    _, cs, _ = make_event(3, settle_calls=2)
    for a, b in zip(cs[:-1], cs[1:]):
        r = P.push(a, b, P.prediction_from_targets(a), _oracle_breaks(a))
        for arr in S.CARRIED_COLUMNS:
            np.testing.assert_allclose(r.arrays[arr], b.arrays[arr], atol=2e-6, err_msg=arr)


def test_topology_with_colliding_chunk_and_impactor_keys():
    """Impactor key 0 equals chunk key (0 << 32) | 0: the remap must not mix namespaces."""
    _, (c0, c1), _ = make_event(3)
    for c in (c0, c1):
        c.arrays = dict(c.arrays, impactor_key=np.array([0], np.int64))
    ob = _oracle_breaks(c0)
    out = P.apply_topology(c0, c1, set())  # the model keeps every oracle break's bond
    S.validate_call(out)
    assert ob <= set(out.arrays["bond_key"].tolist())
    nc = out.counts["chunks"]
    assert out.arrays["bond_src"].max() < nc and out.arrays["bond_dst"].max() < nc


def test_pushforward_keeps_only_samples_on_the_oracle_topology(tmp_path):
    """A successor is built only where the prediction breaks exactly the oracle's bonds
    (and its velocities are within tolerance); a different break set describes another wall
    than the target's history, so that sample is skipped."""
    import torch

    from learned_impact.outcome.batch import HierarchyCache, collate, pad_batch
    from learned_impact.outcome.ceiling import targets_to_outputs
    from learned_impact.outcome.synthetic import write_dataset
    from learned_impact.outcome.train import Data, TrainConfig, _successors

    write_dataset(tmp_path / "d", 6, val_fraction=0.01, settle_calls=2)
    data = Data(str(tmp_path / "d"))
    calls = [c for c in data.train if (c.event_id, c.call_index) in data.successor
             and (c.arrays["y_bond_broke"] > 0).any()][:3]
    assert calls
    cache = HierarchyCache(2)
    coll = lambda cs: pad_batch(collate(cs, data.structures, cache))  # noqa: E731
    b = coll(calls)
    cfg = TrainConfig()
    oracle = {k: v.float() if v.is_floating_point() else v for k, v in targets_to_outputs(b).items()}
    succ, bs, n, skipped = _successors(calls, b, oracle, data, cfg, coll)
    assert n == len(calls) and skipped == 0 and len(succ) == len(calls)
    # one extra break in the first call: that sample alone is excluded
    wrong = dict(oracle)
    lq = wrong["bond_log_q"].clone()
    first = int(torch.nonzero((b.t["bond_call"] == 0) & (b.t["y_bond_broke"] == 0))[0])
    lq[first] = 1.0
    wrong["bond_log_q"] = lq
    succ2, _, n2, skipped2 = _successors(calls, b, wrong, data, cfg, coll)
    assert n2 == len(calls) and skipped2 == 1 and len(succ2) == len(calls) - 1
    # velocities off by more than the tolerance: excluded too
    far = dict(oracle)
    far["chunk_out"] = oracle["chunk_out"] + 1.0
    _, _, n3, skipped3 = _successors(calls, b, far, data, cfg, coll)
    assert skipped3 == n3
