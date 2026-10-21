"""The output side can represent the oracle's answer: targets → raw head outputs → deployed
decode → the same outcome (on the synthetic toy, whose targets are physically consistent)."""

import numpy as np
import pytest

from learned_impact.outcome import ceiling as C
from learned_impact.outcome import state_io as IO
from tests.conftest import batch_of, make_calls


@pytest.mark.parametrize("kw", [dict(anchored=False), dict(anchored=True),
                                dict(anchored=False, gravity=True)])
def test_decode_ceiling_is_exact_on_consistent_targets(kw):
    calls, st = make_calls(range(10), **kw)
    rep = C.run_ceiling([batch_of(calls, st)])
    assert C.verdict(rep) == []
    m = rep["metrics_through_decode"]
    assert m["one_minus_ari"] == 0 and m["w1_log_mass"] == 0 and m["bond_f1"] == 1.0
    assert m["ram_err_rel"] < 1e-6


def test_verdict_flags_a_threshold_mismatch():
    calls, st = make_calls(range(10), anchored=False)
    for c in calls:  # a broken bond whose recorded peak demand is below τ
        a = c.arrays
        if a["y_bond_broke"].any():
            k = int(np.nonzero(a["y_bond_broke"])[0][0])
            a["y_bond_peak_util"] = a["y_bond_peak_util"].copy()
            a["y_bond_peak_util"][k] = 0.9
            break
    probs = C.verdict(C.run_ceiling([batch_of(calls, st)]))
    assert any("τ = 1" in p for p in probs)


def test_closed_loop_ceiling_reproduces_the_oracle_answer():
    calls, st = make_calls([3, 4, 8], anchored=False)
    for c in calls:
        doc = IO.from_targets(c)
        got = C.decode_ceiling(c, st[c.event_id], doc)
        IO.validate(got, c)
        assert {x["key"] for x in got["bonds_broken"]} == {x["key"] for x in doc["bonds_broken"]}
        assert [x["bin"] for x in sorted(got["bonds_broken"], key=lambda x: x["key"])] == \
            [x["bin"] for x in sorted(doc["bonds_broken"], key=lambda x: x["key"])]
        for part in ("chunks", "impactors"):
            for g, o in zip(got[part], doc[part]):
                for f in ("v", "w", "dx", "drot"):
                    np.testing.assert_allclose(g[f], o[f], atol=2e-5, err_msg=f"{part} {f}")
