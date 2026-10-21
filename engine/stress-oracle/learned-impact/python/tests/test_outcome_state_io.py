import numpy as np
import pytest
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome import schema as S
from learned_impact.outcome import state_io as IO
from learned_impact.outcome.model import ImpactOutcomeModel
from tests.conftest import batch_of, make_calls


def test_decoded_states_roundtrip_and_partition(tmp_path):
    calls, st = make_calls(range(3))
    b = batch_of(calls, st)
    torch.manual_seed(0)
    m = ImpactOutcomeModel()
    with torch.no_grad():
        out = m(b)
        out["bond_log_r"] = torch.linspace(-1, 1, out["bond_log_r"].shape[0])  # some breaks
        dec = D.decode(out, b)
    docs = IO.decoded_states(b, out, dec, {"checkpoint": "test"})
    assert len(docs) == len(calls)
    for d, c in zip(docs, calls):
        p = IO.write(tmp_path / f"{c.event_id}_{c.call_index}.json", d)
        back = IO.read(p, c)
        assert back == d  # exact f64 round trip
    assert any(d["bonds_broken"] for d in docs)


def test_from_targets_is_the_oracle_outcome():
    calls, _ = make_calls([3])
    c = calls[0]
    d = IO.from_targets(c)
    IO.validate(d, c)
    v = np.array([n["v"] for n in d["chunks"]])
    np.testing.assert_allclose(v, c.arrays["y_chunk_v_end"], atol=1e-6)
    assert len(d["bonds_broken"]) == int(c.arrays["y_bond_broke"].sum())


def test_validate_rejects_a_missing_bond():
    calls, _ = make_calls([3])
    c = calls[0]
    d = IO.from_targets(c)
    d["bond_damage"] = d["bond_damage"][1:]
    with pytest.raises(S.SchemaError):
        IO.validate(d, c)
