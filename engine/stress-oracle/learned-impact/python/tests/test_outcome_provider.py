"""The provider against a simulated harness speaking PROVIDER_PROTOCOL's files protocol."""

import json
import threading
import time

import pytest
import torch

from learned_impact.outcome import schema as S
from learned_impact.outcome import state_io as IO
from learned_impact.outcome.provider import Provider, ProviderError
from learned_impact.outcome.synthetic import make_event


def _request(root, name, call, structures, with_record=True, strip_targets=False):
    d = root / "req" / name
    d.mkdir(parents=True)
    oracle = IO.from_targets(call)
    if with_record:
        if strip_targets:
            call = S.Call(**{**call.__dict__, "arrays": {
                k: v for k, v in call.arrays.items() if not S._is_target(k)}, "energy": {}})
        m = {"event_id": call.event_id, "family": "f", "split_group": "g", "split": "test",
             "frame_dt": 1 / 60, "available": {k: False for k in S.AVAILABLE_FLAGS}}
        S.write_event(d / "call", m, [call], structures)
    a = call.arrays
    req = {"schema_version": "learned-impact-provider-request/0", "event_id": call.event_id,
           "frame": call.frame, "island": 0, "time": 0.0, "frame_dt": 1 / 60,
           "keys": {"chunks": a["chunk_key"].tolist(), "bonds": a["bond_key"].tolist(),
                    "impactors": a["impactor_key"].tolist()},
           "state": {}, "call_record": "call" if with_record else None}
    (d / "request.json").write_text(json.dumps(req))
    (d / "oracle.json").write_text(json.dumps(oracle))
    (d / "ready").write_text("")
    return d


def _ckpt(tmp_path):
    from learned_impact.outcome.model import ImpactOutcomeModel
    from learned_impact.outcome.train import TrainConfig, save

    torch.manual_seed(0)
    m = ImpactOutcomeModel()
    p = tmp_path / "m.pt"
    save(p, m, TrainConfig(), {}, {}, 0, {})
    return p


@pytest.mark.parametrize("mode", ["model", "echo", "targets"])
def test_provider_answers_every_request(tmp_path, mode):
    _, calls, st = make_event(5)
    for i, c in enumerate(calls):
        _request(tmp_path, f"{c.frame:06d}_0", c, st, strip_targets=(mode == "model"))
    prov = Provider(tmp_path, mode, str(_ckpt(tmp_path)) if mode == "model" else None)
    th = threading.Thread(target=prov.run, kwargs={"idle_timeout_s": 20}, daemon=True)
    th.start()
    for c in calls:  # the harness waits for each response
        p = tmp_path / "resp" / f"{c.frame:06d}_0.json"
        t0 = time.time()
        while not p.exists():
            assert time.time() - t0 < 20, "no response"
            time.sleep(0.01)
        doc = IO.read(p, c)
        if mode in ("echo", "targets"):
            assert doc == IO.from_targets(c)
        else:
            assert doc["meta"]["sha256"]
    (tmp_path / "done").write_text("")
    th.join(timeout=20)
    assert prov.n == len(calls)


def test_model_mode_refuses_a_request_without_call_record(tmp_path):
    _, calls, st = make_event(5)
    d = _request(tmp_path, "000010_0", calls[0], st, with_record=False)
    prov = Provider(tmp_path, "model", str(_ckpt(tmp_path)))
    with pytest.raises(ProviderError, match="call_record is null"):
        prov.answer(d)
