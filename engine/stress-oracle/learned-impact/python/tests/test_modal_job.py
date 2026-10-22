"""The Modal training path, run locally on the CPU: pack → verify and unpack → the unchanged
trainer under a timeout (the exact functions the GPU container calls)."""
import importlib.util
import json
from pathlib import Path

import pytest

MODAL_DIR = Path(__file__).resolve().parents[2] / "modal"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, MODAL_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pack_unpack_train_on_cpu(tmp_path):
    from learned_impact.outcome.synthetic import write_dataset

    write_dataset(tmp_path / "train", 6, val_fraction=0.01)
    write_dataset(tmp_path / "val", 6, seed=500, val_fraction=1.0)
    pack, job = _load("pack_data"), _load("modal_job")
    meta = pack.main(["--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
                      "--name", "toy", "--out-dir", str(tmp_path / "packed")])
    again = pack.main(["--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
                       "--name", "toy2", "--out-dir", str(tmp_path / "packed")])
    assert meta["sha256"] == again["sha256"]  # deterministic: same inputs, same archive
    with pytest.raises(RuntimeError):
        job.unpack(tmp_path / "packed" / "toy.tar", "0" * 64, tmp_path / "x")
    data = job.unpack(tmp_path / "packed" / "toy.tar", meta["sha256"], tmp_path / "unpacked")
    r = job.train_job(data, tmp_path / "out", ["--steps", "4", "--eval-every", "2", "--cost",
                                               "--batch-calls", "4"], 600, "cpu")
    assert r["rc"] == 0, r["tail"]
    assert (tmp_path / "out" / "best.pt").exists()
    rows = [json.loads(x) for x in (tmp_path / "out" / "history.jsonl").read_text().splitlines()]
    assert any(x["kind"] == "eval" for x in rows)
