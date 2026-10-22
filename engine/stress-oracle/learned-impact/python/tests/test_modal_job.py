"""The Modal training path, run locally on the CPU: pack → verify and unpack → step-1 parity →
the unchanged trainer under a timeout (the exact functions the GPU container calls)."""
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
    from learned_impact.outcome import devcheck as DC
    from learned_impact.outcome.synthetic import write_dataset

    write_dataset(tmp_path / "train", 6, val_fraction=0.01)
    write_dataset(tmp_path / "val", 6, seed=500, val_fraction=1.0)
    pack, job = _load("pack_data"), _load("modal_job")
    meta = pack.main(["--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
                      "--name", "toy", "--out-dir", str(tmp_path / "packed")])
    again = pack.main(["--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
                       "--name", "toy2", "--out-dir", str(tmp_path / "packed")])
    assert meta["sha256"] == again["sha256"]  # deterministic: same inputs, same archive
    archive = tmp_path / "packed" / "toy.tar.gz"
    with pytest.raises(RuntimeError):
        job.unpack(archive, "0" * 64, tmp_path / "x")
    # the reference comes from the archive's own contents, as `train_modal.py::reference`
    local = job.unpack(archive, meta["sha256"], tmp_path / "ref")
    ref = DC.step1_losses(str(local / "train"), "cpu", n_calls=8)
    r = job.trial(archive, meta["sha256"], tmp_path / "unpacked", tmp_path / "out",
                  ["--steps", "60", "--eval-every", "30", "--cost", "--batch-calls", "4"],
                  600, "cpu", ref, {})
    assert r["parity_failures"] == [], r
    assert r["rc"] == 0, r["tail"]
    t = r["timing"]
    assert t["last_step"] == 60 and t["evals"] == 2 and t["steps_per_s"] > 0, t
    assert (tmp_path / "out" / "best.pt").exists()
    rows = [json.loads(x) for x in (tmp_path / "out" / "history.jsonl").read_text().splitlines()]
    assert any(x["kind"] == "eval" for x in rows)
    assert (tmp_path / "out" / "progress.log").exists()
    # a wrong reference stops before training
    bad = dict(ref, total=ref["total"] * 2)
    r2 = job.trial(archive, meta["sha256"], tmp_path / "unpacked", tmp_path / "out2", [], 600,
                   "cpu", bad, {})
    assert r2["rc"] == "parity" and not (tmp_path / "out2").exists()
