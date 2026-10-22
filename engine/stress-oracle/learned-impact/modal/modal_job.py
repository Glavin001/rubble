"""The job bodies `train_modal.py` runs in its GPU containers, without importing Modal, so
the exact code path is testable locally on the CPU (`python/tests/test_modal_job.py`)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive: Path, sha: str, dst_root: Path) -> Path:
    """Verify the archive's hash, then extract it once under `dst_root/<stem>`."""
    got = sha256(archive)
    if got != sha:
        raise RuntimeError(f"{archive}: sha256 {got} != {sha}")
    dst = dst_root / archive.stem
    if not (dst / ".ok").exists():
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(dst, filter="data")
        (dst / ".ok").write_text(sha)
    return dst


def train_job(data: Path, out: Path, argv: list[str], max_seconds: int, device: str) -> dict:
    """The unchanged trainer CLI on `data/train` with validation on `data/val`, under a hard
    timeout; stdout and stderr go to `out/stdout.log`."""
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "learned_impact.outcome.train", "--dataset", str(data / "train"),
           "--val-dataset", str(data / "val"), "--out", str(out), "--device", device, *argv]
    with open(out / "stdout.log", "w") as log:
        try:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=max_seconds).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    tail = (out / "stdout.log").read_text().splitlines()[-20:]
    return {"rc": rc, "seconds": time.time() - t0, "tail": tail}
