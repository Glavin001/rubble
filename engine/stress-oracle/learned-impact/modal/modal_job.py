"""The job bodies `train_modal.py` runs in its GPU containers, without importing Modal, so
the exact code path is testable locally on the CPU (`python/tests/test_modal_job.py`)."""

from __future__ import annotations

import hashlib
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive: Path, sha: str, dst_root: Path) -> Path:
    """Verify the archive's hash, then extract it (.tar or .tar.gz) once under
    `dst_root/<name without .tar[.gz]>`."""
    got = sha256(archive)
    if got != sha:
        raise RuntimeError(f"{archive}: sha256 {got} != {sha}")
    dst = dst_root / archive.name.removesuffix(".gz").removesuffix(".tar")
    if not (dst / ".ok").exists():
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(dst, filter="data")
        (dst / ".ok").write_text(sha)
    return dst


def train_job(data: Path, out: Path, argv: list[str], max_seconds: int, device: str) -> dict:
    """The unchanged trainer CLI on `data/train` with validation on `data/val`, under a hard
    timeout (killed, rc "timeout"). Its stdout and stderr go to `out/stdout.log`, each line
    prefixed by seconds since launch, which `timing` turns into the phase times."""
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "learned_impact.outcome.train", "--dataset", str(data / "train"),
           "--val-dataset", str(data / "val"), "--out", str(out), "--device", device, *argv]
    lines: list[tuple[float, str]] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump():
        with open(out / "stdout.log", "w", buffering=1) as log:
            for line in proc.stdout:
                t = time.time() - t0
                lines.append((t, line.rstrip("\n")))
                log.write(f"+{t:8.3f} {line}")

    # Utilisation samples, to tell a GPU-bound step from a CPU- or launch-bound one: the
    # GPU's busy fraction (nvidia-smi, when present) and the trainer's CPU seconds.
    util: list[tuple[float, float]] = []
    done = threading.Event()

    def sample():
        while not done.wait(0.5):
            try:
                v = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=5).stdout.split()
                util.append((time.time() - t0, float(v[0])))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                return

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    if shutil.which("nvidia-smi"):
        threading.Thread(target=sample, daemon=True).start()
    try:
        rc = proc.wait(timeout=max_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = "timeout"
    done.set()
    th.join(timeout=10)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    t = timing(lines)
    t["child_cpu_s"] = ru.ru_utime + ru.ru_stime  # all finished children of this process
    t["child_maxrss_gb"] = ru.ru_maxrss / (1e6 if sys.platform == "linux" else 1e9)  # largest
    if util and t.get("first_step_s") is not None:
        w = [u for ts, u in util if t["first_step_s"] <= ts <= t["first_step_s"] + t.get("train_window_s", 0)]
        t["gpu_util_train_mean"] = sum(w) / len(w) if w else None
        (out / "gpu_util.csv").write_text("".join(f"{ts:.2f},{u:.0f}\n" for ts, u in util))
    return {"rc": rc, "seconds": time.time() - t0, "timing": t,
            "tail": [x for _, x in lines[-20:]]}


STEP = re.compile(r"\] step (\d+)/(\d+) ")
EVAL = re.compile(r"\] val @ (\d+):")


def timing(lines: list[tuple[float, str]]) -> dict:
    """Phase times from the trainer's timestamped output: `startup_s` to its first line
    (imports, data load, model build), `first_step_s` to step 1's line (plus the loss
    baselines and the validation / curve batches), the steady training rate over log
    intervals without an evaluation (the first interval, with CUDA warm-up, apart), and the
    average extra time per evaluation pass."""
    first = next((t for t, x in lines if "] device=" in x), None)
    steps = [(t, int(m.group(1))) for t, x in lines if (m := STEP.search(x))]
    evals = [t for t, x in lines if EVAL.search(x)]
    r = {"startup_s": first, "first_step_s": steps[0][0] if steps else None,
         "last_step": steps[-1][1] if steps else 0, "evals": len(evals),
         "end_s": lines[-1][0] if lines else None}
    if len(steps) < 3:
        return r
    r["warmup_interval"] = {"steps": steps[1][1] - steps[0][1], "s": steps[1][0] - steps[0][0]}
    clean, dirty = [], []
    for (ta, sa), (tb, sb) in zip(steps[1:], steps[2:]):
        (dirty if any(ta < te <= tb for te in evals) else clean).append((sb - sa, tb - ta))
    if clean:
        n, s = sum(x for x, _ in clean), sum(y for _, y in clean)
        r["steady_steps"], r["steady_s"], r["steps_per_s"] = n, s, n / s
        if dirty:
            r["eval_extra_s"] = sum(y - x / r["steps_per_s"] for x, y in dirty) / len(dirty)
    tail_eval = [te for te in evals if te > steps[-1][0]]
    if tail_eval:
        r["final_eval_s"] = tail_eval[-1] - steps[-1][0]
    if r["end_s"] is not None:
        r["post_last_step_s"] = r["end_s"] - steps[-1][0]
    r["train_window_s"] = steps[-1][0] - steps[0][0]
    return r


def trial(archive: Path, sha: str, work: Path, out: Path, argv: list, max_seconds: int,
          device: str, ref: dict | None, env: dict[str, str]) -> dict:
    """The whole GPU-container body: verify and unpack the archive, check the step-1 loss of
    fixed weights on fixed calls against `ref` (the same function run locally on the CPU;
    training is skipped on a mismatch), then the trainer under its timeout."""
    import os

    t0 = time.time()
    os.environ.update(env)
    data = unpack(archive, sha, work)
    r: dict = {"unpack_s": time.time() - t0}
    if ref is not None:
        from learned_impact.outcome import devcheck as DC

        t1 = time.time()
        got = DC.step1_losses(str(data / "train"), device, n_calls=ref["n_calls"])
        r.update(step1=got, step1_ref=ref, parity_failures=DC.compare(ref, got),
                 parity_s=time.time() - t1)
        if r["parity_failures"]:
            r["rc"] = "parity"
            return r
    if argv and isinstance(argv[0], list):
        # several trainers as concurrent processes on the one GPU, into out/<i>
        res: list = [None] * len(argv)

        def one(i):
            res[i] = train_job(data, out / str(i), argv[i], max_seconds, device)

        ths = [threading.Thread(target=one, args=(i,)) for i in range(len(argv))]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        r.update(jobs=res, rc=0 if all(x["rc"] == 0 for x in res) else [x["rc"] for x in res],
                 timing=res[0]["timing"], seconds=max(x["seconds"] for x in res),
                 tail=res[0]["tail"])
    else:
        r.update(train_job(data, out, argv, max_seconds, device))
    return r
