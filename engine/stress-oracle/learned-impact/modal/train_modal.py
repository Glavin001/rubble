"""Train the learned impact-outcome model on Modal (cloud CUDA): only GPU seconds are spent
there. Everything is prepared locally, as in `engine/tools/modal/cuda_tests.py`:

- the dataset is packed by `pack_data.py` into one deterministic `.tar.gz` (sha256) and
  uploaded once to the `learned-impact` Volume (`upload` skips it when the remote copy
  already has that hash);
- the step-1 CPU reference (fixed weights, fixed calls) is computed locally with devcheck on
  the archive's own contents;
- the image pins Python, the CUDA torch wheel and the package's deps (built once; only the
  thin package layer changes with the code).

The GPU container only executes `modal_job.trial`: hash-verified unpack, step-1 loss parity
against the CPU reference (no training on a mismatch), then the *unchanged* trainer CLI with
`--device cuda` under a hard timeout. Checkpoints, `history.jsonl`, `progress.log` and the
timestamped stdout go to the Volume (`/runs/<run>`) and are pulled to `runs/modal/<run>`.

From the repo root (`modal.sh` = the pinned Modal CLI in Docker; the repo is mounted at /repo):

    LI=engine/stress-oracle/learned-impact
    python3 $LI/modal/pack_data.py --train <calls-train> --val <calls-eval> --name sub100
    $LI/python/.venv/bin/python -m learned_impact.outcome.devcheck \\
        --dataset <unpacked sub100>/train --device cpu --out $LI/runs/modal/sub100-step1-cpu.json
    engine/tools/modal/modal.sh run $LI/modal/train_modal.py::probe        # builds the image
    engine/tools/modal/modal.sh run $LI/modal/train_modal.py::upload --name sub100
    engine/tools/modal/modal.sh run $LI/modal/train_modal.py::trial --name sub100 --gpu L4 \\
        --run l4-w64 --steps 600 --max-seconds 150 --accept-v3 --extra "--pushforward 0.3"
    engine/tools/modal/modal.sh run $LI/modal/train_modal.py::launch --plan <plan.json>
    engine/tools/modal/modal.sh run $LI/modal/train_modal.py::pull --run <run>

Budget: every paid entrypoint estimates its worst case (hard timeout + start-up allowance,
at the per-second price of GPU + reserved cores + memory, ×MARGIN), refuses it if the
ledger (`findings/modal-spend.md`) would pass CAP_USD, and appends the actual cost charged
on the *local wall time* of the call (an upper bound on the container's billed seconds).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/root")
LI = HERE.parent  # learned-impact/
RUNS = LI / "runs" / "modal"
CAP_USD = 2.0  # the trial's hard budget (2026-10-22); raise it deliberately for real runs
# modal.com/pricing and `modal billing rates` (2026-10-22), per second: GPU, plus CPU (per
# physical core) and memory (per GiB), billed on the reservation (or usage, if higher).
GPU_USD_S = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542,
             "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097}
CPU_USD_CORE_S = 0.0000131
MEM_USD_GIB_S = 0.00000222
CORES, MEM_GIB = 4, 8  # measured: sub100 loads in 1.3 GB RSS
MARGIN = 1.10
STARTUP_ALLOWANCE_S = 120  # cold start + unpack + parity, for the estimate only
TORCH, NUMPY, ZSTD = "2.14.1", "2.5.4", "0.25.0"  # = the local .venv

app = modal.App("learned-impact-train")
vol = modal.Volume.from_name("learned-impact", create_if_missing=True)
# Pinned base (not debian_slim(), whose distro follows the workspace's builder version).
# The PyPI torch wheel for linux x86_64 is the CUDA build (its CUDA runtime, cuDNN and
# cuBLAS come as nvidia-* wheels); Modal mounts the driver.
base = modal.Image.from_registry("debian:bookworm-slim", add_python="3.12")
image = (
    base.pip_install(f"torch=={TORCH}", f"numpy=={NUMPY}", f"zstandard=={ZSTD}")
    # torch 2.14 routes some eager CUDA ops (bmm outer products in autograd) to Triton
    # kernels, and Triton JIT-builds its driver shim with the system C compiler.
    .apt_install("gcc", "libc6-dev")
    .add_local_dir(str(LI / "python" / "learned_impact"), "/root/pkg/learned_impact", copy=True,
                   ignore=["**/__pycache__"])
    .add_local_file(str(HERE / "modal_job.py"), "/root/modal_job.py", copy=True)
    .env({"PYTHONPATH": "/root/pkg"})
)


def usd_per_s(gpu: str, cores: float = CORES, mem_gib: float = MEM_GIB) -> float:
    return GPU_USD_S[gpu] + cores * CPU_USD_CORE_S + mem_gib * MEM_USD_GIB_S


# ---------------------------------------------------------------- ledger (local side)
LEDGER = LI / "findings" / "modal-spend.md"
HEADER = ("| date (UTC) | job | gpu | estimate | wall s | $/s | cost | note |\n"
          "|---|---|---|---|---|---|---|---|\n")


def spent() -> float:
    """The sum of the ledger's cost column (rows whose first cell is a date)."""
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 8 and cells[1][:4].isdigit():
            total += float(cells[7].lstrip("$"))
    return total


def record(what: str, gpu: str, estimate: float, seconds: float, note: str,
           rate: float | None = None) -> float:
    rate = usd_per_s(gpu) if rate is None else rate
    cost = seconds * rate
    if not LEDGER.exists():
        LEDGER.write_text(f"# Modal spend ledger (learned impact)\n\nCap ${CAP_USD:.2f}.\n\n{HEADER}")
    with open(LEDGER, "a") as f:
        f.write(f"| {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} | {what} | {gpu} | "
                f"${estimate:.4f} | {seconds:.0f} | {rate:.6f} | ${cost:.4f} | {note} |\n")
    return cost


def guard(estimate_usd: float, what: str) -> None:
    s = spent()
    if s + estimate_usd > CAP_USD:
        raise SystemExit(f"budget: {what} estimated ${estimate_usd:.3f}, spent ${s:.3f}, "
                         f"cap ${CAP_USD:.2f}: refused")
    print(f"[budget] {what}: estimate ${estimate_usd:.3f}; spent ${s:.3f} of ${CAP_USD:.2f}")


# ---------------------------------------------------------------- remote side
CPU_RATE = 1 * CPU_USD_CORE_S + 2 * MEM_USD_GIB_S  # the 1-core, 2 GiB helpers below


@app.function(image=base, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=600)
def verify(file: str, sha: str) -> str:
    import hashlib

    vol.reload()  # a warm container still sees the Volume as of its start
    p = Path("/vol/data") / file
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return "ok" if h.hexdigest() == sha else "MISMATCH"


@app.function(image=image, cpu=1, memory=2048, timeout=600)
def env_info() -> dict:
    import platform

    import numpy
    import torch

    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": numpy.__version__}


GPU_FN = dict(image=image, volumes={"/vol": vol}, cpu=CORES, memory=MEM_GIB * 1024,
              retries=0, single_use_containers=True)


@app.function(gpu="L4", timeout=24 * 3600, **GPU_FN)
def train_run(file: str, sha: str, run: str, argv: list[str], max_seconds: int, env: dict,
              ref: dict | None = None) -> dict:
    """One training job: `modal_job.trial` on CUDA, outputs committed to /vol/runs/<run>."""
    t_entry = time.time()
    import modal_job as J
    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = Path("/tmp/out") / run
    r = J.trial(Path("/vol/data") / file, sha, Path("/tmp/data"), out, argv, max_seconds,
                "cuda", ref, env)
    t_copy = time.time()
    dst = Path("/vol/runs") / run
    dst.mkdir(parents=True, exist_ok=True)
    if out.exists():
        subprocess.run(["cp", "-r", f"{out}/.", str(dst)], check=True)
    (dst / "trial.json").write_text(json.dumps(r, indent=1, default=str))
    vol.commit()
    return {**r, "run": run, "gpu": smi, "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "t_entry": t_entry, "copy_s": time.time() - t_copy, "container_s": time.time() - t_entry}


# ---------------------------------------------------------------- local entrypoints
def _meta(name: str) -> dict:
    return json.loads((RUNS / f"{name}.json").read_text())


@app.local_entrypoint()
def probe():
    """Build the training image (timed) and report its torch / CUDA versions (1 CPU core)."""
    t0 = time.time()
    r = env_info.remote()
    wall = time.time() - t0
    record("probe (image build + env_info)", "cpu", 0.0, wall, json.dumps(r), CPU_RATE)
    print(json.dumps(r), f"wall {wall:.0f} s")


@app.local_entrypoint()
def upload(name: str):
    """Upload `runs/modal/<name>.tar.gz` once: skipped when the Volume already has that hash."""
    m = _meta(name)
    t0 = time.time()
    state = verify.remote(m["file"], m["sha256"])
    if state != "ok":
        with vol.batch_upload(force=True) as up:
            up.put_file(str(RUNS / m["file"]), f"/data/{m['file']}")
        state = verify.remote(m["file"], m["sha256"])
        print(f"[upload] {m['file']} ({m['bytes'] / 1e6:.1f} MB) in {time.time() - t0:.0f} s: {state}")
    else:
        print(f"[upload] {m['file']} already on the Volume with sha256 {m['sha256'][:12]}: skipped")
    record(f"upload {name}", "cpu", 0.0, time.time() - t0, f"remote hash {state}", CPU_RATE)
    if state != "ok":
        raise SystemExit("upload: remote hash check failed")


def _pull(run: str) -> Path:
    dst = RUNS / run
    dst.mkdir(parents=True, exist_ok=True)
    for entry in vol.listdir(f"/runs/{run}", recursive=True):
        if entry.type == modal.volume.FileEntryType.FILE:
            target = dst / Path(entry.path).relative_to(f"runs/{run}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
    return dst


@app.local_entrypoint()
def trial(name: str, run: str, gpu: str = "L4", steps: int = 600, eval_every: int = 0,
          max_seconds: int = 150, accept_v3: bool = False, extra: str = "", parity: bool = True,
          parallel: int = 1, cores: float = CORES, mem_gib: int = MEM_GIB):
    """One measured training job: step-1 parity vs `runs/modal/<name>-step1-cpu.json`, then
    the trainer (default model: W64, rounds 3,2,2,2,3) for `steps` under `max_seconds`.
    `--parallel N` runs seeds 0..N-1 as N concurrent trainers sharing the one GPU."""
    m = _meta(name)
    ref = json.loads((RUNS / f"{name}-step1-cpu.json").read_text()) if parity else None
    rate = usd_per_s(gpu, cores, mem_gib)
    est = (max_seconds + STARTUP_ALLOWANCE_S) * rate * MARGIN
    guard(est, f"trial {run} on {gpu} (≤ {max_seconds} s training)")
    argv = ["--steps", str(steps), "--eval-every", str(eval_every or steps), *shlex.split(extra)]
    if parallel > 1:
        argv = [[*argv, "--seeds", str(i)] for i in range(parallel)]
    env = {"LEARNED_IMPACT_ACCEPT_V3": "1"} if accept_v3 else {}
    t0 = time.time()
    try:
        r = train_run.with_options(gpu=gpu, timeout=max_seconds + 600, cpu=cores,
                                   memory=mem_gib * 1024).remote(
            m["file"], m["sha256"], run, argv, max_seconds, env, ref)
    except BaseException:
        record(f"trial {run} (FAILED)", gpu, est, time.time() - t0, "exception; see output", rate)
        raise
    wall = time.time() - t0
    r["local_wall_s"], r["cold_start_s"] = wall, r["t_entry"] - t0
    ts = [j["timing"] for j in r["jobs"]] if "jobs" in r else [r.get("timing", {})]
    rates = " + ".join(f"{t.get('steps_per_s', 0):.2f}" for t in ts)
    note = (f"rc {r.get('rc')}; parity {'OK' if not r.get('parity_failures') else 'FAIL'}; "
            f"{rates} steps/s steady; {ts[0].get('last_step', 0)} steps; "
            f"{cores:g} cores, {mem_gib} GiB")
    cost = record(f"trial {run}", gpu, est, wall, note, rate)
    dst = _pull(run)
    (dst / "trial-local.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({k: v for k, v in r.items() if k not in ("step1", "step1_ref", "tail")},
                     indent=1, default=str))
    print("\n".join(r.get("tail", [])))
    print(f"[trial] wall {wall:.0f} s, cold start {r['cold_start_s']:.1f} s, recorded ${cost:.4f}; "
          f"ledger ${spent():.3f} → {dst}")
    if r.get("rc") != 0:
        raise SystemExit(f"trial {run}: rc {r.get('rc')}")


@app.local_entrypoint()
def launch(plan: str, accept_v3: bool = False):
    """`plan`: JSON {"dataset": name, "gpu": "L4", "cores": 6, "mem_gib": 16,
    "jobs": [{"run", "argv", "max_seconds"}]}; `argv` is one trainer argument list or a list
    of them (concurrent trainers sharing the container's GPU, into <run>/<i>); max_seconds
    from the measured rate (`trial`). Refused as a whole if its worst case would pass the
    cap; jobs run in parallel containers."""
    p = json.loads(Path(plan).read_text())
    m = _meta(p["dataset"])
    gpu = p["gpu"]
    cores, mem_gib = p.get("cores", CORES), p.get("mem_gib", MEM_GIB)
    rate = usd_per_s(gpu, cores, mem_gib)
    ests = [(j["max_seconds"] + STARTUP_ALLOWANCE_S) * rate * MARGIN for j in p["jobs"]]
    guard(sum(ests), f"{len(p['jobs'])} jobs on {gpu}")
    env = {"LEARNED_IMPACT_ACCEPT_V3": "1"} if accept_v3 else {}
    t0 = time.time()
    calls = [(j, e, train_run.with_options(gpu=gpu, timeout=j["max_seconds"] + 600, cpu=cores,
                                           memory=mem_gib * 1024).spawn(
        m["file"], m["sha256"], j["run"], j["argv"], j["max_seconds"], env)) for j, e in zip(p["jobs"], ests)]
    for j, e, c in calls:
        r = c.get()
        # parallel jobs: charge each its own container time plus the shared start-up bound
        secs = r["container_s"] + max(0.0, r["t_entry"] - t0)
        cost = record(f"train {j['run']}", gpu, e, secs, f"rc {r['rc']}", rate)
        _pull(j["run"])
        print(f"[launch] {j['run']}: rc {r['rc']}, {secs:.0f} s, ${cost:.3f}")


@app.local_entrypoint()
def pull(run: str):
    print(f"[pull] {run} → {_pull(run)}")
