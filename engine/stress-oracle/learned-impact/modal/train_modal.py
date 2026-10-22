"""Train the learned impact-outcome model on Modal (cloud CUDA): only GPU seconds are spent
there. Everything is prepared locally:
- the dataset is packed by `pack_data.py` (deterministic tar, sha256);
- the image pins torch and our package, and is built once per code change.

The GPU container unpacks the dataset (hash-verified) and runs the *unchanged* trainer with
`--device cuda` under a hard timeout. Checkpoints, `history.jsonl` and logs go to the
`learned-impact` Volume and are pulled back.

From the repo root (`modal.sh` = the pinned Modal CLI in Docker; the repo is mounted at /repo):

    M=engine/stress-oracle/learned-impact/modal/train_modal.py
    engine/tools/modal/modal.sh run $M::upload --name bulk-v1-calls3     # once per dataset
    engine/tools/modal/modal.sh run $M::run_bench --name bulk-v1-calls3 --gpu L4 --steps 150
    engine/tools/modal/modal.sh run $M::launch --plan <plan.json>       # budget-guarded
    engine/tools/modal/modal.sh run $M::pull --run <run>                # → runs/modal/<run>

Budget: `launch` and `bench` estimate each job's cost before starting (price × measured
seconds), refuse anything that would take the ledger (`findings/modal-spend.md`) past
CAP_USD, and append the measured cost of every finished job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
HERE = Path(__file__).resolve().parent if modal.is_local() else Path("/root")
LI = HERE.parent  # learned-impact/
CAP_USD = 10.0
# modal.com/pricing (2026-10), per second: GPU, plus CPU (per physical core) and memory
# (per GiB). The ledger records estimates at these rates with a 10% margin.
GPU_USD_S = {"T4": 0.000164, "L4": 0.000222, "A10G": 0.000306, "L40S": 0.000542,
             "A100-40GB": 0.000583, "H100": 0.001097}
CPU_USD_CORE_S = 0.0000131
MEM_USD_GIB_S = 0.00000222
CORES, MEM_GIB = 8, 64
MARGIN = 1.10
TORCH = "2.14.1"

app = modal.App("learned-impact-train")
vol = modal.Volume.from_name("learned-impact", create_if_missing=True)
image = (
    modal.Image.from_registry("debian:bookworm-slim", add_python="3.12")
    .pip_install(f"torch=={TORCH}", "numpy==2.5.4", "zstandard==0.25.0")
    .add_local_dir(str(LI / "python" / "learned_impact"), "/root/pkg/learned_impact", copy=True)
    .add_local_file(str(LI / "modal" / "ref" / "step1-cpu.json"), "/root/ref/step1-cpu.json", copy=True)
    .add_local_file(str(LI / "modal" / "modal_job.py"), "/root/modal_job.py", copy=True)
    .env({"PYTHONPATH": "/root/pkg"})
)


def usd_per_s(gpu: str) -> float:
    return GPU_USD_S[gpu] + CORES * CPU_USD_CORE_S + MEM_GIB * MEM_USD_GIB_S


# ---------------------------------------------------------------- ledger (local side)
LEDGER = LI / "findings" / "modal-spend.md"


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 6 and cells[1][:4].isdigit():
            total += float(cells[6].lstrip("$"))
    return total


def record(what: str, gpu: str, seconds: float, note: str) -> float:
    cost = seconds * usd_per_s(gpu) * MARGIN
    if not LEDGER.exists():
        LEDGER.write_text(
            "# Modal spend ledger (learned impact training)\n\n"
            f"Cap ${CAP_USD:.2f}. Cost = container seconds × (GPU + {CORES} cores + {MEM_GIB} GiB) "
            f"at modal.com/pricing (2026-10) × {MARGIN} margin; written by `train_modal.py`.\n\n"
            "| date | job | gpu | seconds | $/s | cost | note |\n|---|---|---|---|---|---|---|\n")
    with open(LEDGER, "a") as f:
        f.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {what} | {gpu} | {seconds:.0f} | "
                f"{usd_per_s(gpu):.6f} | ${cost:.4f} | {note} |\n")
    return cost


def guard(estimate_usd: float, what: str) -> None:
    s = spent()
    if s + estimate_usd > CAP_USD:
        raise SystemExit(f"budget: {what} estimated ${estimate_usd:.2f}, spent ${s:.2f}, "
                         f"cap ${CAP_USD:.2f}: refused")
    print(f"[budget] {what}: estimate ${estimate_usd:.2f}; spent ${s:.2f} of ${CAP_USD:.2f}")


# ---------------------------------------------------------------- remote side
import modal_job as J  # noqa: E402  (the job bodies, testable without Modal)


def _unpack(name: str, sha: str) -> Path:
    return J.unpack(Path("/vol/data") / f"{name}.tar", sha, Path("/tmp/data"))


GPU_FN = dict(image=image, volumes={"/vol": vol}, cpu=CORES, memory=MEM_GIB * 1024,
              retries=0, single_use_containers=True)


@app.function(image=image, volumes={"/vol": vol}, cpu=1, memory=2048, timeout=900)
def verify(name: str, sha: str) -> str:
    return "ok" if J.sha256(Path("/vol/data") / f"{name}.tar") == sha else "MISMATCH"


@app.function(gpu="L4", timeout=600, **GPU_FN)
def bench(name: str, sha: str, steps: int, env: dict) -> dict:
    """Parity (step-1 loss vs the local CPU reference) and throughput of the unchanged
    trainer step, plus the time of one validation pass, on this GPU."""
    t0 = time.time()
    os.environ.update(env)
    data = _unpack(name, sha)
    import torch
    from learned_impact.outcome import devcheck as DC
    from learned_impact.outcome import metrics as M
    from learned_impact.outcome.batch import HierarchyCache, collate, pad_batch
    from learned_impact.outcome.model import ImpactOutcomeModel, ModelConfig
    from learned_impact.outcome.train import Data, _batches

    gpu = torch.cuda.get_device_name(0)
    ref = json.loads(Path("/root/ref/step1-cpu.json").read_text())
    cuda = DC.step1_losses(str(data / "train"), "cuda", n_calls=ref["n_calls"])
    parity = DC.compare(ref, cuda)
    tp = DC.throughput(str(data / "train"), "cuda", ModelConfig(), steps, "/tmp/bench")
    # one validation pass (the trainer's evaluate on the full val split)
    d = Data(str(data / "train"), val_dataset=str(data / "val"))
    cache = HierarchyCache(2)
    vb = [pad_batch(collate(c, d.structures, cache)) for c in _batches(d.val, 20000, 64)]
    m = ImpactOutcomeModel(ModelConfig()).cuda()
    t1 = time.time()
    M.evaluate(m, vb)
    eval_s = time.time() - t1
    return {"gpu": gpu, "parity_failures": parity, "cuda_step1": cuda, **tp,
            "val_calls": len(d.val), "eval_pass_s": eval_s, "container_s": time.time() - t0}


@app.function(gpu="L4", timeout=24 * 3600, **GPU_FN)
def train_run(name: str, sha: str, run: str, argv: list[str], max_seconds: int, env: dict) -> dict:
    t0 = time.time()
    os.environ.update(env)
    data = _unpack(name, sha)
    out = Path("/tmp/out") / run
    r = J.train_job(data, out, argv, max_seconds, "cuda")
    dst = Path("/vol/runs") / run
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", f"{out}/.", str(dst)], check=True)
    vol.commit()
    return {"run": run, "rc": r["rc"], "container_s": time.time() - t0, "tail": r["tail"]}


# ---------------------------------------------------------------- local entrypoints
def _meta(name: str) -> dict:
    return json.loads((LI / "runs" / "modal" / f"{name}.json").read_text())


@app.local_entrypoint()
def upload(name: str):
    m = _meta(name)
    with vol.batch_upload(force=True) as up:
        up.put_file(str(LI / "runs" / "modal" / m["file"]), f"/data/{m['file']}")
    print(f"[upload] {m['file']} ({m['bytes'] / 1e9:.2f} GB): remote check", verify.remote(name, m["sha256"]))


@app.local_entrypoint()
def run_bench(name: str, gpu: str = "L4", steps: int = 150, accept_v3: bool = False):
    m = _meta(name)
    guard(120 * usd_per_s(gpu) * MARGIN, f"bench on {gpu} (≤ 120 s)")
    env = {"LEARNED_IMPACT_ACCEPT_V3": "1"} if accept_v3 else {}
    t0 = time.time()
    r = bench.with_options(gpu=gpu).remote(name, m["sha256"], steps, env)
    cost = record(f"bench {steps} steps", gpu, r["container_s"], f"{r['steps_per_s']:.2f} steps/s; "
                  f"eval {r['eval_pass_s']:.0f} s; parity {'OK' if not r['parity_failures'] else 'FAIL'}")
    print(json.dumps({k: v for k, v in r.items() if k != "cuda_step1"}, indent=1))
    print(f"[bench] wall {time.time() - t0:.0f} s, recorded ${cost:.4f}; ledger ${spent():.2f}")


@app.local_entrypoint()
def launch(plan: str, accept_v3: bool = False):
    """`plan`: JSON {"dataset": name, "gpu": "L4", "jobs": [{"run", "argv", "est_seconds"}]};
    est_seconds comes from the benchmark. Refused as a whole if it would pass the cap."""
    p = json.loads(Path(plan).read_text())
    m = _meta(p["dataset"])
    gpu = p["gpu"]
    est = sum(j["est_seconds"] for j in p["jobs"]) * usd_per_s(gpu) * MARGIN
    guard(est, f"{len(p['jobs'])} jobs on {gpu}")
    env = {"LEARNED_IMPACT_ACCEPT_V3": "1"} if accept_v3 else {}
    fn = train_run.with_options(gpu=gpu)
    calls = [(j, fn.spawn(p["dataset"], m["sha256"], j["run"], j["argv"],
                          int(j["est_seconds"] * 1.5), env)) for j in p["jobs"]]
    for j, c in calls:
        r = c.get()
        cost = record(f"train {j['run']}", gpu, r["container_s"], f"rc {r['rc']}")
        print(f"[launch] {j['run']}: rc {r['rc']}, {r['container_s']:.0f} s, ${cost:.3f}")


@app.local_entrypoint()
def pull(run: str):
    dst = LI / "runs" / "modal" / run
    dst.mkdir(parents=True, exist_ok=True)
    for entry in vol.listdir(f"/runs/{run}", recursive=True):
        if entry.type == modal.volume.FileEntryType.FILE:
            target = dst / Path(entry.path).relative_to(f"runs/{run}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                for chunk in vol.read_file(entry.path):
                    f.write(chunk)
    print(f"[pull] {run} → {dst}")
