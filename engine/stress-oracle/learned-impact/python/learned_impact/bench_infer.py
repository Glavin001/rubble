"""Inference latency vs building count (PyTorch CPU/MPS proxy for plan §9).

Runs the trained (or random) MGN-T forward on one recorded graph, repeated as N
independent buildings (batch-serial). Reports ms/building and total.

  python -m learned_impact.bench_infer --event /tmp/li-wall-set/wall_10mps_s02 \\
    --weights /tmp/li-train-walls --counts 1,5,10,20,50,100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .dataset import load_first_frame
from .normalize import normalize_batch
from .train import _make_model, param_count


@torch.no_grad()
def time_forward(model: torch.nn.Module, batch: dict, n_buildings: int, warmup: int, repeats: int) -> float:
    model.eval()
    for _ in range(warmup):
        model(batch)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for _ in range(n_buildings):
            model(batch)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Inference latency 1→N buildings")
    p.add_argument("--event", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--h", type=int, default=48)
    p.add_argument("--p", type=int, default=16)
    p.add_argument("--counts", type=str, default="1,5,10,20,50,100")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = p.parse_args(argv)

    if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    batch = normalize_batch(load_first_frame(args.event))
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    model = _make_model("mgn", args.h, args.p).to(device)
    if args.weights and (args.weights / "state_dict.pt").exists():
        model.load_state_dict(torch.load(args.weights / "state_dict.pt", map_location=device, weights_only=True))
    n_params, n_bytes = param_count(model)
    n_chunks = batch["chunk_mass"].shape[0]
    n_bonds = batch["bond_joint_state"].shape[0]
    print(
        f"device={device} params={n_params:,} ({n_bytes/1024:.0f} KiB) "
        f"graph chunks={n_chunks} bonds={n_bonds} event={args.event}"
    )

    counts = [int(x) for x in args.counts.split(",") if x.strip()]
    print(f"{'N':>6} {'total_ms':>10} {'ms/bldg':>10} {'bonds_active':>14}  note")
    plan_targets = {20: 4.0, 100: 8.0}  # ms targets from plan §9 (impacted / 30% active)
    for n in counts:
        total_s = time_forward(model, batch, n, args.warmup, args.repeats)
        total_ms = total_s * 1e3
        per = total_ms / n
        bonds = n * n_bonds
        note = ""
        if n in plan_targets:
            note = f"plan target ≤{plan_targets[n]:.0f} ms (GPU Slang est.; this is PyTorch {device})"
        print(f"{n:6d} {total_ms:10.2f} {per:10.3f} {bonds:14,d}  {note}")


if __name__ == "__main__":
    main()
