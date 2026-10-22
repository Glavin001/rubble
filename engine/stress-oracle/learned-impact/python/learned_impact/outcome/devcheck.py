"""Device parity and throughput for remote training (Modal): the same trainer, any device.

- `step1_losses`: the per-term loss of a freshly seeded model (fixed weights, input
  standardisation from the same calls) on a fixed batch: the first `n_calls` routed calls in
  sorted event order. CPU, MPS and CUDA must agree to fp32 tolerance before a remote run is
  trusted.
- `throughput`: steps/s of the unchanged training step (no validation) for a model config.

    python -m learned_impact.outcome.devcheck --dataset <calls> --device cpu --out ref-cpu.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import losses as Lo
from . import schema as S
from .batch import HierarchyCache, collate
from .model import ImpactOutcomeModel, ModelConfig
from .train import Data, TrainConfig, _batches, input_stats, pick_device, train


def fixed_calls(dataset: str, n_calls: int) -> tuple[list[S.Call], dict]:
    paths = sorted(S.find_events(dataset))
    calls, structures = [], {}
    for p in paths:
        e = S.load_event(p)
        structures[e.manifest["event_id"]] = e.structures
        calls += [c for c in e.calls if c.routed]
        if len(calls) >= n_calls:
            break
    return calls[:n_calls], structures


def step1_losses(dataset: str, device: str, model_cfg: ModelConfig | None = None,
                 n_calls: int = 64, seed: int = 0) -> dict:
    calls, st = fixed_calls(dataset, n_calls)
    cfg = model_cfg or ModelConfig()
    torch.manual_seed(seed)
    model = ImpactOutcomeModel(cfg)
    model.set_input_stats(input_stats(calls, cfg.variant))
    dev = pick_device(device)
    model = model.to(dev)
    b = collate(calls, st, HierarchyCache(cfg.levels))
    lc = Lo.LossConfig()
    norms = Lo.zero_baselines([b], lc)
    bd = b.to(dev)
    with torch.no_grad():
        terms = Lo.loss_terms(model(bd), bd, lc)
        total, normed = Lo.total(terms, norms, lc)
    return {"device": str(dev), "n_calls": len(calls), "total": float(total),
            "terms": {k: float(v) for k, v in normed.items()}}


def compare(a: dict, b: dict, rtol: float = 2e-3, atol: float = 1e-5) -> list[str]:
    """Terms whose values differ beyond fp32 accumulation-order tolerance."""
    bad = []
    for k in ["total", *sorted(a["terms"])]:
        x = a["total"] if k == "total" else a["terms"][k]
        y = b["total"] if k == "total" else b["terms"].get(k, np.nan)
        if not np.isfinite(y) or abs(x - y) > atol + rtol * abs(x):
            bad.append(f"{k}: {x:.6g} vs {y:.6g}")
    return bad


def throughput(dataset: str, device: str, model_cfg: ModelConfig, steps: int, out: str,
               val_dataset: str | None = None) -> dict:
    """Steps/s of the unchanged trainer (validation off), excluding data loading."""
    t0 = time.time()
    data = Data(dataset, val_dataset=val_dataset)
    data.val = []
    load_s = time.time() - t0
    cfg = TrainConfig(dataset=dataset, out=out, steps=steps, device=device, model=model_cfg,
                      eval_every=10 ** 9, select_tau=False, pushforward_prob=0.3)
    t1 = time.time()
    train(cfg, data)
    train_s = time.time() - t1
    return {"device": str(pick_device(device)), "steps": steps, "load_s": load_s,
            "train_s": train_s, "steps_per_s": steps / train_s,
            "train_calls": len(data.train)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-calls", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--against", default=None, help="a reference JSON to compare with")
    a = ap.parse_args(argv)
    r = step1_losses(a.dataset, a.device, n_calls=a.n_calls)
    Path(a.out).write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    if a.against:
        bad = compare(json.loads(Path(a.against).read_text()), r)
        print("parity:", "OK" if not bad else bad)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
