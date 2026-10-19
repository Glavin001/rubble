"""CLI: smoke, overfit, and MLP-vs-MGN-T probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .dataset import iter_event_frames, list_events, load_all_frames, load_first_frame, synthetic_batch
from .export import export_weights
from .losses import total_loss
from .model import BondMLPBaseline, LearnedImpactNet
from .normalize import normalize_batch
from .schema import load_event


def _make_model(arch: str, h: int, p: int) -> torch.nn.Module:
    if arch == "mlp":
        return BondMLPBaseline(h=h)
    model = LearnedImpactNet(h=h, p=p)
    if model.attn.heads > h:
        model.attn.heads = max(1, h // 2)
    return model


def _batches_for(dataset: Path, structure_type: str | None) -> list[dict[str, torch.Tensor]]:
    if structure_type:
        batches: list[dict[str, torch.Tensor]] = []
        for ev in list_events(dataset):
            man = load_event(ev).manifest
            if man.get("structure_type") == structure_type:
                batches.extend(iter_event_frames(ev))
        if not batches:
            raise FileNotFoundError(f"no events with structure_type={structure_type!r} in {dataset}")
        return batches
    return load_all_frames(dataset)


def run_smoke(event: Path | None, steps: int, h: int, p: int, out: Path) -> float:
    torch.manual_seed(0)
    if event is not None and event.exists():
        batch = normalize_batch(load_first_frame(event))
    else:
        batch = normalize_batch(synthetic_batch())
    model = _make_model("mgn", h, p)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    last = 0.0
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        parts = total_loss(pred, batch)
        parts["loss"].backward()
        opt.step()
        last = float(parts["loss"].detach())
        if not torch.isfinite(parts["loss"]):
            raise RuntimeError(f"non-finite loss: {parts}")
    export_weights(model, out, h=h, p=p)
    return last


def run_overfit(
    dataset: Path,
    steps: int,
    h: int,
    p: int,
    out: Path,
    *,
    structure_type: str | None = None,
    arch: str = "mgn",
    lr: float = 1e-3,
) -> tuple[float, float]:
    torch.manual_seed(0)
    raw = _batches_for(dataset, structure_type)
    batches = [normalize_batch(b) for b in raw]
    model = _make_model(arch, h, p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    first = None
    last = 0.0
    for step in range(steps):
        batch = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        parts = total_loss(pred, batch)
        parts["loss"].backward()
        opt.step()
        last = float(parts["loss"].detach())
        if first is None:
            first = last
        if not torch.isfinite(parts["loss"]):
            raise RuntimeError(f"non-finite loss at step {step}: {parts}")
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(
                f"[{arch}] step {step}/{steps} loss={last:.6f} "
                f"(fail={float(parts['fail']):.4f} util={float(parts['util']):.4f} delta={float(parts['delta']):.4f})"
            )
    export_weights(model, out, h=h, p=getattr(model, "p", p) or p)
    return float(first), last


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="learned-impact trainer")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--compare-baselines", action="store_true", help="overfit MGN-T and MLP; print final losses")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--p", type=int, default=4)
    p.add_argument("--arch", choices=("mgn", "mlp"), default="mgn")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--event",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "fixtures" / "tiny_event",
    )
    p.add_argument("--dataset", type=Path, default=Path("/tmp/li-smoke-set"))
    p.add_argument("--structure-type", type=str, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.smoke:
        out = args.out or Path("/tmp/learned-impact-smoke-weights")
        steps = args.steps if args.steps is not None else 3
        loss = run_smoke(args.event, steps, args.h, args.p, out)
        print(f"smoke ok loss={loss:.6f} weights={out}")
        return

    if args.compare_baselines:
        steps = args.steps if args.steps is not None else 80
        out_root = args.out or Path("/tmp/learned-impact-compare")
        results = {}
        for arch in ("mlp", "mgn"):
            first, last = run_overfit(
                args.dataset,
                steps,
                args.h,
                args.p,
                out_root / arch,
                structure_type=args.structure_type,
                arch=arch,
                lr=args.lr,
            )
            results[arch] = (first, last)
        print(
            f"compare ok mlp {results['mlp'][0]:.4f}→{results['mlp'][1]:.4f}  "
            f"mgn {results['mgn'][0]:.4f}→{results['mgn'][1]:.4f}"
        )
        return

    if args.overfit:
        out = args.out or Path("/tmp/learned-impact-overfit-weights")
        steps = args.steps if args.steps is not None else 50
        first, last = run_overfit(
            args.dataset,
            steps,
            args.h,
            args.p,
            out,
            structure_type=args.structure_type,
            arch=args.arch,
            lr=args.lr,
        )
        print(f"overfit ok arch={args.arch} first={first:.6f} last={last:.6f} weights={out}")
        return

    print("Pass --smoke, --overfit, or --compare-baselines.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
