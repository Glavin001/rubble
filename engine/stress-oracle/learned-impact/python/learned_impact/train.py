"""CLI: smoke, overfit, compare baselines, and mini-set train with held-out structure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .dataset import iter_event_frames, list_events, load_all_frames, load_first_frame, synthetic_batch
from .export import export_weights
from .losses import total_loss
from .metrics import eval_batch, mean_metrics
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


def param_count(model: torch.nn.Module) -> tuple[int, int]:
    n = sum(p.numel() for p in model.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return n, bytes_


def _pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" or (prefer == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _batches_matching(
    dataset: Path,
    *,
    structure_type: str | None = None,
    exclude_structure: str | None = None,
    load_types: set[str] | None = None,
    exclude_load_types: set[str] | None = None,
    event_prefixes: set[str] | None = None,
    exclude_event_prefixes: set[str] | None = None,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for ev in list_events(dataset):
        man = load_event(ev).manifest
        st = man.get("structure_type")
        lt = man.get("load_type") or ""
        eid = str(man.get("event_id") or ev.name)
        if structure_type is not None and st != structure_type:
            continue
        if exclude_structure is not None and st == exclude_structure:
            continue
        if load_types is not None and lt not in load_types:
            continue
        if exclude_load_types is not None and lt in exclude_load_types:
            continue
        if event_prefixes is not None and not any(eid.startswith(p) for p in event_prefixes):
            continue
        if exclude_event_prefixes is not None and any(eid.startswith(p) for p in exclude_event_prefixes):
            continue
        batches.extend(iter_event_frames(ev))
    if not batches:
        raise FileNotFoundError(
            f"no frames in {dataset} (structure={structure_type!r} exclude={exclude_structure!r} "
            f"loads={load_types!r} exclude_loads={exclude_load_types!r} "
            f"event_prefixes={event_prefixes!r} exclude_event_prefixes={exclude_event_prefixes!r})"
        )
    return batches


def _batches_for(dataset: Path, structure_type: str | None) -> list[dict[str, torch.Tensor]]:
    if structure_type:
        return _batches_matching(dataset, structure_type=structure_type)
    return load_all_frames(dataset)


@torch.no_grad()
def eval_mean_loss(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], *, normalized: bool = False) -> dict[str, float]:
    model.eval()
    tot = fail = util = delta = 0.0
    metric_rows: list[dict[str, float]] = []
    for raw in batches:
        batch = raw if normalized else normalize_batch(raw)
        pred = model(batch)
        parts = total_loss(pred, batch)
        tot += float(parts["loss"])
        fail += float(parts["fail"])
        util += float(parts["util"])
        delta += float(parts["delta"])
        metric_rows.append(eval_batch(pred, batch))
    n = max(1, len(batches))
    out = {"loss": tot / n, "fail": fail / n, "util": util / n, "delta": delta / n}
    out.update(mean_metrics(metric_rows))
    return out


@torch.no_grad()
def run_eval(
    dataset: Path,
    weights: Path,
    h: int,
    p: int,
    *,
    structure_type: str | None,
    arch: str,
) -> dict[str, float]:
    """Evaluate a real PyTorch checkpoint; never substitute random weights."""
    raw = _batches_for(dataset, structure_type)
    model = _make_model(arch, h, p)
    sd_path = weights / "state_dict.pt"
    if sd_path.exists():
        model.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
    else:
        raise FileNotFoundError(f"evaluation requires {sd_path}; refusing random weights")
    return eval_mean_loss(model, raw)


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
    n_params, n_bytes = param_count(model)
    print(f"[{arch}] params={n_params:,} ({n_bytes / 1024:.1f} KiB f32) frames={len(batches)}")
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


def run_train(
    dataset: Path,
    steps: int,
    h: int,
    p: int,
    out: Path,
    *,
    holdout: str | None,
    holdout_loads: set[str] | None,
    holdout_event_prefixes: set[str] | None = None,
    arch: str,
    lr: float,
    device: str = "auto",
) -> dict[str, float]:
    torch.manual_seed(0)
    dev = _pick_device(device)
    train_raw = _batches_matching(
        dataset,
        exclude_structure=holdout,
        exclude_load_types=holdout_loads,
        exclude_event_prefixes=holdout_event_prefixes,
    )
    if holdout:
        hold_raw = _batches_matching(dataset, structure_type=holdout)
    elif holdout_loads:
        hold_raw = _batches_matching(dataset, load_types=holdout_loads)
    elif holdout_event_prefixes:
        hold_raw = _batches_matching(dataset, event_prefixes=holdout_event_prefixes)
    else:
        hold_raw = []
    train = [_batch_to_device(normalize_batch(b), dev) for b in train_raw]
    hold_norm = [_batch_to_device(normalize_batch(b), dev) for b in hold_raw] if hold_raw else []
    model = _make_model(arch, h, p).to(dev)
    n_params, n_bytes = param_count(model)
    print(
        f"[{arch}] device={dev} params={n_params:,} ({n_bytes / 1024:.1f} KiB f32) "
        f"train_frames={len(train)} holdout_struct={holdout!r} "
        f"holdout_loads={sorted(holdout_loads) if holdout_loads else None} "
        f"holdout_event_prefixes={sorted(holdout_event_prefixes) if holdout_event_prefixes else None} "
        f"hold_frames={len(hold_norm)}"
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    first = last = 0.0
    for step in range(steps):
        batch = train[step % len(train)]
        opt.zero_grad(set_to_none=True)
        parts = total_loss(model(batch), batch)
        parts["loss"].backward()
        opt.step()
        last = float(parts["loss"].detach())
        if step == 0:
            first = last
        if not torch.isfinite(parts["loss"]):
            raise RuntimeError(f"non-finite loss at step {step}: {parts}")
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            msg = (
                f"[{arch}] step {step}/{steps} train={last:.6f} "
                f"(fail={float(parts['fail']):.4f} util={float(parts['util']):.4f} delta={float(parts['delta']):.4f})"
            )
            if hold_norm and (step % max(1, steps // 5) == 0 or step == steps - 1):
                hol = eval_mean_loss(model, hold_norm, normalized=True)
                model.train()
                msg += f"  holdout={hol['loss']:.6f} f1={hol['fail_f1']:.3f}"
            print(msg)
    export_weights(model.cpu(), out, h=h, p=getattr(model, "p", p) or p)
    result = {"train_first": first, "train_last": last, "device": str(dev)}
    if hold_norm:
        # Re-eval on CPU after export move for stable metrics.
        hold_cpu = [_batch_to_device(b, torch.device("cpu")) for b in hold_norm]
        hol = eval_mean_loss(model, hold_cpu, normalized=True)
        result.update({f"holdout_{k}": v for k, v in hol.items()})
        # Also report train-set metrics on a sample of train frames (all frames).
        train_cpu = [_batch_to_device(b, torch.device("cpu")) for b in train]
        tr = eval_mean_loss(model, train_cpu, normalized=True)
        result.update({f"train_{k}": v for k, v in tr.items()})
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="learned-impact trainer")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--train", action="store_true", help="multi-event train; optional held-out structure")
    p.add_argument("--eval", action="store_true", help="eval weights on dataset (fail F1 / util / delta)")
    p.add_argument("--compare-baselines", action="store_true", help="overfit MGN-T and MLP; print final losses")
    p.add_argument("--params", action="store_true", help="print parameter counts and exit")
    p.add_argument("--weights", type=Path, default=None, help="weights dir with state_dict.pt (for --eval)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--h", type=int, default=None)
    p.add_argument("--p", type=int, default=None)
    p.add_argument("--arch", choices=("mgn", "mlp"), default="mgn")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--event",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "fixtures" / "tiny_event",
    )
    p.add_argument("--dataset", type=Path, default=Path("/tmp/li-smoke-set"))
    p.add_argument("--structure-type", type=str, default=None)
    p.add_argument("--holdout-structure-type", type=str, default=None)
    p.add_argument(
        "--holdout-load-type",
        action="append",
        default=None,
        help="Hold out load_type(s); repeatable. Useful for wall speed splits.",
    )
    p.add_argument(
        "--holdout-event-prefix",
        action="append",
        default=None,
        help="Hold out event_id prefix(es); repeatable. E.g. cantilever_long for arm-length split.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Training device (auto prefers MPS when available).",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    # Sensible defaults: tiny for smoke, mid for overfit/compare, full for train.
    def hp(default_h: int, default_p: int) -> tuple[int, int]:
        return (args.h if args.h is not None else default_h, args.p if args.p is not None else default_p)

    if args.params:
        for label, h, ptok, arch in [
            ("mgn smoke", 8, 4, "mgn"),
            ("mgn overfit/train", 48, 16, "mgn"),
            ("mgn plan-ish", 48, 64, "mgn"),
            ("mlp", 48, 0, "mlp"),
        ]:
            m = _make_model(arch, h, ptok)
            n, b = param_count(m)
            print(f"{label} h={h} p={ptok}: {n:,} params · {b / 1024:.1f} KiB f32")
        return

    if args.smoke:
        h, ptok = hp(8, 4)
        out = args.out or Path("/tmp/learned-impact-smoke-weights")
        steps = args.steps if args.steps is not None else 3
        loss = run_smoke(args.event, steps, h, ptok, out)
        print(f"smoke ok loss={loss:.6f} weights={out}")
        return

    if args.compare_baselines:
        h, ptok = hp(48, 16)
        steps = args.steps if args.steps is not None else 80
        out_root = args.out or Path("/tmp/learned-impact-compare")
        results = {}
        for arch in ("mlp", "mgn"):
            first, last = run_overfit(
                args.dataset,
                steps,
                h,
                ptok,
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

    if args.eval:
        h, ptok = hp(48, 16)
        weights = args.weights or args.out or Path("/tmp/learned-impact-train-weights")
        m = run_eval(
            args.dataset,
            weights,
            h,
            ptok,
            structure_type=args.structure_type,
            arch=args.arch,
        )
        print(
            f"eval ok loss={m['loss']:.6f} fail_f1={m['fail_f1']:.3f} "
            f"util_mae={m['util_mae']:.4f} delta_rmse={m['delta_rmse']:.4f} n={int(m['n'])}"
        )
        return

    if args.train:
        h, ptok = hp(48, 16)
        out = args.out or Path("/tmp/learned-impact-train-weights")
        steps = args.steps if args.steps is not None else 400
        holdout = args.holdout_structure_type
        hold_loads = set(args.holdout_load_type) if args.holdout_load_type else None
        hold_prefixes = set(args.holdout_event_prefix) if args.holdout_event_prefix else None
        r = run_train(
            args.dataset,
            steps,
            h,
            ptok,
            out,
            holdout=holdout,
            holdout_loads=hold_loads,
            holdout_event_prefixes=hold_prefixes,
            arch=args.arch,
            lr=args.lr,
            device=args.device,
        )
        extra = ""
        if "holdout_loss" in r:
            extra = (
                f" holdout={r['holdout_loss']:.6f} f1={r.get('holdout_fail_f1', 0):.3f}"
                f" util_mae={r.get('holdout_util_mae', 0):.4f}"
            )
        if "train_fail_f1" in r:
            extra += f" train_f1={r['train_fail_f1']:.3f}"
        # Persist metrics next to weights for findings.
        metrics_path = out / "metrics.json"
        out.mkdir(parents=True, exist_ok=True)
        serializable = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()}
        metrics_path.write_text(json.dumps(serializable, indent=2) + "\n")
        print(
            f"train ok arch={args.arch} first={r['train_first']:.6f} last={r['train_last']:.6f}"
            f"{extra} weights={out} metrics={metrics_path}"
        )
        return

    if args.overfit:
        h, ptok = hp(48, 16)
        out = args.out or Path("/tmp/learned-impact-overfit-weights")
        steps = args.steps if args.steps is not None else 50
        first, last = run_overfit(
            args.dataset,
            steps,
            h,
            ptok,
            out,
            structure_type=args.structure_type,
            arch=args.arch,
            lr=args.lr,
        )
        print(f"overfit ok arch={args.arch} first={first:.6f} last={last:.6f} weights={out}")
        return

    print("Pass --smoke, --overfit, --train, --compare-baselines, or --params.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
