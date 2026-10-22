"""P0 step 5: pilot learnability and data-scaling study (plan, P0 step 5).

Trains the vanilla configuration on nested subsets of the training events (e.g. 250 / 1k /
4k free-wall events), for each seed and variant (A impact-only, B residual over QS), all
against the *same* validation events. It then reports:
- the validation skill metrics, with the ensemble-calibrated gate where ensembles exist;
- the three-way report against QS and the punch;
- a saturating power-law fit `e(N) = a·N^(−b) + c` of each primary metric against data
  size, projected to the gate.

Runs are sequential (one training job at a time) and resumable: a finished run's
`result.json` is reused.

    python -m learned_impact.outcome.pilot --dataset <calls> --out ../runs/p0-step5 \
        --sizes 250,1000,4000 --seeds 0,1,2 --variants A,B --family free_wall
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import decode as D
from . import metrics as M
from . import schema as S
from .batch import HierarchyCache, collate
from .model import ModelConfig
from .train import Data, Progress, TrainConfig, _batches, load_model, train

# Metrics whose gate is "≤ 1" after division by their bar (ensemble) or tolerance (single).
GATE_ENSEMBLE = {"skill_over_bar_ram": 1.0, "skill_over_bar_partition": 1.0,
                 "skill_over_bar_sizes": 1.0}
GATE_SINGLE = {"ram_err_rel": 0.05}  # the plan's 5% of v0, single-run fallback


def nested_subsets(groups: list[str], sizes: list[int], seed: int) -> dict[int, set[str]]:
    """Nested training subsets by split_group (members and branches of one recipe stay
    together), so a bigger size always contains the smaller ones."""
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    return {n: set(order[:min(n, len(order))]) for n in sizes}


def fit_power_law(n: np.ndarray, e: np.ndarray) -> dict | None:
    """Least squares `e = a·n^(−b) + c` with c ≥ 0 by a grid on c (closed form for a, b in
    log space). Needs ≥ 3 sizes; with 2 it fits c = 0."""
    n, e = np.asarray(n, float), np.asarray(e, float)
    ok = np.isfinite(e) & (e > 0)
    n, e = n[ok], e[ok]
    if len(n) < 2:
        return None
    cs = [0.0] if len(n) < 3 else np.linspace(0.0, 0.99 * e.min(), 100)
    best = None
    for c in cs:
        y = np.log(e - c)
        A = np.vstack([np.ones_like(n), -np.log(n)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(np.exp(coef[0])), float(coef[1])
        rss = float(np.sum((a * n ** (-b) + c - e) ** 2))
        if best is None or rss < best["rss"]:
            best = {"a": a, "b": b, "c": float(c), "rss": rss, "n_sizes": int(len(n))}
    return best


def project(fit: dict | None, bar: float) -> dict:
    """Events needed to reach `bar` under the fit (inf if the asymptote c is above it, or the
    curve does not improve)."""
    if fit is None:
        return {"n_needed": None, "reason": "too few sizes"}
    if fit["b"] <= 0:
        return {"n_needed": float("inf"), "reason": "no improvement with data (b ≤ 0)"}
    if fit["c"] >= bar:
        return {"n_needed": float("inf"),
                "reason": f"asymptote {fit['c']:.3g} ≥ bar {bar}: more data alone will not pass"}
    reason = "fit" if fit.get("n_sizes", 3) >= 3 else \
        "2 sizes: asymptote assumed 0, so this is optimistic"
    log_n = np.log(fit["a"] / (bar - fit["c"])) / fit["b"]  # log space: no overflow when flat
    if log_n > np.log(1e12):
        return {"n_needed": float("inf"), "reason": "effectively flat (> 10¹² events)"}
    return {"n_needed": float(np.exp(log_n)), "reason": reason}


def _val_report(ckpt: Path, data: Data, cfg: TrainConfig) -> dict:
    model, ck = load_model(ckpt, "cpu")
    cache = HierarchyCache(model.cfg.levels)
    batches = [collate(bc, data.structures, cache)
               for bc in _batches(data.val, cfg.batch_budget, cfg.batch_calls)]
    dcfg = D.DecodeConfig(temperature=ck.get("temperature", 1.0), tau=ck.get("tau", 1.0),
                          energy_tolerance=cfg.decode.energy_tolerance)
    rep = M.three_way(model, batches, dcfg, ck.get("priors"))
    return rep


def run(dataset: str, out: str, sizes: list[int], seeds: list[int], variants: list[str],
        family: str | None, steps_per_event: float, min_steps: int, max_steps: int,
        device: str, model_cfg: ModelConfig, pushforward: float = 0.0,
        rereport: bool = False, val_dataset: str | None = None,
        fractions: list[float] | None = None, train_events: list | None = None,
        call_cache: dict | None = None) -> dict:
    root = Path(out)
    prog = Progress(root, total=len(sizes) * len(seeds) * len(variants))
    full = Data(dataset, val_dataset=val_dataset, train_events=train_events, call_cache=call_cache)
    keep = (lambda c: True) if family is None else (lambda c: c.family == family)
    train_groups = sorted({c.split_group for c in full.train if keep(c)})
    if fractions:  # shares of the training layout groups (nested, as for sizes)
        sizes = [max(1, round(f * len(train_groups))) for f in fractions]
    val_calls = [c for c in full.val if keep(c)]
    if not val_calls:
        raise S.SchemaError(f"no validation calls for family {family!r}")
    has_ens = any(c.ensemble_id for c in val_calls)
    prog.line(f"pilot: {len(train_groups)} train groups, {len(val_calls)} val calls "
              f"(ensembles: {has_ens}), sizes {sizes}, seeds {seeds}, variants {variants}")
    results = []
    k = 0
    for variant in variants:
        for seed in seeds:
            subsets = nested_subsets(train_groups, sizes, seed)
            for n in sizes:
                k += 1
                tag = f"{variant}_n{n}_s{seed}"
                rdir = root / tag
                res_path = rdir / "result.json"
                if res_path.exists():
                    r = json.loads(res_path.read_text())
                    if rereport:  # re-score the saved best checkpoint with the current code
                        d = Data.__new__(Data)
                        d.__dict__ = dict(full.__dict__)
                        d.val = val_calls
                        r["three_way"] = _val_report(rdir / "best.pt", d, TrainConfig())
                        # keep training-time metrics the per-call report does not compute
                        r["val"] = {**r.get("val", {}), **r["three_way"]["metrics"]["model"]}
                        res_path.write_text(json.dumps(r, indent=1, default=float))
                    results.append(r)
                    prog.line(f"[{k}/{prog.total}] {tag}: cached" + (" (re-scored)" if rereport else ""))
                    continue
                groups = subsets[n]
                d = Data.__new__(Data)
                d.__dict__ = dict(full.__dict__)
                d.train = [c for c in full.train if keep(c) and c.split_group in groups]
                d.val = val_calls
                d.successor = {kk: v for kk, v in full.successor.items()
                               if v.split_group in groups}
                n_events = len({c.event_id for c in d.train})
                steps = int(np.clip(steps_per_event * n_events, min_steps, max_steps))
                mc = dataclasses.replace(model_cfg, variant=variant)
                cfg = TrainConfig(dataset=dataset, val_dataset=val_dataset or "", out=str(rdir),
                                  steps=steps, seed=seed,
                                  device=device, model=mc, eval_every=max(steps // 8, 50),
                                  pushforward_prob=pushforward)
                t0 = time.time()
                prog.line(f"[{k}/{prog.total}] {tag}: {n_events} events, "
                          f"{len(d.train)} calls, {steps} steps")
                prog.update(force=True, state="running", step=k - 1, current=tag)
                tr = train(cfg, d)
                rep = _val_report(rdir / "best.pt", d, cfg)
                r = {"variant": variant, "seed": seed, "n_groups": len(groups),
                     "n_events": n_events, "n_calls": len(d.train), "steps": steps,
                     "train_s": time.time() - t0, "best_step": tr["best"]["step"],
                     "val": tr["best"]["metrics"], "three_way": rep}
                res_path.write_text(json.dumps(r, indent=1, default=float))
                results.append(r)
                prog.line(f"[{k}/{prog.total}] {tag}: done in {r['train_s'] / 60:.1f} min, "
                          f"score {tr['best']['score']:.3f}")
                prog.update(force=True, step=k)
    summary = summarise(results, has_ens)
    (root / "pilot.json").write_text(json.dumps(summary, indent=1, default=float))
    (root / "pilot.md").write_text(markdown(summary))
    prog.update(force=True, state="done")
    prog.line(f"pilot written: {root / 'pilot.md'}")
    return summary


def summarise(results: list[dict], ensembles: bool) -> dict:
    gates = GATE_ENSEMBLE if ensembles else GATE_SINGLE
    out = {"gates": gates, "ensembles": ensembles, "variants": {}}
    for variant in sorted({r["variant"] for r in results}):
        rs = [r for r in results if r["variant"] == variant]
        # one column per requested size (nested subsets of recipe groups); the x of the fit is
        # the mean number of events at that size over the seeds
        group_sizes = sorted({r["n_groups"] for r in rs})
        ev = {g: float(np.mean([r["n_events"] for r in rs if r["n_groups"] == g]))
              for g in group_sizes}
        sizes = [int(round(ev[g])) for g in group_sizes]
        table = {}
        for metric in [*gates, "score", "crps_ram", "dense_1mARI", "dense_w1_log_mass", "bond_f1",
                       "bond_recall", "false_break_rate",
                       "regime_brier"]:
            per = {int(round(ev[g])): [r["val"].get(metric) for r in rs if r["n_groups"] == g
                                       and r["val"] and r["val"].get(metric) is not None]
                   for g in group_sizes}
            table[metric] = {str(n): {"mean": float(np.mean(v)) if v else None,
                                      "min": float(np.min(v)) if v else None,
                                      "max": float(np.max(v)) if v else None, "seeds": len(v)}
                             for n, v in per.items()}
        fits = {}
        for metric, bar in gates.items():
            pts = [(n, table[metric][str(n)]["mean"]) for n in sizes
                   if table[metric][str(n)]["mean"] is not None]
            f = fit_power_law(np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                              ) if pts else None
            fits[metric] = {"fit": f, "bar": bar, **project(f, bar)}
        three = {str(r["n_events"]) + f"/s{r['seed']}": {
            fam: {"superiority": e.get("superiority"),
                  "non_inferiority": (lambda n: None if not n else
                                      f"{n.get('pass')} ({n.get('model_not_worse', 0):.1%} of "
                                      f"{n.get('qs_right_calls')} calls not worse)")(
                                          e.get("non_inferiority"))}
            for fam, e in r["three_way"].get("families", {}).items()} for r in rs}
        out["variants"][variant] = {"sizes": sizes, "table": table, "fits": fits,
                                    "three_way": three}
    return out


def markdown(s: dict) -> str:
    lines = ["# Pilot learnability and data scaling (P0 step 5)", "",
             f"Ensemble-calibrated gates: {s['ensembles']}. Gates (≤): "
             + ", ".join(f"`{k}` {v}" for k, v in s["gates"].items()), ""]
    for v, d in s["variants"].items():
        lines += [f"## Variant {v}", "", "| metric | " + " | ".join(
            f"N={n}" for n in d["sizes"]) + " | fit a·N^-b + c | events to the gate |",
            "|---|" + "---|" * len(d["sizes"]) + "---|---|"]
        for metric, row in d["table"].items():
            cells = []
            for n in d["sizes"]:
                c = row[str(n)]
                cells.append("—" if c["mean"] is None else
                             f"{c['mean']:.3f} [{c['min']:.3f}, {c['max']:.3f}] ×{c['seeds']}")
            f = d["fits"].get(metric)
            fit_s = proj = ""
            if f is not None:
                ff = f["fit"]
                fit_s = "—" if ff is None else f"a={ff['a']:.3g} b={ff['b']:.3g} c={ff['c']:.3g}"
                nn = f["n_needed"]
                proj = ("—" if nn is None else f"∞ ({f['reason']})" if not np.isfinite(nn)
                        else "> 10⁷ (effectively flat)" if nn > 1e7 else f"{nn:.0f}"
                        + ("" if f["reason"] == "fit" else f" ({f['reason']})"))
            lines.append(f"| `{metric}` | " + " | ".join(cells) + f" | {fit_s} | {proj} |")
        lines += ["", "Three-way (per run: family → superiority / non-inferiority):", ""]
        for k, fams in d["three_way"].items():
            lines.append(f"- N/seed {k}: " + "; ".join(
                f"{fam}: superior to QS and punch={e['superiority']}, "
                f"non-inferior on QS-correct calls={e['non_inferiority']}"
                for fam, e in fams.items()))
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-dataset", default=None,
                    help="validate on this dataset's val split (the bulk evaluation ensembles)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="250,1000,4000")
    ap.add_argument("--fractions", default=None,
                    help="instead of --sizes: shares of the training groups, e.g. 0.25,0.5,1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--family", default=None)
    ap.add_argument("--steps-per-event", type=float, default=4.0)
    ap.add_argument("--min-steps", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reference", action="store_true", help="W64 / 12 rounds (the default)")
    ap.add_argument("--large", action="store_true", help="W128 / 16 rounds")
    ap.add_argument("--cost", action="store_true", help="W32 / 8 rounds (P0's cost point)")
    ap.add_argument("--full-state", action="store_true", help="/5 full-state inputs and heads")
    ap.add_argument("--train-events", default="", help="file of P1 recording dirs: training calls via CallCache")
    ap.add_argument("--call-cache-module", default="")
    ap.add_argument("--call-cache", default="")
    ap.add_argument("--call-builder", default="")
    ap.add_argument("--cache-gb", type=float, default=1.5,
                    help="total disk bound of the call cache (as train.py)")
    ap.add_argument("--break-time-head", action="store_true")
    ap.add_argument("--hidden-head", action="store_true")
    ap.add_argument("--kinematic", action="store_true")
    ap.add_argument("--pushforward", type=float, default=0.0)
    ap.add_argument("--rereport", action="store_true",
                    help="re-score every finished run's best.pt with the current metrics")
    a = ap.parse_args(argv)
    mc = (ModelConfig.large() if a.large else ModelConfig.cost() if a.cost
          else ModelConfig.reference())
    mc.full_state = a.full_state
    mc.break_time_head, mc.hidden_head, mc.kinematic = a.break_time_head, a.hidden_head, a.kinematic
    run(a.dataset, a.out, [int(x) for x in a.sizes.split(",")],
        [int(x) for x in a.seeds.split(",")], a.variants.split(","), a.family,
        a.steps_per_event, a.min_steps, a.max_steps, a.device, mc, a.pushforward, a.rereport,
        a.val_dataset, [float(x) for x in a.fractions.split(",")] if a.fractions else None,
        [x for x in Path(a.train_events).read_text().split() if x] if a.train_events else None,
        {"module": a.call_cache_module, "root": a.call_cache, "builder": a.call_builder,
         "max_gb": a.cache_gb} if a.train_events else None)


if __name__ == "__main__":
    main()
