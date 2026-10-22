"""Offline evaluation of a checkpoint on one split: skill metrics with the oracle-calibrated
gate, and the three-way report against `blast_parity` QS and the (CPU) punch on the same
calls. Per-call only: the closed-loop full-event gates need the rollout harness
(EVALUATION.md).

    python -m learned_impact.outcome.report --checkpoint ../runs/<job>/best.pt \
        --dataset <dataset> --split val --out ../runs/<job>/report-val.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from . import decode as D
from . import metrics as M
from . import schema as S
from .batch import HierarchyCache, collate
from .train import TAU_GRID, _batches, decode_config, load_model


def _track(a, rep: dict) -> None:
    """The report as its own W&B run in the training run's group (summary + artifact)."""
    from . import tracking as TR

    run = Path(a.checkpoint).resolve().parent
    tr = TR.Tracker(a.wandb, run, {"checkpoint": str(a.checkpoint), "split": a.split,
                                   "dataset": str(a.dataset)},
                    ["report", a.split], group=TR.run_group(run), job_type="report",
                    name=f"{run.name}-report-{a.split}", marker=False)
    tr.summary(json.loads(json.dumps(rep, default=float)))
    tr.artifact(Path(a.out), f"{run.parent.name}-{run.name}-report-{a.split}")
    tr.finish(0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "sealed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--confirm-sealed", action="store_true",
                    help="required for --split sealed: the sealed set is opened once, at P4")
    ap.add_argument("--wandb", default="auto", choices=("auto", "off", "offline", "online"))
    ap.add_argument("--tiers", default="", help="coverage tiers (outcome.coverage): the headline is "
                    "tiers 1+2, each tier reported separately")
    ap.add_argument("--holdout-ballistic", default="", help="the training run's held-out ballistic "
                    "speed positions (tier 1, validation)")
    ap.add_argument("--no-tau-sweep", action="store_true",
                    help="skip the informational τ sweep on test (the verdict uses the frozen τ)")
    a = ap.parse_args(argv)
    if a.split == "sealed" and not a.confirm_sealed:
        ap.error("the sealed set stays closed until P4; pass --confirm-sealed to open it")
    model, ck = load_model(a.checkpoint, a.device)
    from .train import _split_of
    held = tuple(int(x) for x in a.holdout_ballistic.split(",") if x)
    keep_split = {a.split, "train"} if held and a.split == "val" else {a.split}
    events = [S.load_event(p) for p in S.find_events(a.dataset) if _split_of(p) in keep_split]
    if held and a.split == "val":  # the training run's in-memory held-out ballistic speeds
        from .train import holdout_ballistic_speeds
        holdout_ballistic_speeds(events, held)
    events = [e for e in events if e.manifest["split"] == a.split
              and not e.manifest.get("truncated", False)]
    # ensemble soft labels and alignment, exactly as training's `Data` derives them
    from collections import defaultdict

    from .ensemble import aggregate
    groups = defaultdict(list)
    for e in events:
        if e.manifest.get("ensemble_id"):
            for c in e.calls:
                c.arrays = dict(c.arrays)
            groups[e.manifest["ensemble_id"]].append(e)
    for evs in groups.values():
        aggregate(evs)
    calls = [c for e in events for c in e.calls if c.routed]
    if not calls:
        ap.error(f"no routed calls in split {a.split!r} under {a.dataset}")
    st = {e.manifest["event_id"]: e.structures for e in events}
    cache = HierarchyCache(model.cfg.levels)
    tc = ck["train_config"]
    batches = [collate(bc, st, cache) for bc in
               _batches(calls, tc.get("batch_budget", 20000), tc.get("batch_calls", 64))]
    dcfg = decode_config(ck)
    tier_of = None
    if a.tiers:  # three tiers, never one blended average; the headline is tiers 1+2
        from .coverage import TIER_NAMES, event_tier
        tmap = json.loads(Path(a.tiers).read_text())
        tier_of = {id(c): event_tier(tmap, c.event_id, a.split) for c in calls}
        sel = [c for c in calls if tier_of[id(c)] in (1, 2)]
        if not sel:
            ap.error("no tier-1/2 calls: nothing to select or report as the headline")
        batches = [collate(bc, st, cache) for bc in
                   _batches(sel, tc.get("batch_budget", 20000), tc.get("batch_calls", 64))]
    rep = M.three_way(model, batches, dcfg, ck.get("priors"))
    if tier_of is not None:
        rep["headline"] = "tiers 1+2"
        rep["tiers"] = {}
        for t in (1, 2, 3):
            ct = [c for c in calls if tier_of[id(c)] == t]
            if not ct:
                continue
            bt = [collate(bc, st, cache) for bc in
                  _batches(ct, tc.get("batch_budget", 20000), tc.get("batch_calls", 64))]
            rt = M.three_way(model, bt, dcfg, ck.get("priors"))
            rep["tiers"][str(t)] = {"name": TIER_NAMES[t], "n_calls": len(ct),
                                    "n_events": len({c.event_id for c in ct}),
                                    "selection_score": rt["selection_score"],
                                    "families": rt["families"], "acceptance": rt["acceptance"]}
    # τ was chosen on validation and is frozen in the checkpoint. On test it is confirmed,
    # not re-chosen: the verdict is the acceptance gate at the frozen τ; the sweep over the
    # same grid is reported only to show the regret of the validation choice.
    rep["tau_frozen"] = dcfg.tau
    if a.split != "val" and not a.no_tau_sweep and dcfg.break_mode == "threshold":
        sweep = {t: M.evaluate(model, batches, dataclasses.replace(dcfg, tau=t), ck.get("priors")).score
                 for t in TAU_GRID}
        best = min(sweep, key=sweep.get)
        rep["tau_sweep"] = {"scores": sweep, "best_on_this_split": best,
                            "regret": sweep.get(dcfg.tau, rep["selection_score"]) - sweep[best]}
    rep["accepted"] = bool(rep["acceptance"]) and all(f["accept"] for f in rep["acceptance"].values())
    rep["checkpoint"] = str(a.checkpoint)
    rep["split"] = a.split
    rep["n_events"], rep["n_calls"] = len(events), len(calls)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    _track(a, rep)
    for fam, e in sorted(rep["families"].items()):
        print(f"[report] {fam}: superiority={e.get('superiority')} "
              f"non_inferiority={e.get('non_inferiority', {}).get('pass')} model={e.get('model')}")
    for fam, acc in sorted(rep["acceptance"].items()):
        cells = [f"{k}: model {acc[k]['model_pass_rate']:.2f} vs member {acc[k]['member_pass_rate']:.2f}"
                 f" pass, s/bar {acc[k]['model_mean_skill_over_bar']:.2f} vs "
                 f"{acc[k]['member_mean_skill_over_bar']:.2f}" for k in M.GATE_METRICS if k in acc]
        print(f"[report] acceptance {fam} ({acc['n_ensembles']} ensembles): "
              f"gates-v1.1 {'ACCEPT' if acc['accept'] else 'reject'} "
              f"(v1 {'accept' if acc['accept_v1'] else 'reject'}); " + "; ".join(cells))
    if "tau_sweep" in rep:
        ts = rep["tau_sweep"]
        print(f"[report] frozen tau {rep['tau_frozen']}; best on {a.split} {ts['best_on_this_split']} "
              f"(regret {ts['regret']:.3f}, informational)")
    print(f"[report] {len(rep['worse_than_qs'])} calls worse than QS (listed in {a.out})")


if __name__ == "__main__":
    main()
