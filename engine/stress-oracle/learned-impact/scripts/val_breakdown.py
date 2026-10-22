"""Per-family validation breakdown of a checkpoint (P2 analysis): gates' mean AND median
skill ÷ bar over aligned active ensemble calls, pass rates against the held-out member, and
for ballistic sweeps, per speed: the model's impactor exit speed and perforation probability
at the aligned calls against the ensemble's.

    cd learned-impact/python && .venv/bin/python ../scripts/val_breakdown.py --checkpoint <best.pt> \
        --dataset <calls> --recordings <p1 runs root> --out <json>
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from learned_impact.outcome import decode as D
from learned_impact.outcome import metrics as M
from learned_impact.outcome import schema as S
from learned_impact.outcome.batch import HierarchyCache, collate
from learned_impact.outcome.ensemble import aggregate
from learned_impact.outcome.train import _batches, _split_of, decode_config, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--recordings", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
paths = [p for p in S.find_events(a.dataset) if _split_of(p) == "val"]
evs = [S.load_event(p) for p in paths]
groups = collections.defaultdict(list)
for e in evs:
    for c in e.calls:
        c.arrays = dict(c.arrays)
    groups[e.manifest.get("ensemble_id")].append(e)
for g in groups.values():
    aggregate(g)
calls = [c for e in evs for c in e.calls if c.routed]
st = {e.manifest["event_id"]: e.structures for e in evs}
model, ck = load_model(a.checkpoint)
model.eval()
dcfg = decode_config(ck)
cache = HierarchyCache(model.cfg.levels)
rows, bp, by = [], [], []
perf = {}
with torch.no_grad():
    for bc in _batches(calls, 20000, 64):
        b = collate(bc, st, cache)
        out = model(b)
        dec = D.decode(out, b, dcfg)
        rows.extend(M._per_call_outcomes(b, dec))
        bv = b.t["bond_valid"].numpy()
        bp.append(dec["bond_broke"].numpy().astype(float)[bv])
        by.append((b.t["y_bond_broke"].numpy() > 0).astype(float)[bv])
        pp = torch.sigmoid(out["imp_perf_logit"]).numpy()
        ic = b.t["imp_call"].numpy()
        for g, c in enumerate(bc):
            sel = ic == g
            if sel.any():
                perf[(c.event_id, c.call_index)] = float(pp[sel][0])
s = M._summarise(rows, bp, by, [np.zeros(0)], [np.zeros(0)], 0, 1, acceptance=True)
out = {"checkpoint": a.checkpoint, "step": ck.get("step"), "families": {}}
by_fam = collections.defaultdict(list)
for r in s.ensemble_rows:
    by_fam[r["family"]].append(r)
for fam, rs in sorted(by_fam.items()):
    d = {"aligned_active_calls": len(rs), "ensembles": len({r["ensemble"] for r in rs})}
    for gname in ("ram", "partition", "sizes"):
        mod = [r["model"][gname] for r in rs if gname in r["model"]]
        mem = [r["member"][gname] for r in rs if gname in r["member"]]
        if not mod:
            continue
        d[gname] = {"model_pass": float(np.mean([x[0] for x in mod])),
                    "member_pass": float(np.mean([x[0] for x in mem])),
                    "model_sob_mean": float(np.mean([x[1] for x in mod])),
                    "model_sob_median": float(np.median([x[1] for x in mod])),
                    "member_sob_mean": float(np.mean([x[1] for x in mem])),
                    "member_sob_median": float(np.median([x[1] for x in mem]))}
    pf = s.per_family.get(fam, {})
    d["dense"] = {k: pf.get(k) for k in ("crps_ram", "dense_1mARI", "dense_w1_log_mass")}
    out["families"][fam] = d
# ballistic per speed: aligned calls (the members agree on inputs), the model vs the members
rec = {p.name: p for p in Path(a.recordings).glob("*/events/*")}
sweep = collections.defaultdict(lambda: collections.defaultdict(list))
grp = collections.defaultdict(list)
for r in rows:
    c = r["call"]
    if c.family.startswith("bl_") and c.ensemble_id and "y_bond_break_freq" in c.arrays and len(r["imp_speed"]):
        grp[(c.ensemble_id, c.call_index)].append(r)
for (eid, ci), rs in grp.items():
    ctx = rec.get(rs[0]["call"].event_id)
    speed = None
    if ctx is not None and (ctx / "context.json").exists():
        speed = json.loads((ctx / "context.json").read_text()).get("parameters", {}).get("ballistic", {}).get("speed")
    members = [float(x["imp_speed"][0]) for x in rs]
    regimes = [int(x["call"].arrays["y_impactor_regime"][0]) for x in rs]
    sweep[eid]["v0"] = speed
    sweep[eid]["calls"].append({"call": ci, "model_exit": float(rs[0]["imp_speed_hat"][0]),
                                "member_median": float(np.median(members)),
                                "member_iqr": float(np.subtract(*np.percentile(members, [75, 25]))),
                                "model_p_perf": perf.get((rs[0]["call"].event_id, ci)),
                                "member_perf_rate": float(np.mean([g == S.REGIMES["perforate"] for g in regimes]))})
out["ballistic_sweep"] = {k: {"v0": v["v0"], "calls": sorted(v["calls"], key=lambda x: x["call"])}
                          for k, v in sorted(sweep.items(), key=lambda kv: (kv[1]["v0"] or 0))}
Path(a.out).write_text(json.dumps(out, indent=1))
for fam, d in out["families"].items():
    print(fam, d["ensembles"], "ens", d["aligned_active_calls"], "calls")
    for g in ("ram", "partition", "sizes"):
        if g in d:
            x = d[g]
            print(f"  {g:9s} pass {x['model_pass']:.2f} vs {x['member_pass']:.2f} | s/bar mean {x['model_sob_mean']:.2f} median {x['model_sob_median']:.2f}"
                  f" (member mean {x['member_sob_mean']:.2f} median {x['member_sob_median']:.2f})")
for eid, v in out["ballistic_sweep"].items():
    first = v["calls"][0]
    print(f"v0={v['v0']:.2f} {eid[-22:]}: call {first['call']} exit model {first['model_exit']:.2f} vs members {first['member_median']:.2f} (IQR {first['member_iqr']:.2f}); "
          f"p_perf {first['model_p_perf'] if first['model_p_perf'] is None else round(first['model_p_perf'],2)} vs member rate {first['member_perf_rate']:.2f}; aligned calls {len(v['calls'])}")
