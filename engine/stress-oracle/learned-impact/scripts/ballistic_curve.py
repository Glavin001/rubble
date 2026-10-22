"""Per-speed ballistic curves of a checkpoint against the oracle ensembles (teacher-forced,
the aligned impact call of each 8-member ensemble): perforation probability (model's
p_perforate vs the ensemble's perforation frequency) and the impactor's frame-end speed
along its incoming direction (model vs the members' mean and range). Families and splits
as recorded; a training family's curve is a fit, not a generalisation.

    python ../scripts/ballistic_curve.py --checkpoint best.pt --dataset <calls-v5> [--families a,b] [--out curve.json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import decode as D, schema as S  # noqa: E402
from learned_impact.outcome.batch import HierarchyCache, collate  # noqa: E402
from learned_impact.outcome.ensemble import aggregate  # noqa: E402
from learned_impact.outcome.train import decode_config, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--families", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()
fams = set(a.families.split(",")) if a.families else None
groups = defaultdict(list)
for p in S.find_events(a.dataset):
    m = json.loads((Path(p) / "manifest.json").read_text())
    if not m.get("ensemble_id") or not m["family"].startswith("bl_") or (fams and m["family"] not in fams):
        continue
    groups[m["ensemble_id"]].append(p)
model, ck = load_model(Path(a.checkpoint), "cpu")
model.eval()
dcfg = decode_config(ck)
cache = HierarchyCache(model.cfg.levels)
rows = []
for eid, paths in sorted(groups.items()):
    evs = [S.load_event(p) for p in paths]
    if len(evs) < int(evs[0].manifest.get("ensemble_size", len(evs))):
        continue
    for e in evs:
        for c in e.calls:
            c.arrays = dict(c.arrays)
    aggregate(evs)
    calls = [c for e in evs for c in e.calls if c.routed and "ens_impactor_v_end" in c.arrays]
    if not calls:
        continue
    st = {e.manifest["event_id"]: e.structures for e in evs}
    b = collate(calls, st, cache)
    with torch.no_grad():
        out = model(b)
        dec = D.decode(out, b, dcfg)
    p_perf = torch.sigmoid(out["imp_perf_logit"]).numpy()
    ic = b.t["imp_call"].numpy()
    v_hat = dec["imp_v_end"].numpy()
    m0 = evs[0].manifest
    rec = json.loads((Path(paths[0]) / "manifest.json").read_text())
    for g, c in enumerate(calls[:1]):  # one aligned call per ensemble (the members' are aligned)
        sel = ic == g
        v_in = c.arrays["impactor_v"].astype(float)[0]
        u = v_in / max(np.linalg.norm(v_in), 1e-12)
        ens = c.arrays["ens_impactor_v_end"].astype(float)[:, 0] @ u
        model_v = np.array([v_hat[ic == k][0] @ u for k in range(len(calls))])
        rows.append({"family": m0["family"], "split": m0["split"], "ensemble": eid,
                     "speed": float(np.linalg.norm(v_in)),
                     "perf_freq": float(c.arrays["y_impactor_perforate_freq"][0]),
                     "p_perf_model": float(np.mean([p_perf[ic == k][0] for k in range(len(calls))])),
                     "v_end_ens_mean": float(ens.mean()), "v_end_ens_min": float(ens.min()),
                     "v_end_ens_max": float(ens.max()), "v_end_model": float(model_v.mean()),
                     "frame": int(c.frame)})
rows.sort(key=lambda r: (r["family"], r["speed"]))
for fam in sorted({r["family"] for r in rows}):
    rr = [r for r in rows if r["family"] == fam]
    print(f"\n{fam} ({rr[0]['split']}), aligned call per ensemble")
    print("| v0 m/s | perf freq (oracle) | p_perf (model) | v_end oracle mean [min, max] | v_end model |")
    print("|---:|---:|---:|---|---:|")
    for r in rr:
        print(f"| {r['speed']:.2f} | {r['perf_freq']:.2f} | {r['p_perf_model']:.2f} | "
              f"{r['v_end_ens_mean']:.2f} [{r['v_end_ens_min']:.2f}, {r['v_end_ens_max']:.2f}] | {r['v_end_model']:.2f} |")
if a.out:
    Path(a.out).write_text(json.dumps(rows, indent=1))
