"""Overfit milestone on real oracle calls (viability step 0, planning session 2026-10-22).

Train and evaluate on the *same* events, with no held-out data and no regularisation (no
weight decay, noise, pushforward or label smoothing), at the default W64/12. A
representation, architecture or decode that cannot memorise a handful of real events is a
bug to fix before anything else. On the training events themselves it reports:

- every loss term's floor (normalised by its zero predictor; the eval-mode curves are in
  `history.jsonl`);
- bond break P/R/F1 against the hard labels, with the deployed margin decision (τ = 1);
- chunk and impactor velocity errors next to the ensemble spread: the RMS difference
  between the event's members' outcomes for the same call index and bodies (P1's 8-member
  ensembles);
- the share of calls whose decoded fragments equal the oracle's (mass-weighted ARI = 1);
- the invariants: momentum residual on free islands, KE growth beyond the bound, and the
  energy-bound activation rate.

Events: single members (`_e0`) of P1's evaluation ensembles, which hold every call from
impact through settling.
- 1 event: an anchored tier-1 wall that the projectile perforates.
- 10 events: free and anchored, tiers 1 and 2, at least one perforation and one arrest.
- 100 events: a seeded, family-stratified draw.

    LEARNED_IMPACT_ACCEPT_V3=1 python -m learned_impact.outcome.overfit_real \
        --dataset <calls-v3-eval> --n-events 10 --steps 4000 --out ../runs/overfit/n10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import metrics as M
from . import schema as S
from .batch import SCALE_INDEX, HierarchyCache, collate
from .train import Data, TrainConfig, _batches, load_model, train


def _regime(e: S.Event) -> int:
    regs = [int(r) for c in e.calls for r in c.arrays["y_impactor_regime"]]
    return regs[-1] if regs else S.REGIME_UNDECIDED


def select(root: Path, n: int, seed: int = 0) -> list[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith("_e0"))
    metas = []
    for p in paths:
        m = json.loads((p / "manifest.json").read_text())
        if m["split"] in ("val", "test"):
            metas.append((p, m["family"], "_t2_" in p.name))
    rng = random.Random(seed)
    rng.shuffle(metas)
    ev = lambda p: S.load_event(p)  # noqa: E731
    if n == 1:
        for p, fam, t2 in metas:
            if fam == "anchored_wall" and not t2:
                e = ev(p)
                if _regime(e) == S.REGIMES["perforate"] and len(e.calls) >= 20:
                    return [p]
        raise SystemExit("no anchored t1 perforation with ≥ 20 calls")
    if n <= 20:
        out, need = [], {("anchored_wall", False), ("anchored_wall", True), ("free_wall", False),
                         ("free_wall", True)}
        regimes = set()
        for p, fam, t2 in metas:
            key = (fam, t2)
            e = ev(p)
            r = _regime(e)
            want_regime = (r == S.REGIMES["perforate"] and "perf" not in regimes) or \
                          (r == S.REGIMES["arrest"] and "arrest" not in regimes)
            if key in need or want_regime or len(out) < n - (len(need) + 2 - len(regimes)):
                out.append(p)
                need.discard(key)
                if r == S.REGIMES["perforate"]:
                    regimes.add("perf")
                if r == S.REGIMES["arrest"]:
                    regimes.add("arrest")
            if len(out) == n:
                break
        return out
    by_fam = defaultdict(list)
    for p, fam, _t2 in metas:
        by_fam[fam].append(p)
    out, fams = [], sorted(by_fam)
    while len(out) < n and any(by_fam.values()):
        for f in fams:
            if by_fam[f] and len(out) < n:
                out.append(by_fam[f].pop())
    return out


def ensemble_spread(root: Path, event_dir: Path) -> dict[tuple[int, int], float]:
    """Per (call index, chunk key): RMS over the other members of |v_member − v_e0| at frame
    end, for bodies present in both (the oracle's own variability at that call)."""
    base = S.load_event(event_dir)
    stem = event_dir.name.rsplit("_e", 1)[0]
    others = [S.load_event(p) for p in sorted(root.glob(stem + "_e*")) if p != event_dir]
    out = {}
    for c in base.calls:
        keys = c.arrays["chunk_key"]
        acc = defaultdict(list)
        for o in others:
            oc = next((x for x in o.calls if x.call_index == c.call_index), None)
            if oc is None:
                continue
            idx = {int(k): i for i, k in enumerate(oc.arrays["chunk_key"])}
            for i, k in enumerate(keys.tolist()):
                if k in idx:
                    acc[k].append(float(np.linalg.norm(oc.arrays["y_chunk_v_end"][idx[k]]
                                                       - c.arrays["y_chunk_v_end"][i])))
        for k, v in acc.items():
            out[(c.call_index, k)] = float(np.sqrt(np.mean(np.square(v))))
    return out


@torch.no_grad()
def evaluate_memorisation(ckpt: Path, data: Data, spreads: dict) -> dict:
    model, ck = load_model(ckpt)
    model.eval()
    dcfg = D.DecodeConfig(tau=1.0, energy_tolerance=ck["train_config"]["decode"]["energy_tolerance"])
    cache = HierarchyCache(model.cfg.levels)
    tp = fp = fn = 0
    v_err, v_spread, i_err, frag_same, n_calls = [], [], [], 0, 0
    mom_res, ke_excess, bound_active = [], [], 0
    for bc in _batches(data.val, 20000, 64):
        b = collate(bc, data.structures, cache)
        out = model(b)
        dec = D.decode(out, b, dcfg)
        t = b.t
        pred = dec["bond_broke"].numpy().astype(bool)
        truth = t["y_bond_broke"].numpy() > 0
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        rows = M._per_call_outcomes(b, dec)
        for r in rows:
            n_calls += 1
            frag_same += int(M.weighted_ari(r["pred"], r["true"], r["mass"]) > 1 - 1e-9)
        cv = dec["chunk_v_end"].numpy()
        yv = t["y_chunk_v_end"].numpy()
        cc = t["chunk_call"].numpy()
        for g, c in enumerate(b.calls):
            sel = np.nonzero(cc == g)[0]
            for j, k in zip(sel, c.arrays["chunk_key"].tolist()):
                s = spreads.get((c.event_id, c.call_index, k))
                e = float(np.linalg.norm(cv[j] - yv[j]))
                v_err.append(e / c.scales["v"])
                if s is not None:
                    v_spread.append((e, s))
        iv = dec["imp_v_end"].numpy()
        if len(iv):
            ic = t["imp_call"].numpy()
            vr = t["scales"][:, SCALE_INDEX["v"]].numpy()
            i_err.extend((np.linalg.norm(iv - t["y_imp_v_end"].numpy(), axis=1) / vr[ic]).tolist())
        # invariants on free islands: momentum (inputs + gravity impulse) and KE
        ph = dec["_phys"]
        G = b.n_calls
        free = dec["free"].numpy()
        pos0 = D._com_rel(ph, ph["pos"], G)
        p0, _ = D.momentum(ph, ph["v"], ph["w"], pos0, ph["inertia_w"], G)
        p1, _ = D.momentum(ph, ph["v_end"], ph["w_end"], pos0, ph["inertia_w"], G)
        sc = t["scales"].double()
        dt = sc[:, SCALE_INDEX["dt"]].unsqueeze(-1)
        mt = D.scatter_sum(ph["mass"].unsqueeze(-1), ph["call"], G)
        want = p0 + mt * t["gravity"].double() * dt
        res = ((p1 - want).norm(dim=-1) / (mt.squeeze(-1) * sc[:, SCALE_INDEX["v"]])).numpy()
        mom_res.extend(res[free[:G]].tolist())
        en = t["energy"].double()
        ke = D.scatter_sum(0.5 * ph["mass"] * (ph["v_end"] ** 2).sum(-1)
                           + 0.5 * (ph["w_end"].unsqueeze(-2) @ ph["inertia_w"]
                                    @ ph["w_end"].unsqueeze(-1)).reshape(-1), ph["call"], G)
        e_in = sc[:, SCALE_INDEX["E_in"]]
        excess = ((ke - en[:, 0] - en[:, 2] - en[:, 3]) / e_in - dcfg.energy_tolerance).numpy()
        ke_excess.extend(excess[:len(b.calls)].tolist())
        bound_active += int(dec["energy_bound_active"][:len(b.calls)].sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    ratio = [e / s for e, s in v_spread if s > 0]
    return {
        "calls": n_calls, "bond_tp_fp_fn": [tp, fp, fn], "bond_precision": p, "bond_recall": r,
        "bond_f1": 2 * p * r / max(p + r, 1e-12),
        "fragments_identical": frag_same / max(n_calls, 1),
        "chunk_v_err_over_v_median_p99": [float(np.median(v_err)), float(np.quantile(v_err, 0.99))],
        "chunk_v_err_over_ensemble_spread_median_p90": (
            [float(np.median(ratio)), float(np.quantile(ratio, 0.9))] if ratio else None),
        "impactor_v_err_over_v_median_max": ([float(np.median(i_err)), float(np.max(i_err))]
                                             if i_err else None),
        "momentum_residual_free_max": float(np.max(mom_res)) if mom_res else None,
        "ke_excess_over_bound_max": float(np.max(ke_excess)) if ke_excess else None,
        "energy_bound_active_rate": bound_active / max(n_calls, 1),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True, help="P1's evaluation calls (full sequences)")
    ap.add_argument("--n-events", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    root, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = select(root, a.n_events, a.seed)
    (out / "events.json").write_text(json.dumps([p.name for p in paths], indent=1))
    print(f"[overfit] {len(paths)} events: {[p.name for p in paths][:12]}")
    events = [S.load_event(p) for p in paths]
    d = Data.__new__(Data)
    d.events = events
    d.structures = {e.manifest["event_id"]: e.structures for e in events}
    calls = [c for e in events for c in e.calls if c.routed]
    for c in calls:  # one member per ensemble: train and score on its own hard targets
        c.arrays = {k: v for k, v in c.arrays.items()
                    if k not in ("y_bond_break_freq", "y_impactor_perforate_freq", "ens_impactor_v_end")}
    d.train, d.val = calls, calls
    d.successor, d.ensemble_aligned_calls = {}, 0
    pos = sum(float(c.arrays["y_bond_broke"].sum()) for c in calls)
    tot = sum(float(len(c.arrays["y_bond_broke"])) for c in calls)
    d.priors = {f: max(pos / max(tot, 1), 1e-4) for f in {c.family for c in calls}}
    cfg = TrainConfig(dataset=str(root), out=str(out), steps=a.steps, lr=a.lr, seed=a.seed,
                      device=a.device, weight_decay=0.0, noise_sigma=0.0, pushforward_prob=0.0,
                      eval_every=max(a.steps // 10, 50), select_tau=False,
                      curve_calls=len(calls))
    cfg.loss = dataclasses.replace(cfg.loss, label_smoothing=0.0)
    print(f"[overfit] {len(calls)} calls, {int(pos)} broken bonds, W{cfg.model.width} "
          f"rounds {cfg.model.rounds}")
    train(cfg, d)
    spreads = {}
    for p in paths:
        eid = json.loads((p / "manifest.json").read_text())["event_id"]
        for (ci, k), s in ensemble_spread(root, p).items():
            spreads[(eid, ci, k)] = s
    rows = [json.loads(x) for x in (out / "history.jsonl").read_text().splitlines()]
    ev = [r for r in rows if r["kind"] == "eval"]
    res = evaluate_memorisation(out / "last.pt", d, spreads)
    res["loss_floor"] = ev[-1]["train"]["terms"] if ev else None
    res["loss_total_first_last"] = [ev[0]["train"]["loss"], ev[-1]["train"]["loss"]] if ev else None
    res["events"] = [p.name for p in paths]
    (out / "memorisation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    main()
