"""Validation metrics (plan, "Objectives"): checkpoints are selected by these, not by loss.

- ensemble-relative skill: the model's mean distance to the members ÷ a typical member's
  mean distance to the others (1 = indistinguishable from one more oracle run);
- mass-weighted adjusted Rand index of the predicted partition against each member;
- Wasserstein-1 distance on log fragment mass (mass-weighted), against the members' own
  pairwise spread;
- regime calibration (Brier, ECE) of P(perforate); bond precision / recall / F1 pooled
  against the consensus; false breaks on no-break calls;
- the Lambert–Jonas ballistic-limit fit for the curve gates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from . import decode as D
from . import schema as S
from .batch import SCALE_INDEX, Batch


# ---- partitions and sizes -------------------------------------------------------------
def weighted_ari(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    """Adjusted Rand index where a pair (i, j) counts with weight w_i·w_j (mass-weighted);
    equals the standard ARI for unit weights."""
    a = np.asarray(a)
    b = np.asarray(b)
    w = np.ones(len(a)) if w is None else np.asarray(w, dtype=np.float64)
    if len(a) < 2:
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)

    def pairs(idx: np.ndarray, k: int) -> np.ndarray:
        s = np.bincount(idx, weights=w, minlength=k)
        s2 = np.bincount(idx, weights=w * w, minlength=k)
        return 0.5 * (s * s - s2)

    cell = ai * (bi.max() + 1) + bi
    c = pairs(cell, int(cell.max()) + 1).sum()
    pa = pairs(ai, int(ai.max()) + 1).sum()
    pb = pairs(bi, int(bi.max()) + 1).sum()
    tot = 0.5 * (w.sum() ** 2 - (w * w).sum())
    expected = pa * pb / tot
    mx = 0.5 * (pa + pb)
    if mx == expected:
        return 1.0
    return float((c - expected) / (mx - expected))


def fragment_masses(labels: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.bincount(np.unique(labels, return_inverse=True)[1], weights=mass)


def wasserstein_log_mass(ma: np.ndarray, mb: np.ndarray) -> float:
    """W1 between the mass-weighted distributions of ln(fragment mass)."""
    xa, xb = np.log(np.asarray(ma)), np.log(np.asarray(mb))
    wa, wb = np.asarray(ma) / np.sum(ma), np.asarray(mb) / np.sum(mb)
    xs = np.sort(np.concatenate([xa, xb]))
    oa, ob = np.argsort(xa), np.argsort(xb)
    ca = np.searchsorted(xa[oa], xs[:-1], side="right")
    cb = np.searchsorted(xb[ob], xs[:-1], side="right")
    cdfa = np.concatenate([[0], np.cumsum(wa[oa])])[ca]
    cdfb = np.concatenate([[0], np.cumsum(wb[ob])])[cb]
    return float(np.sum(np.abs(cdfa - cdfb) * np.diff(xs)))


# ---- ensemble-relative skill -----------------------------------------------------------
def skill(pred, members: list, dist) -> float:
    """d(model, members) ÷ mean over members of d(member, other members)."""
    m = len(members)
    if m < 2:
        return float("nan")
    d_model = np.mean([dist(pred, y) for y in members])
    d_member = np.mean([np.mean([dist(members[i], members[k]) for k in range(m) if k != i])
                        for i in range(m)])
    return float(d_model / max(d_member, 1e-12))


def crps_ensemble(x: float, members: np.ndarray) -> float:
    y = np.asarray(members, dtype=np.float64)
    return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(y[:, None] - y[None, :])))


# ---- calibration ---------------------------------------------------------------------
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)) if len(p) else float("nan")


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for k in range(bins):
        sel = idx == k
        if sel.any():
            tot += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(tot)


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Validation-only temperature scaling for p_b (grid search on log-loss)."""
    best, best_t = np.inf, 1.0
    for t in np.exp(np.linspace(np.log(0.25), np.log(8.0), 61)):
        p = 1 / (1 + np.exp(-logits / t))
        ll = -np.mean(targets * np.log(p + 1e-12) + (1 - targets) * np.log(1 - p + 1e-12))
        if ll < best:
            best, best_t = ll, float(t)
    return best_t


# ---- ballistic limit -------------------------------------------------------------------
def lambert_jonas(v: np.ndarray, a: float, p: float, v_bl: float) -> np.ndarray:
    v = np.asarray(v, float)
    return np.where(v > v_bl, a * np.maximum(v ** p - v_bl ** p, 0) ** (1 / p), 0.0)


def fit_ballistic_limit(v0: np.ndarray, v_exit: np.ndarray) -> dict[str, float]:
    """Least-squares Lambert–Jonas fit `v_r = a (v0^p − v_bl^p)^(1/p)`; `a` in closed form
    for each (v_bl, p) on a grid. Arrest/rebound exits count as 0."""
    v0 = np.asarray(v0, float)
    vr = np.maximum(np.asarray(v_exit, float), 0.0)
    best = (np.inf, 1.0, 2.0, float(v0.min()))
    for v_bl in np.linspace(0.0, float(v0.max()), 201):
        for p in (1.5, 2.0, 2.5, 3.0, 4.0):
            basis = np.where(v0 > v_bl, np.maximum(v0 ** p - v_bl ** p, 0) ** (1 / p), 0.0)
            den = float(basis @ basis)
            a = float(basis @ vr) / den if den > 0 else 0.0
            a = min(max(a, 0.0), 1.0)
            err = float(np.sum((a * basis - vr) ** 2))
            if err < best[0]:
                best = (err, a, p, float(v_bl))
    return {"rss": best[0], "a": best[1], "p": best[2], "v_bl": best[3]}


# ---- evaluation over a validation set --------------------------------------------------
@dataclass
class EvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_family: dict[str, dict[str, float]] = field(default_factory=dict)
    score: float = float("inf")  # lower is better; the selection criterion


def _per_call_outcomes(b: Batch, dec: dict) -> list[dict]:
    """Per call: predicted and target partitions, masses, impactor exit speeds."""
    t = b.t
    frag_pred = D.fragments(b, dec["bond_broke"])
    out = []
    chunk_call = t["chunk_call"].cpu().numpy()
    imp_call = t["imp_call"].cpu().numpy()
    bond_call = t["bond_call"].cpu().numpy()
    mass = t["chunk_mass"].cpu().numpy().astype(np.float64)
    yfrag = t["y_chunk_fragment"].cpu().numpy()
    iv_hat = dec["imp_v_end"].detach().cpu().numpy()
    iv = t["y_imp_v_end"].cpu().numpy()
    iv0 = t["imp_v"].cpu().numpy()
    broke_hat = dec["bond_broke"].cpu().numpy()
    broke = t["y_bond_broke"].cpu().numpy() > 0
    freq = t["y_bond_freq"].cpu().numpy()
    has_freq = t["y_bond_has_freq"].cpu().numpy()
    v_ref = t["scales"][:, SCALE_INDEX["v"]].cpu().numpy()
    for g, c in enumerate(b.calls):
        cs = chunk_call == g
        is_ = imp_call == g
        bs = bond_call == g
        u0 = iv0[is_]
        dirn = u0 / np.maximum(np.linalg.norm(u0, axis=1, keepdims=True), 1e-9)
        out.append({
            "call": c, "mass": mass[cs], "pred": frag_pred[cs], "true": yfrag[cs],
            "imp_speed_hat": np.sum(iv_hat[is_] * dirn, axis=1),
            "imp_speed": np.sum(iv[is_] * dirn, axis=1), "v_ref": float(v_ref[g]),
            "imp_v0": np.linalg.norm(u0, axis=1),
            "broke_hat": broke_hat[bs], "broke": broke[bs],
            "freq": np.where(has_freq[bs], freq[bs], broke[bs].astype(float)),
        })
    return out


@torch.no_grad()
def evaluate(model, batches, decode_cfg: D.DecodeConfig | None = None,
             priors: dict[str, float] | None = None) -> EvalResult:
    """Run the model, decode, and compute the skill metrics. Ensemble members are grouped
    by (ensemble_id, call_index) when present; otherwise each call is scored against its
    single run (errors, not skill)."""
    from .losses import prior_logit_shift

    dcfg = decode_cfg or D.DecodeConfig()
    model.eval()
    rows: list[dict] = []
    bond_p, bond_y, imp_p, imp_y = [], [], [], []
    bound_active = bound_total = 0
    for b in batches:
        dev = next(model.parameters()).device
        bd = b.to(dev)
        out = model(bd)
        bc = bd if dev.type != "mps" else bd.to("cpu")
        out = {k: v.cpu() for k, v in out.items()} if dev.type == "mps" else out
        dec = D.decode(out, bc, dcfg)
        bound_active += int(dec["energy_bound_active"].sum())
        bound_total += bc.n_calls
        rows.extend(_per_call_outcomes(bc, dec))
        shift = prior_logit_shift(bc, priors)
        logit = out["bond_break_logit"] + (shift if shift is not None else 0)
        bond_p.append(torch.sigmoid(logit / dcfg.temperature).cpu().numpy())
        bond_y.append(np.where(bc.t["y_bond_has_freq"].numpy(), bc.t["y_bond_freq"].numpy(),
                               (bc.t["y_bond_broke"].numpy() > 0).astype(float)))
        reg = bc.t["y_imp_regime"].numpy()
        valid = reg != S.REGIME_UNDECIDED
        imp_p.append(torch.sigmoid(out["imp_perf_logit"]).cpu().numpy()[valid])
        imp_y.append(np.where(bc.t["y_imp_has_freq"].numpy(), bc.t["y_imp_perf_freq"].numpy(),
                              (reg == S.REGIMES["perforate"]).astype(float))[valid])
    res = EvalResult()
    bp, by = np.concatenate(bond_p), np.concatenate(bond_y)
    ip, iy = np.concatenate(imp_p), np.concatenate(imp_y)
    pred_all = np.concatenate([r["broke_hat"] for r in rows])
    cons_all = np.concatenate([r["freq"] for r in rows]) >= 0.5
    p, r_, f = prf(pred_all, cons_all)
    nobreak = [r for r in rows if not r["broke"].any()]
    m = res.metrics
    m.update(bond_precision=p, bond_recall=r_, bond_f1=f, bond_brier=brier(bp, by),
             bond_ece=ece(bp, by), regime_brier=brier(ip, iy), regime_ece=ece(ip, iy),
             false_break_rate=float(np.mean(np.concatenate([r["broke_hat"] for r in nobreak])))
             if nobreak else 0.0,
             energy_bound_rate=bound_active / max(bound_total, 1))
    # group ensemble members
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        c = r["call"]
        key = (c.ensemble_id, c.call_index) if c.ensemble_id else (c.event_id, c.call_index)
        groups[key].append(r)
    per = defaultdict(lambda: defaultdict(list))
    for _key, rs in groups.items():
        fam = rs[0]["call"].family
        x = rs[0]
        if len(rs) >= 2:  # ensemble: the model's prediction is scored against the members
            pm = x["pred"]
            members = [r["true"] for r in rs]
            per[fam]["skill_partition"].append(skill(pm, members, lambda a, b_: 1 - weighted_ari(
                a, b_, x["mass"])))
            ms = [fragment_masses(r["true"], r["mass"]) for r in rs]
            per[fam]["skill_sizes"].append(skill(fragment_masses(pm, x["mass"]), ms,
                                                 wasserstein_log_mass))
            if len(x["imp_speed"]):
                per[fam]["skill_ram"].append(skill(float(x["imp_speed_hat"][0]),
                                                   [float(r["imp_speed"][0]) for r in rs],
                                                   lambda a, b_: abs(a - b_)))
                per[fam]["crps_ram"].append(crps_ensemble(float(x["imp_speed_hat"][0]),
                                                          [float(r["imp_speed"][0]) for r in rs])
                                            / x["v_ref"])
        for r in rs:
            per[fam]["one_minus_ari"].append(1 - weighted_ari(r["pred"], r["true"], r["mass"]))
            per[fam]["w1_log_mass"].append(wasserstein_log_mass(
                fragment_masses(r["pred"], r["mass"]), fragment_masses(r["true"], r["mass"])))
            if len(r["imp_speed"]):
                per[fam]["ram_err_rel"].append(float(np.mean(np.abs(r["imp_speed_hat"]
                                                                    - r["imp_speed"])))
                                               / r["v_ref"])
            nf_hat = len(np.unique(r["pred"]))
            nf = len(np.unique(r["true"]))
            per[fam]["fragment_count_ratio"].append(nf_hat / max(nf, 1))
    for fam, d in per.items():
        res.per_family[fam] = {k: float(np.nanmean(v)) for k, v in d.items() if len(v)}
    keys = sorted({k for d in res.per_family.values() for k in d})
    for k in keys:
        m[k] = float(np.nanmean([d[k] for d in res.per_family.values() if k in d]))
    # selection score: ensemble skills when present (≤ 1.25 is the bar), else single-run
    # errors scaled to the plan's tolerances (ram 5% of v0)
    if "skill_ram" in m:
        parts = [m.get("skill_ram"), m.get("skill_partition"), m.get("skill_sizes")]
    else:
        parts = [m.get("ram_err_rel", 0.0) / 0.05, m.get("one_minus_ari", 0.0) * 4,
                 m.get("w1_log_mass", 0.0), (1 - f) * 4]
    worst_family = max((np.nanmean([d.get("ram_err_rel", 0) / 0.05,
                                    d.get("one_minus_ari", 0) * 4])
                        for d in res.per_family.values()), default=0.0)
    res.score = float(np.nanmean([x for x in parts if x is not None]) + 0.25 * worst_family)
    m["score"] = res.score
    model.train()
    return res
