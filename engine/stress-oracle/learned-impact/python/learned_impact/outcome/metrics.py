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
def skill(pred, members: list, dist, floor: float = 1e-12) -> float:
    """d(model, members) ÷ max(mean over members of d(member, other members), floor).

    `floor` is the metric's physical tolerance ÷ the literal bar, so where the ensemble
    agrees exactly the rule collapses to "within tolerance of the members" (the plan:
    "on scenes where the ensemble agrees exactly, that collapses to exact agreement")
    instead of dividing by zero."""
    m = len(members)
    if m < 2:
        return float("nan")
    d_model = np.mean([dist(pred, y) for y in members])
    d_member = np.mean([np.mean([dist(members[i], members[k]) for k in range(m) if k != i])
                        for i in range(m)])
    return float(d_model / max(d_member, floor, 1e-12))


LITERAL_BAR = 1.25
ACTIVE_PHASES = ("contact", "secondary", "blast")


# Physical tolerances per metric (the floor of the member spread is tolerance ÷ literal bar):
# ram = the plan's max(IQR, 0.05·v0) (as a fraction of v_ref); partition 1 − ARI 0.05;
# fragment sizes W1 on ln(mass) 0.1. Pre-registered with the gates (EVALUATION.md §1).
TOLERANCE = {"ram": 0.05, "partition": 0.05, "sizes": 0.1}


def heldout_member_skills(members: list, dist, floor: float = 1e-12) -> np.ndarray:
    """Each oracle member scored against the other members exactly as a candidate is."""
    m = len(members)
    if m < 3:
        return np.array([])
    return np.array([skill(members[i], members[:i] + members[i + 1:], dist, floor)
                     for i in range(m)])


def calibrated_gate(pred, members: list, dist, literal: float = LITERAL_BAR,
                    tolerance: float = 0.0) -> dict:
    """The user's post-P0 skill rule: a candidate passes a metric when its skill is within
    what a held-out oracle member reaches against the other members, i.e.
    `skill ≤ max(literal, mean held-out member skill)` (P0's "calibrated" verdict; the oracle
    itself fails the literal 1.25 on some scenes, with means of 1.0–1.42). The literal
    verdict and the held-out range are reported alongside."""
    floor = tolerance / literal
    s = skill(pred, members, dist, floor)
    h = heldout_member_skills(members, dist, floor)
    mean = float(np.mean(h)) if len(h) else float("nan")
    bar = max(literal, mean) if len(h) else literal
    return {"skill": s, "heldout_mean": mean,
            "heldout_min": float(np.min(h)) if len(h) else float("nan"),
            "heldout_max": float(np.max(h)) if len(h) else float("nan"),
            "bar": bar, "pass": bool(s <= bar), "pass_literal": bool(s <= literal)}


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
GATE_METRICS = ("ram", "partition", "sizes")


def _loo_gates(D: np.ndarray, tol: float, literal: float = LITERAL_BAR) -> tuple[float, float]:
    """Each member held out and scored against the others by `calibrated_gate`'s rule, from
    the members' pairwise distance matrix (same numbers as calling `calibrated_gate` per
    member, without recomputing distances): mean pass and mean skill ÷ bar."""
    m = len(D)
    floor = max(tol / literal, 1e-12)

    def skill_in(i: int, pool: list[int]) -> float:  # member i against `pool` (i ∉ pool)
        if len(pool) < 2:
            return float("nan")
        d_c = D[i, pool].mean()
        d_m = np.mean([D[j, [k for k in pool if k != j]].mean() for j in pool])
        return float(d_c / max(d_m, floor))

    passes, sob = [], []
    for i in range(m):
        others = [k for k in range(m) if k != i]
        s_ = skill_in(i, others)
        h = [skill_in(j, [k for k in others if k != j]) for j in others] if len(others) >= 3 else []
        bar = max(literal, float(np.mean(h))) if h else literal
        passes.append(s_ <= bar)
        sob.append(s_ / bar)
    return float(np.mean(passes)), float(np.mean(sob))


def _acceptance_row(rs: list[dict], gates: dict, pm, mass) -> dict:
    """One active ensemble call: the model's pass and skill ÷ bar per metric, and the same
    for each held-out oracle member scored by the identical rule against the other members
    (leave-one-out: the member's own bar comes from the remaining members)."""
    trues = [r["true"] for r in rs]
    sizes = [fragment_masses(r["true"], r["mass"]) for r in rs]
    v_ref = rs[0]["v_ref"]
    spec = {"partition": (trues, lambda a, b_: 1 - weighted_ari(a, b_, mass), TOLERANCE["partition"]),
            "sizes": (sizes, wasserstein_log_mass, TOLERANCE["sizes"])}
    if "ram" in gates:
        spec["ram"] = ([float(r["imp_speed"][0]) for r in rs], lambda a, b_: abs(a - b_),
                       TOLERANCE["ram"] * v_ref)
    row = {"family": rs[0]["call"].family, "ensemble": rs[0]["call"].ensemble_id, "model": {},
           "member": {}}
    for name, (ys, dist, tol) in spec.items():
        g = gates[name]
        row["model"][name] = (float(g["pass"]), g["skill"] / g["bar"])
        m = len(ys)
        D = np.zeros((m, m))
        for i in range(m):
            for k in range(i + 1, m):
                D[i, k] = D[k, i] = dist(ys[i], ys[k])
        row["member"][name] = _loo_gates(D, tol)
    return row


# gates-v1.1 (EVALUATION.md §1): non-inferiority margins against the held-out member
NONINF_PASS_POINTS = 0.05  # pass rate: at most 5 points below the held-out member
NONINF_SKILL_RATIO = 0.05  # mean skill ÷ bar: at most 5% above the held-out member's


def family_acceptance(rows: list[dict], n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """The pre-registered ensemble gate (EVALUATION.md §1), per family and metric, over the
    family's active ensemble calls, against the held-out oracle member scored by the
    identical rule. Bootstraps resample whole ensembles (calls of one ensemble are
    correlated).

    gates-v1.1 (`accept`, the current gate): the one-sided (1 − alpha) upper bootstrap bound
    of (member pass rate − model pass rate) ≤ 5 points, and of (model mean skill÷bar ÷
    member mean − 1) ≤ 5%.
    gates-v1 (`accept_v1`, kept reported): point estimates, model pass rate ≥ member's and
    model mean skill÷bar ≤ member's; `confident` adds that the two-sided CI of each
    difference excludes the failing side."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam]
        ens = sorted({r["ensemble"] for r in fr})
        idx = {e: [i for i, r in enumerate(fr) if r["ensemble"] == e] for e in ens}
        fam_out = {"n_ensembles": len(ens), "n_calls": len(fr)}
        for name in GATE_METRICS:
            sel = [i for i, r in enumerate(fr) if name in r["model"]]
            if not sel:
                continue
            a = np.array([[*fr[i]["model"][name], *fr[i]["member"][name]] for i in sel])
            pos = {i: k for k, i in enumerate(sel)}
            groups = [[pos[i] for i in idx[e] if i in pos] for e in ens]
            groups = [g for g in groups if g]

            def stats(x):
                mp, ms, hp, hs = x.mean(0)
                return np.array([mp, hp, mp - hp, ms, hs, ms - hs, hp - mp,
                                 ms / max(hs, 1e-12) - 1])
            point = stats(a)
            boot = np.array([stats(a[np.concatenate([groups[j] for j in
                                                     rng.integers(0, len(groups), len(groups))])])
                             for _ in range(n_boot)])
            lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
            upper = np.quantile(boot, 1 - alpha, axis=0)  # one-sided bound
            ci = lambda k: [float(lo[k]), float(hi[k])]  # noqa: E731
            accept_v1 = bool(point[2] >= 0 and point[5] <= 0)
            accept = bool(upper[6] <= NONINF_PASS_POINTS and upper[7] <= NONINF_SKILL_RATIO)
            fam_out[name] = {
                "model_pass_rate": float(point[0]), "model_pass_rate_ci": ci(0),
                "member_pass_rate": float(point[1]), "member_pass_rate_ci": ci(1),
                "pass_rate_diff_ci": ci(2),
                "model_mean_skill_over_bar": float(point[3]), "model_mean_skill_over_bar_ci": ci(3),
                "member_mean_skill_over_bar": float(point[4]),
                "member_mean_skill_over_bar_ci": ci(4), "skill_over_bar_diff_ci": ci(5),
                "pass_shortfall_upper": float(upper[6]),
                "skill_ratio_excess_upper": float(upper[7]),
                "accept": accept, "accept_v1": accept_v1,
                "confident": bool(accept_v1 and lo[2] >= 0 and hi[5] <= 0)}
        metric_entries = [fam_out[k] for k in GATE_METRICS if k in fam_out]
        fam_out["accept"] = bool(metric_entries) and all(e["accept"] for e in metric_entries)
        fam_out["accept_v1"] = bool(metric_entries) and all(e["accept_v1"] for e in metric_entries)
        out[fam] = fam_out
    return out


@dataclass
class EvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_family: dict[str, dict[str, float]] = field(default_factory=dict)
    score: float = float("inf")  # lower is better; the selection criterion
    ensemble_rows: list = field(default_factory=list)  # per active ensemble call: gate records
    acceptance: dict = field(default_factory=dict)  # per family: the pre-registered gate


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
             priors: dict[str, float] | None = None, acceptance: bool = False) -> EvalResult:
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
        bc = bd if dev.type == "cpu" else bd.to("cpu")
        out = {k: v.cpu() for k, v in out.items()} if dev.type != "cpu" else out
        dec = D.decode(out, bc, dcfg)
        bound_active += int(dec["energy_bound_active"][:len(bc.calls)].sum())
        bound_total += len(bc.calls)
        rows.extend(_with_clamps(_per_call_outcomes(bc, dec), dec))
        shift = prior_logit_shift(bc, priors)
        logit = out["bond_break_logit"] + (shift if shift is not None else 0)
        bv = bc.t["bond_valid"].numpy()
        bond_p.append(torch.sigmoid(logit / dcfg.temperature).cpu().numpy()[bv])
        bond_y.append(np.where(bc.t["y_bond_has_freq"].numpy(), bc.t["y_bond_freq"].numpy(),
                               (bc.t["y_bond_broke"].numpy() > 0).astype(float))[bv])
        reg = bc.t["y_imp_regime"].numpy()
        valid = reg != S.REGIME_UNDECIDED
        imp_p.append(torch.sigmoid(out["imp_perf_logit"]).cpu().numpy()[valid])
        imp_y.append(np.where(bc.t["y_imp_has_freq"].numpy(), bc.t["y_imp_perf_freq"].numpy(),
                              (reg == S.REGIMES["perforate"]).astype(float))[valid])
    res = _summarise(rows, bond_p, bond_y, imp_p, imp_y, bound_active, bound_total, acceptance)
    model.train()
    return res


def _with_clamps(rows: list[dict], dec: dict) -> list[dict]:
    """Each call's admissibility clamps (node axes clamped into the kinematic envelope)."""
    n = dec["admissibility_clamps"].cpu().numpy()
    for i, r in enumerate(rows):
        r["clamps"] = int(n[i])
    return rows


def _summarise(rows, bond_p, bond_y, imp_p, imp_y, bound_active, bound_total,
               acceptance: bool = False) -> EvalResult:
    """`acceptance` adds the pre-registered family gate (held-out members, bootstrap CIs):
    for reports; training's validation passes select on the dense score and skip it."""
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
             energy_bound_rate=bound_active / max(bound_total, 1),
             # calls with ≥ 1 node axis clamped into the kinematic envelope, and clamped axes
             # per call (kinematic checkpoints: at the envelope's edge)
             admissibility_clamp_rate=float(np.mean([r.get("clamps", 0) > 0 for r in rows])),
             admissibility_clamps_per_call=float(np.mean([r.get("clamps", 0) for r in rows])))
    # group ensemble members
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        c = r["call"]
        if c.ensemble_id and "y_bond_break_freq" in c.arrays:
            # Members are scored together only on calls `ensemble.aggregate` marked aligned:
            # the same bodies, and inputs equal up to the ensemble's perturbation (after the
            # first contact the members' states diverge, and scoring one member's input
            # against the others' outcomes would compare different calls). Every other call
            # is scored against its own run.
            ck = c.arrays["chunk_key"]
            order = np.argsort(ck, kind="stable")
            r = dict(r, mass=r["mass"][order], pred=r["pred"][order], true=r["true"][order])
            key = (c.ensemble_id, c.call_index, hash(ck[order].tobytes()))
        else:
            key = (c.event_id, c.call_index, 0)
        groups[key].append(r)
    per = defaultdict(lambda: defaultdict(list))
    n_ens_groups = sum(1 for k, rs in groups.items() if len(rs) >= 3)
    for _key, rs in groups.items():
        fam = rs[0]["call"].family
        x = rs[0]
        # Active calls carry the gates: an impact or secondary contact, or a break in any
        # member. Quiet calls (most frames are settling) are reported apart with "_quiet":
        # pooled, "predict nothing" passes them and would hide a model that never breaks.
        active = (any(r["call"].phase in ACTIVE_PHASES for r in rs)
                  or any(r["broke"].any() for r in rs))
        sfx = "" if active else "_quiet"
        per[fam]["n_active" if active else "n_quiet"].append(1.0)
        if len(rs) >= 3:  # ensemble: the model's prediction is scored against the members
            pm = x["pred"]
            mass = x["mass"]
            gates = {
                "partition": calibrated_gate(pm, [r["true"] for r in rs],
                                             lambda a, b_: 1 - weighted_ari(a, b_, mass),
                                             tolerance=TOLERANCE["partition"]),
                "sizes": calibrated_gate(fragment_masses(pm, mass),
                                         [fragment_masses(r["true"], r["mass"]) for r in rs],
                                         wasserstein_log_mass, tolerance=TOLERANCE["sizes"]),
            }
            if all(len(r["imp_speed"]) for r in rs):  # every member still has the impactor
                speeds = [float(r["imp_speed"][0]) for r in rs]
                gates["ram"] = calibrated_gate(float(x["imp_speed_hat"][0]), speeds,
                                               lambda a, b_: abs(a - b_),
                                               tolerance=TOLERANCE["ram"] * x["v_ref"])
                per[fam]["crps_ram" + sfx].append(crps_ensemble(float(x["imp_speed_hat"][0]), speeds)
                                            / x["v_ref"])
            # dense per-call selection metrics: the model's prediction against every member
            per[fam]["dense_1mARI" + sfx].append(float(np.mean(
                [1 - weighted_ari(pm, r["true"], mass) for r in rs])))
            per[fam]["dense_w1_log_mass" + sfx].append(float(np.mean(
                [wasserstein_log_mass(fragment_masses(pm, mass), fragment_masses(r["true"], r["mass"]))
                 for r in rs])))
            if active and acceptance:
                res.ensemble_rows.append(_acceptance_row(rs, gates, pm, mass))
            for name, gt in gates.items():
                per[fam][f"skill_{name}" + sfx].append(gt["skill"])
                per[fam][f"skill_bar_{name}" + sfx].append(gt["bar"])
                per[fam][f"skill_over_bar_{name}" + sfx].append(gt["skill"] / gt["bar"])
                per[fam][f"pass_{name}" + sfx].append(float(gt["pass"]))
                per[fam][f"pass_literal_{name}" + sfx].append(float(gt["pass_literal"]))
            n_frag = [len(np.unique(r["true"])) for r in rs]
            nf_hat = len(np.unique(pm))
            per[fam]["fragments_in_range" + sfx].append(float(min(n_frag) <= nf_hat <= max(n_frag)))
        else:  # single runs: the dense metrics are their errors (CRPS of one member = |error|)
            for r in rs:
                per[fam]["dense_1mARI" + sfx].append(1 - weighted_ari(r["pred"], r["true"], r["mass"]))
                per[fam]["dense_w1_log_mass" + sfx].append(wasserstein_log_mass(
                    fragment_masses(r["pred"], r["mass"]), fragment_masses(r["true"], r["mass"])))
                if len(r["imp_speed"]):
                    per[fam]["crps_ram" + sfx].append(abs(float(r["imp_speed_hat"][0])
                                                          - float(r["imp_speed"][0])) / r["v_ref"])
        for r in rs:
            per[fam]["one_minus_ari" + sfx].append(1 - weighted_ari(r["pred"], r["true"], r["mass"]))
            per[fam]["w1_log_mass" + sfx].append(wasserstein_log_mass(
                fragment_masses(r["pred"], r["mass"]), fragment_masses(r["true"], r["mass"])))
            if len(r["imp_speed"]):
                per[fam]["ram_err_rel" + sfx].append(float(np.mean(np.abs(r["imp_speed_hat"]
                                                                    - r["imp_speed"])))
                                               / r["v_ref"])
            nf_hat = len(np.unique(r["pred"]))
            nf = len(np.unique(r["true"]))
            per[fam]["fragment_count_ratio" + sfx].append(nf_hat / max(nf, 1))
    for fam, d in per.items():
        res.per_family[fam] = {k: (float(np.sum(v)) if k in ("n_active", "n_quiet")
                                   else float(np.nanmean(v))) for k, v in d.items() if len(v)}
    keys = sorted({k for d in res.per_family.values() for k in d})
    for k in keys:
        m[k] = float(np.nanmean([d[k] for d in res.per_family.values() if k in d]))
    # Selection score (checkpoints, τ, temperature; validation only): the dense per-call
    # metrics on active calls — CRPS of the ram exit speed (÷ v_ref), 1 − ARI and W1 on log
    # mass against every member — each ÷ its physical tolerance, mean over families plus a
    # quarter of the worst family. The ensemble pass/fail gates (`family_acceptance`) are too
    # coarse to select on with ~10² ensembles per family; they are for reporting.
    def composite(d: dict) -> float:
        parts = [d.get("crps_ram", np.nan) / TOLERANCE["ram"],
                 d.get("dense_1mARI", np.nan) / TOLERANCE["partition"],
                 d.get("dense_w1_log_mass", np.nan) / TOLERANCE["sizes"]]
        parts = [x for x in parts if np.isfinite(x)]
        return float(np.mean(parts)) if parts else float("nan")
    fam_scores = [c for c in (composite(d) for d in res.per_family.values()) if np.isfinite(c)]
    res.score = (float(np.mean(fam_scores) + 0.25 * max(fam_scores)) if fam_scores
                 else float("inf"))
    if acceptance:
        res.acceptance = family_acceptance(res.ensemble_rows)
    m["score"] = res.score
    m["ensemble_aligned_groups"] = float(n_ens_groups)
    return res



# ---- three-way report: model vs blast_parity QS vs punch, same calls, same metrics -----
BASELINES = ("qs", "punch")
PRIMARY_ENSEMBLE = ("skill_over_bar_ram", "skill_over_bar_partition", "skill_over_bar_sizes")
PRIMARY_SINGLE = ("ram_err_rel", "one_minus_ari", "w1_log_mass")


def _baseline_rows(b: Batch, name: str) -> list[dict] | None:
    t = b.t
    kb, ki = f"base_{name}_bond_broke", f"base_{name}_impactor_v_end"
    if kb not in t or ki not in t:
        return None
    dec = {"bond_broke": t[kb].cpu() > 0, "imp_v_end": t[ki].cpu()}
    return _per_call_outcomes(b, dec)


def _qs_right(r: dict, tol_ram: float = 0.05, tol_ari: float = 0.05) -> bool:
    """A call QS already gets right (the non-inferiority subset), against its own run."""
    ram_ok = (not len(r["imp_speed"])) or float(np.mean(np.abs(r["imp_speed_hat"] - r["imp_speed"]))
                                             ) <= tol_ram * r["v_ref"]
    return ram_ok and 1 - weighted_ari(r["pred"], r["true"], r["mass"]) <= tol_ari


@torch.no_grad()
def three_way(model, batches, decode_cfg: D.DecodeConfig | None = None,
              priors: dict[str, float] | None = None,
              margin: tuple[float, float] | None = None) -> dict:
    """The plan's three-way report on identical calls and metrics, per family:
    `superiority` (the model beats QS and the punch on every primary metric), and
    `non_inferiority` (on calls QS already gets right, the model is not worse than QS by
    more than `margin`). Lower is better for every primary metric. Cases where the model is
    worse than QS are listed by event and call, never averaged away."""
    from .losses import prior_logit_shift

    dcfg = decode_cfg or D.DecodeConfig()
    model.eval()
    dev = next(model.parameters()).device
    rows = {"model": [], **{n: [] for n in BASELINES}}
    probs = {k: ([], [], [], []) for k in rows}
    bounds = [0, 0]
    for b in batches:
        bd = b.to(dev)
        out = model(bd)
        bc = bd if dev.type == "cpu" else bd.to("cpu")
        out = {k: v.cpu() for k, v in out.items()}
        dec = D.decode(out, bc, dcfg)
        bounds[0] += int(dec["energy_bound_active"][:len(bc.calls)].sum())
        bounds[1] += len(bc.calls)
        m_rows = _with_clamps(_per_call_outcomes(bc, dec), dec)
        rows["model"].extend(m_rows)
        y = np.where(bc.t["y_bond_has_freq"].numpy(), bc.t["y_bond_freq"].numpy(),
                     (bc.t["y_bond_broke"].numpy() > 0).astype(float))
        sh = prior_logit_shift(bc, priors)
        lg = out["bond_break_logit"] + (sh if sh is not None else 0)
        bv = bc.t["bond_valid"].numpy()
        probs["model"][0].append(torch.sigmoid(lg / dcfg.temperature).numpy()[bv])
        probs["model"][1].append(y[bv])
        for n in BASELINES:
            br = _baseline_rows(bc, n)
            if br is None:
                continue
            rows[n].extend(br)
            probs[n][0].append(bc.t[f"base_{n}_bond_broke"].numpy().astype(float)[bv])
            probs[n][1].append(y[bv])
    report: dict = {"families": {}, "worse_than_qs": [], "available": []}
    summaries = {}
    for k, rs in rows.items():
        if not rs:
            continue
        report["available"].append(k)
        pp = probs[k]
        summaries[k] = _summarise(rs, pp[0], pp[1], [np.zeros(0)], [np.zeros(0)],
                                  *(bounds if k == "model" else (0, 1)), acceptance=k == "model")
    if "model" not in summaries:
        return report
    ens = any(k in summaries["model"].metrics for k in PRIMARY_ENSEMBLE)
    primary = PRIMARY_ENSEMBLE if ens else PRIMARY_SINGLE
    for fam, mm in summaries["model"].per_family.items():
        entry = {"model": {k: mm.get(k) for k in primary}}
        for n in BASELINES:
            if n in summaries and fam in summaries[n].per_family:
                entry[n] = {k: summaries[n].per_family[fam].get(k) for k in primary}
        beats = [all(entry["model"][k] is not None and entry[n][k] is not None
                     and entry["model"][k] < entry[n][k] for k in primary)
                 for n in BASELINES if n in entry]
        entry["superiority"] = bool(beats) and all(beats)
        report["families"][fam] = entry
    if "qs" in rows and rows["qs"]:
        by_key = {(r["call"].event_id, r["call"].call_index): r for r in rows["qs"]}
        per_fam, agg = {}, {}
        for r in rows["model"]:
            key = (r["call"].event_id, r["call"].call_index)
            q = by_key.get(key)
            if q is None or not _qs_right(q):
                continue
            m_err = (1 - weighted_ari(r["pred"], r["true"], r["mass"]),
                     float(np.mean(np.abs(r["imp_speed_hat"] - r["imp_speed"]))) / r["v_ref"]
                     if len(r["imp_speed"]) else 0.0)
            q_err = (1 - weighted_ari(q["pred"], q["true"], q["mass"]),
                     float(np.mean(np.abs(q["imp_speed_hat"] - q["imp_speed"]))) / q["v_ref"]
                     if len(q["imp_speed"]) else 0.0)
            # "Not worse than QS within the spread": the gates' tolerances (1 − ARI, ram ÷ v)
            tol = margin if margin is not None else (TOLERANCE["partition"], TOLERANCE["ram"])
            worse = any(a > b_ + t_ for a, b_, t_ in zip(m_err, q_err, tol))
            fam = r["call"].family
            per_fam.setdefault(fam, []).append(not worse)
            agg.setdefault(fam, []).append((m_err, q_err))
            if worse:
                report["worse_than_qs"].append({"event": key[0], "call": key[1], "family": fam,
                                                "model_1mARI_ram": m_err, "qs_1mARI_ram": q_err})
        for fam, ok in per_fam.items():
            # Per family, on the calls QS gets right: the model's mean error is within the
            # tolerance of QS's (the plan: "not worse than QS within the spread"); the share
            # of calls not worse and the list of worse ones are reported beside it.
            me = np.mean([x[0] for x in agg[fam]], axis=0)
            qe = np.mean([x[1] for x in agg[fam]], axis=0)
            tol = margin if margin is not None else (TOLERANCE["partition"], TOLERANCE["ram"])
            report["families"].setdefault(fam, {})["non_inferiority"] = {
                "qs_right_calls": len(ok), "model_not_worse": float(np.mean(ok)),
                "model_mean_1mARI_ram": me.tolist(), "qs_mean_1mARI_ram": qe.tolist(),
                "pass": bool(np.all(me <= qe + np.array(tol)))}
    report["metrics"] = {k: s_.metrics for k, s_ in summaries.items()}
    report["acceptance"] = summaries["model"].acceptance
    report["selection_score"] = summaries["model"].score
    model.train()
    return report
