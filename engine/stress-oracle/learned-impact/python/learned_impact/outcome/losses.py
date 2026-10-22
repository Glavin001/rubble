"""Losses, exactly as the plan's "Loss" section, with P0's correction to `L_energy`.

Each term is divided by its value under the zero predictor on the training set
(`zero_baselines`), so every term starts near 1 (the V34 lesson). Weights λ start at 1, with
motion, impactor and the break margin at 2, and change only on validation metrics.

Change from the plan (evidence: the output-side ceiling on P1's smoke calls): the plan's
break test, peak utilisation r ≥ 1, is the *onset of softening* in this oracle, not failure
(8,338 surviving val bonds reached r ≥ 1, up to 116, against 4,595 breaks). The break
decision is the failure margin q = κ / κ_fail, which is ≥ 1 exactly when a bond fails; r
stays as an auxiliary dense target (`demand`, λ = 1).

`L_energy` uses kinetic-energy changes (exact) and per-bond dissipation, never the booked
contact dissipation: P0 found the penalty contact books its dashpot quadrature rather than
its work, so the ledger is open by +0.3–4.7% E0.

Additions the plan's list predates, both from P0: `pose` (the call must output frame-end
poses, so Δx and Δθ are supervised in the same kinetic-energy norm, ÷ dt²).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..mgn.mesh_blocks import scatter_sum
from . import decode as D
from . import schema as S
from .batch import SCALE_INDEX, Batch

from . import fullstate as _FS

# `/5`: one term per carried bond field (each normalised by its own persistence loss, so
# a field that rarely changes still has to beat "unchanged"), the mode and the reactions
FS_BOND_TERMS = tuple(f"fs_{name}" for name, _, _ in _FS.BOND_FIELDS)
HIDDEN_TERMS = ("hid_u", "hid_th", "hid_v", "hid_w")
FS_CLUSTER_TERMS = ("fs_cl_pos", "fs_cl_rot", "fs_cl_vel", "fs_cl_ang", "fs_cl_timer", "fs_cl_act")
TERMS = ("margin", "demand", "break", "time", "motion", "def_motion", "pose", "impactor",
         "energy", "damage", "regime", *FS_BOND_TERMS, "fs_mode", "fs_chunk", *FS_CLUSTER_TERMS,
         "break_t", *HIDDEN_TERMS)


@dataclass
class LossConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "margin": 5.0, "demand": 1.0, "break": 5.0, "time": 1.0, "motion": 2.0,
        "def_motion": 2.0, "pose": 1.0, "fs_mode": 1.0, "fs_chunk": 1.0, **{k: 1.0 for k in FS_BOND_TERMS}, **{k: 1.0 for k in FS_CLUSTER_TERMS},
        "break_t": 1.0, **{k: 1.0 for k in HIDDEN_TERMS},
        "impactor": 2.0, "energy": 1.0, "damage": 1.0, "regime": 1.0})
    huber_delta: float = 1.0
    label_smoothing: float = 0.02  # single-run hard break labels
    # "none": proper log-loss on the soft labels (default: the P2 sweep found the capped
    # prior reweighting miscalibrated p_b); "prior" (balanced, with the prior correction at
    # inference) and "focal" are ablations.
    break_reweight: str = "none"
    reweight_cap: float = 50.0
    focal_gamma: float = 2.0
    alpha_cut: float = 1.0  # consequence weight: bond on the separating cut
    beta_support: float = 1.0  # consequence weight: bond on a support load path
    energy_tolerance: float = 0.05
    log_r_floor: float = 1e-4
    # Broken bonds must clear the break threshold by this margin in log space (q̂ ≥ e^δ):
    # with δ = 0 the censored hinge stops at q̂ = 1⁺ and leaves every broken bond on the
    # decision boundary, where a small error turns it into a miss.
    break_margin: float = 0.2
    # Survivors' side of the decision: q̂ ≤ e^−δ_s (see `margin` in `loss_terms`).
    survive_margin: float = 0.05
    # "fixed": λ·L/L_zero (the plan's default; λ changes only on validation metrics).
    # "uncertainty": Kendall, Gal & Cipolla (CVPR 2018) homoscedastic weighting on top of
    # the fixed normalisation, an ablation that must beat "fixed" on validation metrics.
    balance: str = "fixed"
    # V34 detector: warn when one term carries more than this share of the summed per-term
    # gradient norms for `dominance_patience` consecutive measurements.
    dominance_warn: float = 0.6
    dominance_patience: int = 3


def bond_family(b: Batch) -> list[str]:
    return [c.family for c in b.calls]


def prior_logit_shift(b: Batch, priors: dict[str, float] | None) -> Tensor | None:
    """Logit correction from the balanced training prior (0.5) back to the family's
    positive rate π: `+ log(π/(1−π))`."""
    if not priors:
        return None
    fam = bond_family(b)
    per_call = torch.tensor([np.log(max(priors.get(f, 0.5), 1e-6) / max(1 - priors.get(f, 0.5),
                                                                        1e-6)) for f in fam]
                            + [0.0] * (b.n_calls - len(fam)))  # padding calls
    return per_call[b.t["bond_call"]].to(b.t["y_bond_broke"].device)


def _huber(x: Tensor, delta: float) -> Tensor:
    return F.huber_loss(x, torch.zeros_like(x), reduction="none", delta=delta)


def consequence_weights(b: Batch, cfg: LossConfig) -> Tensor:
    t = b.t
    node_to_chunk = torch.full((int(t["n_nodes"]),), -1, dtype=torch.long,
                               device=t["chunk_node"].device)
    node_to_chunk[t["chunk_node"]] = torch.arange(t["chunk_node"].shape[0],
                                                  device=node_to_chunk.device)
    be = b.edges["bond"]
    fs = t["y_chunk_fragment"][node_to_chunk[be["src"][0::2]]]
    fd = t["y_chunk_fragment"][node_to_chunk[be["dst"][0::2]]]
    on_cut = ((t["y_bond_broke"] > 0) & (fs != fd)).float()
    w = 1.0 + cfg.alpha_cut * on_cut + cfg.beta_support * t["y_bond_support_path"].float()
    return w * t["bond_valid"].to(w.device).float()  # padding bonds weigh nothing


def _per_call_mean(x: Tensor, call: Tensor, G: int) -> Tensor:
    s = scatter_sum(x, call, G)
    n = scatter_sum(torch.ones_like(x), call, G)
    return s / n.clamp(min=1)


def zero_outputs(b: Batch, n_time_bins: int = S.N_TIME_BINS) -> dict[str, Tensor]:
    t = b.t
    dev = t["chunk_x"].device
    nb = t["y_bond_broke"].shape[0]
    z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
    G = b.n_calls
    return {"bond_log_r": z(nb), "bond_log_q": z(nb), "bond_break_logit": z(nb), "bond_time_logits": z(nb, n_time_bins),
            "bond_ddamage": z(nb), "chunk_out": z(t["chunk_x"].shape[0], 12),
            "imp_out": z(t["imp_x"].shape[0], 12), "imp_perf_logit": z(t["imp_x"].shape[0]),
            "energy_d_bond": z(G), "energy_frac_joint": z(G) + 0.5, "energy_e_elastic": z(G),
            "energy_ke_loss": z(G),
            "bond_break_frac": z(nb) + 0.5,
            **({"chunk_hidden": t["chunk_x"][:, [S.col("chunk_feat", n) for n in _FS.HIDDEN_COLS]].float()}
               if "y_chunk_hid" in t else {}),
            # `/5`: persistence (frame end = frame start) is the full state's zero predictor
            **({"bond_fs": t["y_bond_fs_start"].float(), "bond_fs_mode_logits": z(nb, 3),
                "chunk_fs": t["chunk_fs_in"].float(),
                **({"clu_out": z(t["clu_x"].shape[0], 15)} if "clu_x" in t else {})}
               if "y_bond_fs" in t else {})}


def loss_terms(out: dict[str, Tensor], b: Batch, cfg: LossConfig,
               priors: dict[str, float] | None = None,
               decode_cfg: D.DecodeConfig | None = None) -> dict[str, Tensor]:
    """Unnormalised per-term losses (scalars). Velocity terms use the *projected*
    velocities, so the network learns inside the momentum-conserving space."""
    t = b.t
    G = b.n_calls
    dev = out["bond_log_r"].device
    dtype = torch.float64 if dev.type == "cpu" else torch.float32
    terms: dict[str, Tensor] = {}
    w_b = consequence_weights(b, cfg)
    bcall = t["bond_call"]
    broke = t["y_bond_broke"] > 0

    # demand: Huber on log r for survivors; one-sided hinge (penalise r̂ < 1) for broken
    log_r = torch.log(t["y_bond_peak_util"].clamp(min=cfg.log_r_floor))
    err = torch.where(broke, torch.relu(-out["bond_log_r"]), out["bond_log_r"] - log_r)
    terms["demand"] = (w_b * _huber(err, cfg.huber_delta)).sum() / w_b.sum().clamp(min=1)

    # margin (the break decision): q = κ / κ_fail reaches 1 exactly when the oracle's bond
    # fails, so the same censored treatment is exact here: Huber on log q for survivors,
    # the hinge "q̂ ≥ 1" for broken bonds (q saturates at failure)
    log_q = torch.log(t["y_bond_fail_margin"].clamp(min=cfg.log_r_floor))
    err_q = torch.where(broke, torch.relu(cfg.break_margin - out["bond_log_q"]),
                        out["bond_log_q"] - log_q)
    # Survivors must also clear the threshold, from below (q̂ ≤ e^−δ_s): the regression
    # alone costs ~(0.02)² for a survivor at q = 0.99 predicted at 1.01, yet that flips the
    # break decision (the real-data overfit: recall 1.0, but 96 false breaks on 10 events).
    hinge_s = torch.where(broke, torch.zeros_like(err_q),
                          torch.relu(out["bond_log_q"] + cfg.survive_margin))
    terms["margin"] = (w_b * (_huber(err_q, cfg.huber_delta) + _huber(hinge_s, cfg.huber_delta))
                       ).sum() / w_b.sum().clamp(min=1)

    # break: proper log-loss against the ensemble frequency (or a smoothed hard label)
    eps = cfg.label_smoothing
    hard = broke.float() * (1 - eps) + (~broke).float() * eps
    target = torch.where(t["y_bond_has_freq"], t["y_bond_freq"], hard)
    logit = out["bond_break_logit"]
    if cfg.break_reweight == "focal":
        p = torch.sigmoid(logit)
        pt = target * p + (1 - target) * (1 - p)
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        per = (1 - pt).pow(cfg.focal_gamma) * bce
    else:
        per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        if cfg.break_reweight == "prior" and priors:
            pi = torch.tensor([priors.get(c.family, 0.5) for c in b.calls]
                              + [0.5] * (b.n_calls - len(b.calls)), device=dev)[bcall]
            wp = (0.5 / pi.clamp(min=1e-6)).clamp(max=cfg.reweight_cap)
            wn = (0.5 / (1 - pi).clamp(min=1e-6)).clamp(max=cfg.reweight_cap)
            per = per * (target * wp + (1 - target) * wn)
    terms["break"] = (w_b * per).sum() / w_b.sum().clamp(min=1)

    # time: cross-entropy on the break-time bin, broken bonds only. Every term below is a
    # masked mean, never a host-side branch: `if mask.any()` would sync MPS every step.
    bk = broke.to(out["bond_time_logits"].dtype)
    ce = F.cross_entropy(out["bond_time_logits"], (t["y_bond_break_bin"] - 1).clamp(min=0),
                         reduction="none")
    terms["time"] = (ce * bk).sum() / bk.sum().clamp(min=1)

    # damage: Huber on Δdamage for unbroken bonds
    keep = ((~broke) & t["bond_valid"].to(dev)).to(out["bond_ddamage"].dtype)
    hub = _huber(out["bond_ddamage"] - t["y_bond_ddamage"], cfg.huber_delta * 0.1)
    terms["damage"] = (hub * keep).sum() / keep.sum().clamp(min=1) / (0.1 ** 2)

    # motion and pose in the kinetic-energy norm, ÷ E_in (and ÷ dt² for poses)
    dcfg = decode_cfg or D.DecodeConfig()
    ph = D.to_physical(out, b, dtype)
    free = D.causal_freedom(b)[0].to(dev)  # the deployed rule (inputs only)
    if dcfg.project_momentum:
        ph = D.project(ph, b, free)
    nc = t["chunk_node"].shape[0]
    sc = t["scales"].to(dev, dtype)
    e_in = sc[:, SCALE_INDEX["E_in"]]
    dt = sc[:, SCALE_INDEX["dt"]]
    call = ph["call"]
    m = ph["mass"]
    i0 = ph["inertia_w"]
    yv = torch.cat([t["y_chunk_v_end"], t["y_imp_v_end"]]).to(dtype)
    yw = torch.cat([t["y_chunk_w_end"], t["y_imp_w_end"]]).to(dtype)
    ydx = torch.cat([t["y_chunk_dx"], t["y_imp_dx"]]).to(dtype)
    ydr = torch.cat([t["y_chunk_drot"], t["y_imp_drot"]]).to(dtype)

    def ke_norm(dv: Tensor, dw: Tensor) -> Tensor:
        return m * (dv * dv).sum(-1) + (dw.unsqueeze(-2) @ i0 @ dw.unsqueeze(-1)).reshape(-1)

    node_err = ke_norm(ph["v_end"] - yv, ph["w_end"] - yw)
    motion = scatter_sum(node_err, call, G) / e_in
    # per fragment under the target partition (stays differentiable)
    frag = t["y_chunk_fragment"]
    nf = b.n_fragments  # counted on the host at collation (no device sync)
    mc = m[:nc].unsqueeze(-1)
    mf = scatter_sum(mc, frag, nf).clamp(min=1e-12)
    vf_hat = scatter_sum(mc * ph["v_end"][:nc], frag, nf) / mf
    vf = scatter_sum(mc * yv[:nc], frag, nf) / mf
    fcall = torch.zeros(nf, dtype=torch.long, device=dev).scatter(0, frag, t["chunk_call"])
    frag_err = mf.squeeze(-1) * ((vf_hat - vf) ** 2).sum(-1)
    motion = motion + scatter_sum(frag_err, fcall, G) / e_in
    cv = t["call_valid"].to(dev, dtype)
    vmean = lambda x: (x * cv).sum() / cv.sum().clamp(min=1)  # noqa: E731 (padding calls out)
    terms["motion"] = vmean(motion)
    # def_motion (/4): the frame-end hidden rates about each target body's momentum-exact
    # rigid fit (P0's definition), in the same KE norm. The hidden part is a small share of
    # the chunk motion, so `motion` alone barely sees it. Both fits use the target
    # frame-end positions, so only the velocities differ. The rates come from the f32
    # world arrays (5e-6 m/s round-off at 50 m/s); the exact `y_chunk_def_end` is kept
    # for a displacement term.
    x1 = (t["chunk_pos"] + t["y_chunk_dx"]).to(dev, dtype)
    ic = i0[:nc]
    dv_hat, dw_hat = body_relative_rates(ph["v_end"][:nc], ph["w_end"][:nc], x1, m[:nc], ic,
                                         frag, nf)
    dv_y, dw_y = body_relative_rates(yv[:nc], yw[:nc], x1, m[:nc], ic, frag, nf)
    def_err = (m[:nc] * ((dv_hat - dv_y) ** 2).sum(-1)
               + ((dw_hat - dw_y).unsqueeze(-2) @ ic @ (dw_hat - dw_y).unsqueeze(-1)).reshape(-1))
    terms["def_motion"] = vmean(scatter_sum(def_err, t["chunk_call"], G) / e_in)
    pose_err = ke_norm(ph["dx"] - ydx, ph["drot"] - ydr)
    terms["pose"] = vmean(scatter_sum(pose_err, call, G) / (e_in * dt * dt))

    # `/5` full state: every carried bond field in its encoding (Huber, live bonds, consequence
    # weights), the failure mode (cross-entropy) and the chunks' support reactions
    if "bond_fs" in out:
        bvd = t["bond_valid"].to(dev, dtype)
        e = _huber(out["bond_fs"].to(dtype) - t["y_bond_fs"].to(dev, dtype), cfg.huber_delta)
        den = (w_b * bvd).sum().clamp(min=1)
        for i, name in enumerate(FS_BOND_TERMS):
            terms[name] = (w_b * bvd * e[:, i]).sum() / den
        ce = F.cross_entropy(out["bond_fs_mode_logits"].float(), t["y_bond_fs_mode"].to(dev),
                             reduction="none")
        terms["fs_mode"] = (ce * bvd).sum() / bvd.sum().clamp(min=1)
        cvd = t["chunk_valid"].to(dev, dtype) if "chunk_valid" in t else torch.ones_like(out["chunk_fs"][:, 0])
        ec = _huber(out["chunk_fs"].to(dtype) - t["y_chunk_fs"].to(dev, dtype), cfg.huber_delta).mean(-1)
        terms["fs_chunk"] = (ec * cvd).sum() / cvd.sum().clamp(min=1)
    if "bond_break_frac" in out and "y_bond_break_frac" in t:
        # exact break time (fraction of the frame) on broken bonds where it was recorded
        yb = t["y_bond_break_frac"].to(dev, dtype)
        m_ = (broke & torch.isfinite(yb) & t["bond_valid"].to(dev)).to(dtype)
        e_ = _huber(out["bond_break_frac"].to(dtype) - torch.nan_to_num(yb), cfg.huber_delta)
        terms["break_t"] = (e_ * m_).sum() / m_.sum().clamp(min=1)
    if "chunk_hidden" in out and "y_chunk_hid" in t:  # frame-end hidden state (u, θ, v, ω)
        cvh = t["chunk_valid"].to(dev, dtype) if "chunk_valid" in t else torch.ones_like(out["chunk_hidden"][:, 0])
        eh = _huber(out["chunk_hidden"].to(dtype) - t["y_chunk_hid"].to(dev, dtype), cfg.huber_delta)
        for j, name in enumerate(HIDDEN_TERMS):
            terms[name] = (eh[:, 3 * j:3 * j + 3].mean(-1) * cvh).sum() / cvh.sum().clamp(min=1)
    if "clu_out" in out:  # frame-end bodies, each group in its encoding
        cv_ = t["clu_valid"].to(dev, dtype) if "clu_valid" in t else torch.ones(out["clu_out"].shape[0], device=dev, dtype=dtype)
        co = out["clu_out"].to(dtype)
        yc = t["y_clu"].to(dev, dtype)
        den = cv_.sum().clamp(min=1)
        for name, sl in (("fs_cl_pos", slice(0, 3)), ("fs_cl_rot", slice(3, 6)), ("fs_cl_vel", slice(6, 9)),
                         ("fs_cl_ang", slice(9, 12)), ("fs_cl_timer", slice(12, 13))):
            terms[name] = (_huber(co[:, sl] - yc[:, sl], cfg.huber_delta).mean(-1) * cv_).sum() / den
        ce_c = F.cross_entropy(co[:, 13:15].float(), t["y_clu_act"].to(dev), reduction="none")
        terms["fs_cl_act"] = (ce_c * cv_).sum() / den

    # impactor: (Δv̂ − Δv)² / v0², plus the ensemble energy score when members exist
    if t["imp_x"].shape[0]:
        v0 = sc[t["imp_call"], SCALE_INDEX["v"]].unsqueeze(-1)
        iv = ph["v_end"][nc:]
        imp = (((iv - yv[nc:]) / v0) ** 2).sum(-1)
        mask = t["ens_imp_mask"].to(dev)
        if mask.shape[1]:  # (a shape, not a value: no sync)
            ens = t["ens_imp_v_end"].to(dev, dtype)
            d1 = torch.linalg.norm(iv.unsqueeze(1) - ens, dim=-1) / v0
            mk = mask.to(dtype)
            nm = mk.sum(1).clamp(min=1)
            spread = (torch.linalg.norm(ens.unsqueeze(1) - ens.unsqueeze(2), dim=-1)
                      * mk.unsqueeze(1) * mk.unsqueeze(2)).sum((1, 2)) / (nm * nm) / v0.squeeze(-1)
            es = (d1 * mk).sum(1) / nm - 0.5 * spread
            imp = imp + torch.where(mask.any(1), es, torch.zeros_like(es))
        iv_ok = t["imp_valid"].to(dev, imp.dtype)
        terms["impactor"] = (imp * iv_ok).sum() / iv_ok.sum().clamp(min=1)
        # regime: log-loss of P(perforate) vs the ensemble frequency or geometric label
        reg = t["y_imp_regime"]
        lab = (reg == S.REGIMES["perforate"]).float()
        tgt = torch.where(t["y_imp_has_freq"], t["y_imp_perf_freq"].float(), lab)
        valid = ((reg != S.REGIME_UNDECIDED) | t["y_imp_has_freq"]) & t["imp_valid"].to(dev)
        bce = F.binary_cross_entropy_with_logits(out["imp_perf_logit"], tgt, reduction="none")
        terms["regime"] = (bce * valid).sum() / valid.sum().clamp(min=1)
    else:
        terms["impactor"] = terms["regime"] = out["imp_perf_logit"].sum() * 0.0

    # energy: heads against the exact KE change and per-bond dissipation, ÷ E_in; plus
    # consistency with the decoded velocities and a balance-violation penalty
    en = t["energy"].to(dev, dtype)
    ke_in, ke_out, w_g, w_x, e_el, d_bond = en[:, 0], en[:, 1], en[:, 2], en[:, 3], en[:, 4], en[:, 5]
    ke_loss = ke_in + w_g + w_x - ke_out
    hd = cfg.huber_delta * 0.1
    e = (_huber(out["energy_d_bond"].to(dtype) - d_bond / e_in, hd)
         + _huber(out["energy_ke_loss"].to(dtype) - ke_loss / e_in, hd)
         + _huber(out["energy_e_elastic"].to(dtype) - e_el / e_in, hd))
    split = t["energy_has_split"].to(dev)
    frac = (en[:, 6] / d_bond.clamp(min=1e-12)).clamp(0, 1)
    e = e + torch.where(split & (d_bond > 0),
                        _huber(out["energy_frac_joint"].to(dtype) - frac, hd), 0.0)
    rd = D.rotvec_to_matrix(ph["drot"])
    i1 = rd @ i0 @ rd.transpose(-1, -2)
    ke_hat = D.kinetic(ph, ph["v_end"], ph["w_end"], i1, G)
    w_g_hat = scatter_sum((m.unsqueeze(-1) * t["gravity"].to(dev, dtype)[call] * ph["dx"]).sum(-1),
                          call, G)
    implied = (ke_in + w_g_hat + w_x - ke_hat) / e_in
    e = e + _huber(out["energy_ke_loss"].to(dtype) - implied, hd)
    bound = ke_in + w_g_hat + w_x + cfg.energy_tolerance * e_in
    e = e + torch.relu(ke_hat - bound) / e_in
    terms["energy"] = vmean(e) / (hd ** 2)
    return {k: v.float() for k, v in terms.items()}


def total(terms: dict[str, Tensor], norms: dict[str, float], cfg: LossConfig
          ) -> tuple[Tensor, dict[str, Tensor]]:
    normed = {k: terms[k] / max(norms.get(k, 1.0), 1e-12) for k in terms}
    loss = sum(cfg.weights.get(k, 1.0) * v for k, v in normed.items())
    return loss, normed


class UncertaintyWeights(torch.nn.Module):
    """`L = Σ λ_k (½ e^{−s_k} L_k + ½ s_k)` with learned s_k = log σ_k² (init 0, so it starts
    exactly at the fixed weighting)."""

    def __init__(self, terms=TERMS):
        super().__init__()
        self.terms = tuple(terms)
        self.log_var = torch.nn.Parameter(torch.zeros(len(self.terms)))

    def forward(self, normed: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
        out = 0.0
        for i, k in enumerate(self.terms):
            if k in normed:
                s = self.log_var[i]
                out = out + weights.get(k, 1.0) * (0.5 * torch.exp(-s) * normed[k] + 0.5 * s)
        return out

    def effective(self, weights: dict[str, float]) -> dict[str, float]:
        return {k: weights.get(k, 1.0) * float(0.5 * torch.exp(-self.log_var[i]))
                for i, k in enumerate(self.terms)}


class DominanceMonitor:
    """Flags a term whose gradient share stays above `share` (the V34 failure: one field was
    >99% of the loss)."""

    def __init__(self, share: float, patience: int):
        self.share, self.patience = share, patience
        self.streak: dict[str, int] = {}

    def update(self, gnorms: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        tot = sum(gnorms.values()) or 1.0
        shares = {k: v / tot for k, v in gnorms.items()}
        flagged = []
        for k, sh in shares.items():
            self.streak[k] = self.streak.get(k, 0) + 1 if sh > self.share else 0
            if self.streak[k] >= self.patience:
                flagged.append(k)
        return shares, flagged


def body_relative_rates(v: Tensor, w: Tensor, x: Tensor, m: Tensor, inertia: Tensor,
                        body: Tensor, n_bodies: int) -> tuple[Tensor, Tensor]:
    """Each chunk's velocity relative to its body's momentum-exact rigid motion (world
    frame): v_body = P/M, ω_body = I_w⁻¹ Σ (r × m v + I_c w) about the body COM, with
    I_w = Σ I_c + m(|r|² 1 − r rᵀ). Returns v − v_body − ω_body × r and w − ω_body: zero net
    linear and angular momentum per body by construction."""
    mc = m.unsqueeze(-1)
    M = scatter_sum(mc, body, n_bodies).clamp(min=1e-12)
    X = scatter_sum(mc * x, body, n_bodies) / M
    r = x - X[body]
    vb = scatter_sum(mc * v, body, n_bodies) / M
    L = scatter_sum(torch.cross(r, mc * v, dim=-1) + (inertia @ w.unsqueeze(-1)).squeeze(-1),
                    body, n_bodies)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    rr = (r * r).sum(-1)[:, None, None] * eye - r.unsqueeze(-1) * r.unsqueeze(-2)
    Iw = scatter_sum((inertia + m[:, None, None] * rr).reshape(-1, 9), body, n_bodies).reshape(-1, 3, 3)
    Iw = Iw + 1e-12 * eye
    # 3×3 inverse by the adjugate (batched, works on MPS; Iw is symmetric positive definite)
    a, b_, c = Iw[:, 0], Iw[:, 1], Iw[:, 2]
    adj = torch.stack([torch.cross(b_, c, dim=-1), torch.cross(c, a, dim=-1),
                       torch.cross(a, b_, dim=-1)], -1)
    det = (a * torch.cross(b_, c, dim=-1)).sum(-1)
    wb = (adj @ L.unsqueeze(-1)).squeeze(-1) / det.unsqueeze(-1)
    return v - vb[body] - torch.cross(wb[body], r, dim=-1), w - wb[body]


@torch.no_grad()
def zero_baselines(batches, cfg: LossConfig, priors=None) -> dict[str, float]:
    sums: dict[str, float] = {}
    n = 0
    for b in batches:
        z = zero_outputs(b)
        for k, v in loss_terms(z, b, cfg, priors).items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    # full-state fields (encodings of order 1) that barely change in these batches would get
    # a near-zero persistence baseline and dominate the gradient: floor it
    return {k: max(v / max(n, 1), 1e-4) if (k.startswith("fs_") or k.startswith("hid_")) else ((v / max(n, 1)) if v > 0 else 1.0)
            for k, v in sums.items()}
