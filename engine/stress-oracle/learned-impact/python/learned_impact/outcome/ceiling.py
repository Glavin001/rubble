"""Output-side ceiling: can the heads + decode represent the oracle's own answer?

No training. The oracle's outcome of each call is converted into the network's raw output
form (`targets_to_outputs`: the exact inverse of `decode.to_physical`, body frames and
nondimensional scales included), then pushed through the deployed decode (momentum
projection, energy bound, break threshold τ, damage clamp) and scored with the real
metrics. Anything short of "essentially perfect" is a limit of the design that no training
can remove.

It attributes the loss to each decode stage:
- `encode`: the round trip through the head encoding alone (no projection, no bound);
- `projection`: what the exact momentum projection changes. It is non-zero when the oracle's
  island does not conserve momentum as the decode assumes (outside contacts, supports);
- `bound`: what the KE bound changes, at the configured tolerance and at a sweep of
  tolerances (P0: ledger gains up to 4.7% E0 under the penalty contact);
- `threshold`: breaks the τ test gets wrong when given the oracle's own peak demand
  (the recorded peak utilisation of broken bonds below τ, or of survivors at or above it).

Bin precision (8 bins) and the full event are tested in the closed loop: the provider's
`ceiling` mode answers each call through this decode (`provider.py`).

    python -m learned_impact.outcome.ceiling --dataset <calls> --split val --out ceiling.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import decode as D
from . import metrics as M
from . import schema as S
from .batch import SCALE_INDEX, Batch, HierarchyCache, collate
from .geometry import quat_rotate_inv, world_inertia

BIG = 30.0  # logit for a certain event


def targets_to_outputs(b: Batch, n_time_bins: int = S.N_TIME_BINS) -> dict[str, torch.Tensor]:
    """The raw head outputs that, decoded, give the oracle's outcome exactly (before
    projection and bound). Float64."""
    t = b.t
    f = torch.float64
    nd = D.nodes(b, f)
    sc = t["scales"].to(f)
    call = nd["call"]
    v_ref = sc[call, SCALE_INDEX["v"]].unsqueeze(-1)
    L = sc[call, SCALE_INDEX["L"]].unsqueeze(-1)
    dt = sc[call, SCALE_INDEX["dt"]].unsqueeze(-1)
    g = t["gravity"].to(f)[call]
    q = nd["q"]
    yv = torch.cat([t["y_chunk_v_end"], t["y_imp_v_end"]]).to(f)
    yw = torch.cat([t["y_chunk_w_end"], t["y_imp_w_end"]]).to(f)
    ydx = torch.cat([t["y_chunk_dx"], t["y_imp_dx"]]).to(f)
    ydr = torch.cat([t["y_chunk_drot"], t["y_imp_drot"]]).to(f)
    inv = lambda x: quat_rotate_inv(q, x)  # noqa: E731
    o = torch.cat([
        inv(yv - nd["v"]) / v_ref,
        inv(yw - nd["w"]) * L / v_ref,
        inv(ydx - nd["v"] * dt - 0.5 * g * dt * dt) / (v_ref * dt),
        inv(ydr - nd["w"] * dt) * L / (v_ref * dt),
    ], -1)
    nc = t["chunk_node"].shape[0]
    broke = t["y_bond_broke"] > 0
    bins = (t["y_bond_break_bin"] - 1).clamp(min=0)
    time_logits = torch.full((broke.shape[0], n_time_bins), -BIG, dtype=f)
    time_logits[torch.arange(broke.shape[0]), bins] = BIG
    en = t["energy"].to(f)
    e_in = sc[:, SCALE_INDEX["E_in"]]
    reg = t["y_imp_regime"]
    return {
        "bond_log_r": torch.log(t["y_bond_peak_util"].to(f).clamp(min=1e-12)),
        "bond_break_logit": torch.where(broke, BIG, -BIG).to(f),
        "bond_time_logits": time_logits,
        "bond_ddamage": t["y_bond_ddamage"].to(f),
        "chunk_out": o[:nc], "imp_out": o[nc:],
        "imp_perf_logit": torch.where(reg == S.REGIMES["perforate"], BIG, -BIG).to(f),
        "energy_d_bond": en[:, 5] / e_in,
        "energy_frac_joint": (en[:, 6] / en[:, 5].clamp(min=1e-12)).clamp(0, 1),
        "energy_e_elastic": en[:, 4] / e_in,
        "energy_ke_loss": (en[:, 0] + en[:, 2] + en[:, 3] - en[:, 1]) / e_in,
    }


def _ke_err(ph, v, w, yv, yw, G):
    """Kinetic-energy-norm error per call (the L_motion metric), unnormalised."""
    i0 = world_inertia(ph["q"], ph["inertia"])
    dv, dw = v - yv, w - yw
    e = ph["mass"] * (dv * dv).sum(-1) + (dw.unsqueeze(-2) @ i0 @ dw.unsqueeze(-1)).reshape(-1)
    return D.scatter_sum(e, ph["call"], G)


def stage_errors(b: Batch, tolerances=(0.0, 0.02, 0.05, 0.1)) -> list[dict]:
    """Per call: error after each decode stage, relative to the oracle's outcome."""
    t = b.t
    G = b.n_calls
    out = targets_to_outputs(b)
    e_in = t["scales"].to(torch.float64)[:, SCALE_INDEX["E_in"]]
    v_ref = t["scales"].to(torch.float64)[:, SCALE_INDEX["v"]]
    yv = torch.cat([t["y_chunk_v_end"], t["y_imp_v_end"]]).double()
    yw = torch.cat([t["y_chunk_w_end"], t["y_imp_w_end"]]).double()
    ydx = torch.cat([t["y_chunk_dx"], t["y_imp_dx"]]).double()
    nc = t["chunk_node"].shape[0]

    def run(**kw):
        cfg = D.DecodeConfig(**kw)
        return D.decode(out, b, cfg)

    enc = run(project_momentum=False, energy_bound=False)
    ph_e = enc["_phys"]
    proj = run(energy_bound=False)
    ph_p = proj["_phys"]
    rows = [dict() for _ in range(G)]
    call = ph_e["call"]

    def per_call_max(x):
        return torch.zeros(G, dtype=x.dtype).scatter_reduce(0, call, x, "amax")

    enc_ke = _ke_err(ph_e, ph_e["v_end"], ph_e["w_end"], yv, yw, G) / e_in
    enc_dv = per_call_max((ph_e["v_end"] - yv).norm(dim=-1)) / v_ref
    enc_dx = per_call_max((ph_e["dx"] - ydx).norm(dim=-1))
    proj_ke = _ke_err(ph_p, ph_p["v_end"], ph_p["w_end"], yv, yw, G) / e_in
    proj_dv = per_call_max((ph_p["v_end"] - yv).norm(dim=-1)) / v_ref
    imp_dv = (ph_p["v_end"][nc:] - yv[nc:]).norm(dim=-1)
    imp_err = torch.zeros(G, dtype=torch.float64).scatter_reduce(
        0, t["imp_call"], imp_dv, "amax") / v_ref
    # the oracle's own momentum residual on islands the decode treats as free
    i0 = world_inertia(ph_e["q"], ph_e["inertia"])
    pos0 = D._com_rel(ph_e, ph_e["pos"], G)
    p0, _ = D.momentum(ph_e, ph_e["v"], ph_e["w"], pos0, i0, G)
    p1, _ = D.momentum(ph_e, yv, yw, pos0, i0, G)
    mt = D.scatter_sum(ph_e["mass"].unsqueeze(-1), call, G)
    dt = t["scales"].double()[:, SCALE_INDEX["dt"]].unsqueeze(-1)
    p_res = (p1 - p0 - mt * t["gravity"].double() * dt).norm(dim=-1)
    p_scale = D.scatter_sum((ph_e["mass"].unsqueeze(-1) * ph_e["v"]).norm(dim=-1), call, G)
    bound = {}
    for tol in tolerances:
        d = run(energy_tolerance=tol)
        ph = d["_phys"]
        bound[tol] = (d["energy_bound_active"],
                      _ke_err(ph, ph["v_end"], ph["w_end"], yv, yw, G) / e_in)
    # threshold with the oracle's own peak demand
    broke_t = t["y_bond_broke"] > 0
    broke_hat = D.breaks(out, b, D.DecodeConfig())
    wrong = broke_t != broke_hat
    bc = t["bond_call"]
    n_wrong = torch.zeros(G, dtype=torch.long).scatter_add(0, bc, wrong.long())
    n_broke = torch.zeros(G, dtype=torch.long).scatter_add(0, bc, broke_t.long())
    for g in range(G):
        c = b.calls[g]
        rows[g] = {
            "event": c.event_id, "call": c.call_index, "family": c.family, "phase": c.phase,
            "free": bool(proj["free"][g]), "anchored": c.anchored,
            "encode_ke_err": float(enc_ke[g]), "encode_max_dv": float(enc_dv[g]),
            "encode_max_dx_m": float(enc_dx[g]),
            "projection_ke_err": float(proj_ke[g]), "projection_max_dv": float(proj_dv[g]),
            "oracle_momentum_residual": float(p_res[g] / p_scale[g].clamp(min=1e-12)),
            "impactor_err_after_projection": float(imp_err[g]),
            "bound": {str(tol): {"active": bool(a[g]), "ke_err": float(k[g])}
                      for tol, (a, k) in bound.items()},
            "threshold_wrong": int(n_wrong[g]), "breaks": int(n_broke[g]),
        }
    return rows


def run_ceiling(batches: list[Batch], tolerance: float = 0.05) -> dict:
    """Stage errors, plus the real skill metrics of the perfect model through the decode."""
    rows, mrows = [], []
    bond_p, bond_y = [], []
    for b in batches:
        rows.extend(stage_errors(b))
        out = targets_to_outputs(b)
        dec = D.decode(out, b, D.DecodeConfig(energy_tolerance=tolerance))
        mrows.extend(M._per_call_outcomes(b, dec))
        bond_p.append(dec["bond_broke"].numpy().astype(float))
        bond_y.append((b.t["y_bond_broke"].numpy() > 0).astype(float))
    summary = M._summarise(mrows, bond_p, bond_y, [np.zeros(0)], [np.zeros(0)], 0, 1)
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    fam = {}
    for f, rs in by_fam.items():
        def q(key, sub=None):
            v = [r[key] if sub is None else r[key][sub]["ke_err"] for r in rs]
            return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                    "max": float(np.max(v))}
        fam[f] = {
            "calls": len(rs),
            "encode_ke_err": q("encode_ke_err"), "encode_max_dv": q("encode_max_dv"),
            "projection_ke_err": q("projection_ke_err"),
            "oracle_momentum_residual_free": {
                "median": float(np.median([r["oracle_momentum_residual"] for r in rs
                                           if r["free"]] or [0])),
                "max": float(np.max([r["oracle_momentum_residual"] for r in rs
                                     if r["free"]] or [0]))},
            "impactor_err_after_projection": q("impactor_err_after_projection"),
            "bound_active_rate": {tol: float(np.mean([r["bound"][tol]["active"] for r in rs]))
                                  for tol in rs[0]["bound"]},
            "bound_ke_err": {tol: q("bound", tol) for tol in rs[0]["bound"]},
            "threshold_wrong_bonds": int(sum(r["threshold_wrong"] for r in rs)),
            "breaks": int(sum(r["breaks"] for r in rs)),
        }
    return {"families": fam, "metrics_through_decode": summary.metrics,
            "per_family_metrics": summary.per_family, "calls": rows}


def verdict(rep: dict, ke_tol: float = 1e-3, dv_tol: float = 1e-3) -> list[str]:
    """Problems that would cap any trained model (empty list = the output side can represent
    the truth on this data)."""
    probs = []
    for f, d in rep["families"].items():
        if d["encode_ke_err"]["max"] > ke_tol or d["encode_max_dv"]["max"] > dv_tol:
            probs.append(f"{f}: the head encoding does not round-trip "
                         f"(max KE err {d['encode_ke_err']['max']:.2e}, "
                         f"max |Δv|/v {d['encode_max_dv']['max']:.2e})")
        if d["projection_ke_err"]["p95"] > ke_tol:
            probs.append(f"{f}: the momentum projection moves the oracle's outcome "
                         f"(p95 KE err {d['projection_ke_err']['p95']:.2e}; oracle momentum "
                         f"residual on 'free' islands up to "
                         f"{d['oracle_momentum_residual_free']['max']:.2e})")
        for tol, rate in d["bound_active_rate"].items():
            if float(tol) == 0.05 and rate > 0:
                probs.append(f"{f}: the KE bound at tolerance 0.05 clips {rate:.1%} of "
                             f"oracle outcomes (KE err max {d['bound_ke_err'][tol]['max']:.2e})")
        if d["threshold_wrong_bonds"]:
            probs.append(f"{f}: τ = 1 on the oracle's own peak demand misclassifies "
                         f"{d['threshold_wrong_bonds']} of the bonds ({d['breaks']} breaks)")
    m = rep["metrics_through_decode"]
    for k, bar in (("one_minus_ari", 1e-6), ("w1_log_mass", 1e-6), ("ram_err_rel", 1e-3)):
        if m.get(k, 0.0) > bar:
            probs.append(f"metrics through the decode: {k} = {m[k]:.3g} (should be ≈ 0)")
    return probs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=0.05)
    a = ap.parse_args(argv)
    events = [S.load_event(p) for p in S.find_events(a.dataset)]
    events = [e for e in events if a.split == "all" or e.manifest["split"] == a.split]
    calls = [c for e in events for c in e.calls if c.routed]
    st = {e.manifest["event_id"]: e.structures for e in events}
    cache = HierarchyCache(2)
    batches = [collate(calls[i:i + 32], st, cache) for i in range(0, len(calls), 32)]
    rep = run_ceiling(batches, a.tolerance)
    rep["problems"] = verdict(rep)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for p in rep["problems"] or ["none: the output side represents the oracle's outcome"]:
        print(f"[ceiling] {p}")


if __name__ == "__main__":
    main()


def call_with_decoded_targets(c: S.Call, doc: dict) -> S.Call:
    """A call whose targets are a decoded-state answer (the harness's `oracle.json`), for the
    closed-loop ceiling. The oracle's peak demand is not in a decoded state, so broken
    bonds get r = e (log r = 1) and survivors r = 1/e: the threshold itself is tested offline
    (`stage_errors`), the rest of the output side here."""
    a = dict(c.arrays)
    nc, ni = c.counts["chunks"], c.counts["impactors"]
    by_c = {n["key"]: n for n in doc["chunks"]}
    by_i = {n["key"]: n for n in doc["impactors"]}
    ck, ik = a["chunk_key"].tolist(), a["impactor_key"].tolist()
    for pre, keys, src in (("y_chunk", ck, by_c), ("y_impactor", ik, by_i)):
        for field, name in (("v", "v_end"), ("w", "w_end"), ("dx", "dx"), ("drot", "drot")):
            a[f"{pre}_{name}"] = np.array([src[k][field] for k in keys], dtype=np.float64
                                          ).reshape(-1, 3).astype(np.float32)
    broken = {x["key"]: x["bin"] for x in doc["bonds_broken"]}
    dmg = {x["key"]: x["damage"] for x in doc["bond_damage"]}
    bk = a["bond_key"].tolist()
    d0 = a["bond_feat"][0::2, S.col("bond_feat", "damage")]
    a["y_bond_broke"] = np.array([k in broken for k in bk], np.int32)
    a["y_bond_break_bin"] = np.array([broken.get(k, 0) for k in bk], np.int32)
    a["y_bond_peak_util"] = np.where(a["y_bond_broke"] > 0, np.e, 1 / np.e).astype(np.float32)
    a["y_bond_ddamage"] = np.array([max(dmg.get(k, d) - d, 0.0) for k, d in zip(bk, d0)],
                                   np.float32)
    a["y_impactor_regime"] = np.array([1 if by_i[k].get("p_perforate", 0) > 0.5 else 0
                                       for k in ik], np.int32)
    out = S.with_dummy_targets(S.Call(**{**c.__dict__, "arrays": a}))
    assert out.counts["chunks"] == nc and out.counts["impactors"] == ni
    return out


def decode_ceiling(c: S.Call, structures, doc: dict, cfg: D.DecodeConfig | None = None,
                   cache: HierarchyCache | None = None) -> dict:
    """The oracle's answer pushed through heads + decode, as a decoded state."""
    from . import state_io as IO

    ct = call_with_decoded_targets(c, doc)
    b = collate([ct], {c.event_id: structures}, cache or HierarchyCache(2))
    out = targets_to_outputs(b)
    dec = D.decode(out, b, cfg or D.DecodeConfig())
    return IO.decoded_states(b, out, dec, {"model": "decode-ceiling"})[0]
