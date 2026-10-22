"""Deterministic decode (plan, "Decode"): network outputs → physical frame outcome.

1. Node outputs (body frame, nondimensional) → world Δv, Δω, Δx, Δθ.
2. Free islands: the centre-of-mass displacement is set exactly (Σ m Δx = P₀·dt + ½M g dt²),
   then the closed-form minimum-kinetic-energy-norm projection of all velocities onto the
   input linear and angular momentum (P₀ + M g dt, L₀), evaluated at the frame-end poses.
   Anchored islands are not projected (the support reaction is unknown).
3. Energy bound: KE_out ≤ KE_in + W_gravity + W_external + tol·E_in. On a free island only
   the non-rigid part of the motion is scaled, so P and L stay exact; on an anchored island
   every free node's velocity is scaled. Activations and infeasible cases are counted.
   P0 measured the ledger residual at +0.3–4.7% E0 under the penalty contact, so the default
   tolerance is 0.05 (a config, not a constant).
4. Breaks: `log r̂ ≥ log τ` (τ = 1), or the hash-seeded sampling ablation on `p_b`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..mgn.mesh_blocks import scatter_sum
from . import schema as S
from .batch import SCALE_INDEX, Batch
from .geometry import quat_rotate, quat_to_matrix, rotvec_to_matrix, sym6_to_matrix, world_inertia


# Admissibility (planning, 2026-10-22, P0's energy envelope): no node moves further in a
# frame than its island's frame-start mechanical energy allows. With E = (KE_in + the joint
# law's stored energy E_elastic_start)·(1 + tol), |dx| ≤ dt·(√(2E/m) + |g|·dt) and
# |drot| ≤ dt·√(2E/λ_min(I_w)). The oracle cannot exceed it (no body's kinetic energy can
# pass the island's); a model's buried-ram teleport does. The velocity trapezoid
# dt·(v_in + v_end)/2 ± dt·|Δv|/2 is only a soft loss term (losses: "kinematic"): the oracle's
# velocities are not monotone within a frame under contact (0.85% of chunk displacements
# and 1.7% of rotations lie > 1% of the scale outside it, P1's calls-v5).
# The tolerance: P0's +5% held on its three events, but on all of P1's calls-v5 (40,713 calls,
# 4.4M node-frames) two impactor frames exceed it by 1.46× in radius: a ram struck by
# another ram outside the call's island (p1_anchored_wall_t2_00091_s21202 e5/e7, frame 1:
# KE 79 → 318 J with 46 J of gravity work), energy the island's inputs cannot see; they
# need E·(1 + tol) ≥ 2.04 and 2.78 × the island's (solved exactly, gravity term included).
# +200% (E × 3) covers them; rotations: none outside.
ENERGY_ENVELOPE_TOL = 2.0


@dataclass
class DecodeConfig:
    tau: float = 1.0
    # "threshold": the failure margin q̂ ≥ τ (q = κ / κ_fail, ≥ 1 exactly when the oracle's
    # bond fails); "utilization": peak utilisation r̂ ≥ τ (plan v1; the output-side ceiling
    # on P1's smoke calls showed it is the onset of softening, not failure); "sample":
    # hash-seeded sampling on p_b.
    break_mode: str = "threshold"
    sample_seed: int = 0
    project_momentum: bool = True
    energy_bound: bool = True
    energy_tolerance: float = 0.05  # fraction of E_in (P0: ledger residual up to 4.7%)
    # the energy envelope's tolerance (fraction of E; None: off), see ENERGY_ENVELOPE_TOL
    energy_envelope: float | None = ENERGY_ENVELOPE_TOL
    temperature: float = 1.0  # validation-fitted temperature for p_b
    dtype: torch.dtype = torch.float64


def _skew(r: Tensor) -> Tensor:
    z = torch.zeros_like(r[..., 0])
    return torch.stack([z, -r[..., 2], r[..., 1], r[..., 2], z, -r[..., 0], -r[..., 1],
                        r[..., 0], z], -1).reshape(r.shape[:-1] + (3, 3))


def nodes(b: Batch, dtype=torch.float64) -> dict[str, Tensor]:
    """Chunks then impactors, concatenated per batch (not interleaved per call)."""
    t = b.t
    cat = lambda a, c: torch.cat([t[a], t[c]], 0).to(dtype)  # noqa: E731
    return {
        "mass": cat("chunk_mass", "imp_mass"),
        "inertia": cat("chunk_inertia_body", "imp_inertia_body"),
        "q": cat("chunk_q_world", "imp_q_world"),
        "pos": cat("chunk_pos", "imp_pos"),
        "v": cat("chunk_v", "imp_v"),
        "w": cat("chunk_w", "imp_w"),
        "inertia_w": node_world_inertia(b, dtype),
        "call": torch.cat([t["chunk_call"], t["imp_call"]], 0),
        "anchored": torch.cat([t["chunk_anchored"],
                               torch.zeros_like(t["imp_call"], dtype=torch.bool)], 0),
    }


def node_world_inertia(b: Batch, dtype=torch.float64) -> Tensor:
    """Frame-start world inertia [N,3,3], chunks then impactors. A chunk's inertia is
    `ChunkData.inertia` in its structure's body frame, rotated by the body (cluster)
    rotation only, as the solver does (R_body I R_bodyᵀ): with `/4`'s chunk_inertia_struct
    and chunk_body_q. Without them, the principal diagonal rotated by the chunk's own
    orientation: exact only for chunks with identity rest rotation (every brick so far)."""
    t = b.t
    imp = world_inertia(t["imp_q_world"].to(dtype), t["imp_inertia_body"].to(dtype))
    if "chunk_inertia_struct" in t and "chunk_body_q" in t:
        r = quat_to_matrix(t["chunk_body_q"].to(dtype))
        ch = r @ sym6_to_matrix(t["chunk_inertia_struct"].to(dtype)) @ r.transpose(-1, -2)
    else:
        ch = world_inertia(t["chunk_q_world"].to(dtype), t["chunk_inertia_body"].to(dtype))
    return torch.cat([ch, imp], 0)


def to_physical(out: dict[str, Tensor], b: Batch, dtype=torch.float64) -> dict[str, Tensor]:
    nd = nodes(b, dtype)
    sc = b.t["scales"].to(dtype)
    call = nd["call"]
    v_ref = sc[call, SCALE_INDEX["v"]].unsqueeze(-1)
    L = sc[call, SCALE_INDEX["L"]].unsqueeze(-1)
    dt = sc[call, SCALE_INDEX["dt"]].unsqueeze(-1)
    g = b.t["gravity"].to(dtype)[call]
    o = torch.cat([out["chunk_out"], out["imp_out"]], 0).to(dtype)
    q = nd["q"]
    rot = lambda x: quat_rotate(q, x)  # noqa: E731
    v_end = nd["v"] + rot(o[:, 0:3]) * v_ref
    w_end = nd["w"] + rot(o[:, 3:6]) * v_ref / L
    dx = nd["v"] * dt + 0.5 * g * dt * dt + rot(o[:, 6:9]) * v_ref * dt
    drot = nd["w"] * dt + rot(o[:, 9:12]) * v_ref * dt / L
    anch = nd["anchored"].unsqueeze(-1)
    zero = torch.zeros_like(v_end)
    return {"v_end": torch.where(anch, zero, v_end), "w_end": torch.where(anch, zero, w_end),
            "dx": torch.where(anch, zero, dx), "drot": torch.where(anch, zero, drot), **nd}


def momentum(nd: dict[str, Tensor], v: Tensor, w: Tensor, pos: Tensor, inertia_w: Tensor,
             n_calls: int) -> tuple[Tensor, Tensor]:
    m = nd["mass"].unsqueeze(-1)
    p = scatter_sum(m * v, nd["call"], n_calls)
    lin = torch.cross(pos, m * v, dim=-1) + (inertia_w @ w.unsqueeze(-1)).squeeze(-1)
    return p, scatter_sum(lin, nd["call"], n_calls)


def _com_rel(nd, pos, n_calls):
    m = nd["mass"].unsqueeze(-1)
    mt = scatter_sum(m, nd["call"], n_calls)
    com = scatter_sum(m * pos, nd["call"], n_calls) / mt
    return pos - com[nd["call"]]


def project(ph: dict[str, Tensor], b: Batch, free: Tensor) -> dict[str, Tensor]:
    """Exact momentum projection for islands with `free[g]` (no anchored node)."""
    G = b.n_calls
    call = ph["call"]
    dtype = ph["v_end"].dtype
    m = ph["mass"].unsqueeze(-1)
    mt = scatter_sum(m, call, G)
    g = b.t["gravity"].to(dtype)
    dt = b.t["scales"].to(dtype)[:, SCALE_INDEX["dt"]].unsqueeze(-1)
    node_free = free[call].unsqueeze(-1)
    inertia0 = ph["inertia_w"]
    pos0 = _com_rel(ph, ph["pos"], G)
    p0, l0 = momentum(ph, ph["v"], ph["w"], pos0, inertia0, G)
    # centre-of-mass displacement, exact
    dx = ph["dx"]
    want = p0 * dt + 0.5 * mt * g * dt * dt
    have = scatter_sum(m * dx, call, G)
    dx = torch.where(node_free, dx + ((want - have) / mt)[call], dx)
    # frame-end geometry
    rd = rotvec_to_matrix(ph["drot"])
    inertia1 = rd @ inertia0 @ rd.transpose(-1, -2)
    pos1 = _com_rel(ph, ph["pos"] + dx, G)
    p_t, l_t = p0 + mt * g * dt, l0
    v, w = ph["v_end"], ph["w_end"]
    # A_i = [[m I, 0], [m [r]×, I_w]] ; M_i⁻¹ = diag(I/m, I_w⁻¹)
    n = v.shape[0]
    eye = torch.eye(3, dtype=dtype, device=v.device).expand(n, 3, 3)
    a = torch.zeros(n, 6, 6, dtype=dtype, device=v.device)
    a[:, :3, :3] = m.unsqueeze(-1) * eye
    a[:, 3:, :3] = m.unsqueeze(-1) * _skew(pos1)
    a[:, 3:, 3:] = inertia1
    minv = torch.zeros_like(a)
    minv[:, :3, :3] = eye / m.unsqueeze(-1)
    minv[:, 3:, 3:] = torch.linalg.inv(inertia1)
    x = torch.cat([v, w], -1)
    ax = scatter_sum((a @ x.unsqueeze(-1)).squeeze(-1), call, G)
    amat = scatter_sum((a @ minv @ a.transpose(-1, -2)).reshape(n, 36), call, G).reshape(G, 6, 6)
    resid = torch.cat([p_t, l_t], -1) - ax
    eye6 = torch.eye(6, dtype=dtype, device=v.device).expand(G, 6, 6)
    amat = torch.where(free.view(G, 1, 1), amat, eye6)
    lam = torch.linalg.solve(amat, torch.where(free.unsqueeze(-1), resid, 0.0))
    dxv = (minv @ a.transpose(-1, -2) @ lam[call].unsqueeze(-1)).squeeze(-1)
    x = torch.where(node_free, x + dxv, x)
    out = dict(ph)
    out.update(v_end=x[:, :3], w_end=x[:, 3:], dx=dx, inertia_end=inertia1, pos_end=pos1)
    return out


def kinetic(ph: dict[str, Tensor], v: Tensor, w: Tensor, inertia_w: Tensor, G: int) -> Tensor:
    m = ph["mass"]
    e = 0.5 * (m * (v * v).sum(-1) + (w.unsqueeze(-2) @ inertia_w @ w.unsqueeze(-1)).reshape(-1))
    return scatter_sum(e, ph["call"], G)


def causal_freedom(b: Batch) -> tuple[Tensor, Tensor]:
    """From the call's *inputs* only (never the oracle's ledger, a target):

    - `free[g]`: the island's momentum is conserved (up to gravity), so the projection is
      exact: no anchored chunk, no pressure/blast load, and no gravity (with gravity, an
      unseen ground or static contact would add an impulse; no input says there is none).
    - `external[g]`: a pressure/blast load does work the inputs cannot bound yet; the KE
      bound is skipped for those calls (counted as `energy_bound_skipped`).
    """
    t = b.t
    G = b.n_calls
    dev = t["chunk_call"].device
    anch = torch.zeros(G, dtype=torch.bool, device=dev).index_put_(
        (t["chunk_call"],), t["chunk_anchored"].to(dev).bool(), accumulate=True)
    # the call's own frame-start flag (also set on padding calls, which hold no free body)
    anch = anch | t["call_anchored"].to(dev).bool()
    gf = t["glob_x"]
    blast_cols = [S.col("global_feat", c) for c in ("blast_log_tnt", "blast_z", "blast_impulse")]
    chunk_blast = [S.col("chunk_feat", f"blast_{i}") for i in range(6)]
    ext = (gf[:, blast_cols] != 0).any(-1)
    cb = (t["chunk_x"][:, chunk_blast] != 0).any(-1)
    ext = ext | torch.zeros(G, dtype=torch.bool, device=dev).index_put_(
        (t["chunk_call"],), cb, accumulate=True)
    grav = gf[:, S.col("global_feat", "gravity_mag")] != 0
    return (~anch) & (~ext) & (~grav), ext


def energy_bound(ph: dict[str, Tensor], b: Batch, free: Tensor, tol: float,
                 skip: Tensor | None = None) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    G = b.n_calls
    call = ph["call"]
    dtype = ph["v_end"].dtype
    m = ph["mass"].unsqueeze(-1)
    inertia0 = ph["inertia_w"]
    inertia1 = ph.get("inertia_end")
    if inertia1 is None:
        rd = rotvec_to_matrix(ph["drot"])
        inertia1 = rd @ inertia0 @ rd.transpose(-1, -2)
    pos1 = ph.get("pos_end")
    if pos1 is None:
        pos1 = _com_rel(ph, ph["pos"] + ph["dx"], G)
    g = b.t["gravity"].to(dtype)
    e_in = b.t["scales"].to(dtype)[:, SCALE_INDEX["E_in"]]
    ke_in = kinetic(ph, ph["v"], ph["w"], inertia0, G)
    w_g = scatter_sum((m * g[call] * ph["dx"]).sum(-1), call, G)
    # inputs only: KE_in and gravity's work from the decoded motion (external work is 0
    # without a pressure load; calls with one are skipped, `skip`)
    bound = ke_in + w_g + tol * e_in
    # The event's cumulative budget where the call has one (planning, 2026-10-22; P0's
    # stress_learned::energy_budget): the island's KE + E_elastic + PE may rise by at most
    # `remaining` (a per-frame +tol compounds over an event). The decode bounds kinetic
    # energy: KE_out ≤ KE_in + E_elastic_start + W_g + remaining, never tighter than the
    # oracle (its frame-end elastic energy is ≥ 0); the host's bound adds the decoded elastic.
    rem = b.t["e_budget_remaining"].to(dtype).to(ke_in.device) if "e_budget_remaining" in b.t else None
    if rem is not None:
        st = b.t["e_stored_start"].to(dtype).to(ke_in.device)
        ok = torch.isfinite(rem) & torch.isfinite(st)
        bound = torch.where(ok, ke_in + torch.where(ok, st, torch.zeros_like(st)) + w_g
                            + torch.where(ok, rem, torch.zeros_like(rem)), bound)
    v, w = ph["v_end"], ph["w_end"]
    ke = kinetic(ph, v, w, inertia1, G)
    active = ke > bound
    if skip is not None:
        active = active & ~skip.to(active.device)
    # free: split into COM translation + rigid rotation Ω (I_tot Ω = L_rel) + residual
    mt = scatter_sum(m, call, G)
    vc = scatter_sum(m * v, call, G) / mt
    vr = v - vc[call]
    eye = torch.eye(3, dtype=dtype, device=v.device)
    r2 = (pos1 * pos1).sum(-1, keepdim=True).unsqueeze(-1)
    i_node = m.unsqueeze(-1) * (r2 * eye - pos1.unsqueeze(-1) @ pos1.unsqueeze(-2)) + inertia1
    i_tot = scatter_sum(i_node.reshape(-1, 9), call, G).reshape(G, 3, 3)
    l_rel = scatter_sum(torch.cross(pos1, m * vr, dim=-1)
                        + (inertia1 @ w.unsqueeze(-1)).squeeze(-1), call, G)
    i_safe = torch.where(free.view(G, 1, 1), i_tot, eye.expand(G, 3, 3))
    omega = torch.linalg.solve(i_safe, l_rel.unsqueeze(-1)).squeeze(-1)
    u = vr - torch.cross(omega[call], pos1, dim=-1)
    wres = w - omega[call]
    ke_base = 0.5 * (mt.squeeze(-1) * (vc * vc).sum(-1) + (omega * l_rel).sum(-1))
    ke_res = kinetic(ph, u, wres, inertia1, G)
    s_free = torch.sqrt(((bound - ke_base) / ke_res.clamp(min=1e-30)).clamp(min=0.0, max=1.0))
    s_anch = torch.sqrt((bound.clamp(min=0) / ke.clamp(min=1e-30)).clamp(max=1.0))
    s = torch.where(free, s_free, s_anch)
    s = torch.where(active, s, torch.ones_like(s))
    infeasible = active & free & (ke_base > bound)
    sn = s[call].unsqueeze(-1)
    fn = free[call].unsqueeze(-1)
    v_new = torch.where(fn, vc[call] + torch.cross(omega[call], pos1, dim=-1) + sn * u, sn * v)
    w_new = torch.where(fn, omega[call] + sn * wres, sn * w)
    out = dict(ph)
    out.update(v_end=v_new, w_end=w_new)
    return out, active, infeasible


_M32 = 0xFFFFFFFF


def hash_uniform(keys: Tensor, seed: int) -> Tensor:
    """Deterministic uniform in [0, 1) per stable key (PCG-style mixing in int64 without
    overflow); identical on every device."""
    k = keys.to(torch.int64)
    x = (k & _M32) ^ (((k >> 32) * 0x85EBCA6B) & _M32) ^ ((seed * 0x9E3779B1) & _M32)
    for _ in range(2):
        x = (x * 747796405 + 2891336453) & _M32
        x = ((((x >> ((x >> 28) + 4)) ^ x) * 277803737) & _M32)
        x = (x >> 22) ^ x
    return x.to(torch.float64) / 4294967296.0


def breaks(out: dict[str, Tensor], b: Batch, cfg: DecodeConfig,
           logit_shift: Tensor | None = None) -> Tensor:
    if cfg.break_mode == "threshold":
        return out["bond_log_q"] >= float(np.log(cfg.tau))
    if cfg.break_mode == "utilization":
        return out["bond_log_r"] >= float(np.log(cfg.tau))
    if cfg.break_mode == "sample":
        logit = out["bond_break_logit"]
        if logit_shift is not None:
            logit = logit + logit_shift
        p = torch.sigmoid(logit / cfg.temperature).to(torch.float64)
        return hash_uniform(b.t["bond_key"].to(p.device), cfg.sample_seed) < p
    raise ValueError(cfg.break_mode)


def fragments(b: Batch, broke: Tensor) -> np.ndarray:
    """Connected components of intact bonds after the predicted breaks (chunk labels)."""
    t = b.t
    n_chunks = t["chunk_node"].shape[0]
    node_to_chunk = -np.ones(int(t["n_nodes"]), dtype=np.int64)
    node_to_chunk[t["chunk_node"].cpu().numpy()] = np.arange(n_chunks)
    src = node_to_chunk[b.edges["bond"]["src"][0::2].cpu().numpy()]
    dst = node_to_chunk[b.edges["bond"]["dst"][0::2].cpu().numpy()]
    keep = ~broke.cpu().numpy()
    parent = np.arange(n_chunks)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, c in zip(src[keep].tolist(), dst[keep].tolist()):
        ra, rc = find(a), find(c)
        if ra != rc:
            parent[max(ra, rc)] = min(ra, rc)
    roots = np.array([find(i) for i in range(n_chunks)])
    return np.unique(roots, return_inverse=True)[1]


def trapezoid_excess(ph: dict[str, Tensor], b: Batch) -> tuple[Tensor, Tensor]:
    """Per node and axis of its frame-start frame, how far dx (drot) lies outside the velocity
    trapezoid's envelope |q⁻¹·(dx − dt·(v_in + v_end)/2)| ≤ dt·|q⁻¹·(v_end − v_in)|/2, in
    dt·v_ref (dt·v_ref/L); 0 inside and for anchored nodes. The soft kinematic term."""
    from .geometry import quat_rotate_inv

    dtype = ph["dx"].dtype
    sc = b.t["scales"].to(dtype)[ph["call"]]
    dt = sc[:, SCALE_INDEX["dt"]].unsqueeze(-1)
    v_ref = sc[:, SCALE_INDEX["v"]].unsqueeze(-1)
    L = sc[:, SCALE_INDEX["L"]].unsqueeze(-1)
    q = ph["q"] / ph["q"].norm(dim=-1, keepdim=True)  # f32-stored: |q| = 1 ± 1e-7
    free = ~ph["anchored"].unsqueeze(-1)
    out = []
    for k, a0, a1, s in (("dx", ph["v"], ph["v_end"], dt * v_ref), ("drot", ph["w"], ph["w_end"], dt * v_ref / L)):
        res = quat_rotate_inv(q, ph[k] - dt * (a0 + a1) / 2)
        h = dt * quat_rotate_inv(q, a1 - a0).abs() / 2
        out.append(torch.where(free, (res.abs() - h).clamp(min=0) / s, torch.zeros_like(res)))
    return out[0], out[1]


def _norm3(x: Tensor) -> Tensor:
    """√(x₀² + x₁² + x₂²) in this order (bit-for-bit with the host's Rust; torch.norm's
    reduction order is not fixed)."""
    return torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1] + x[..., 2] * x[..., 2])


def energy_envelope(ph: dict[str, Tensor], b: Batch, external: Tensor,
                    tol: float) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Each free node's dx and drot scaled radially into the energy envelope (see
    ENERGY_ENVELOPE_TOL). Returns the state, the clamped (node, dx|drot) count per call, and
    the calls left unbounded: an external load does work the inputs do not bound, or the
    call carries no frame-start KE_in / E_elastic_start (never a tighter bound than the
    oracle's). Bit-for-bit with the host (stress_learned::kin_envelope, P0's fixture
    kin_envelope_v2.json): E = (energy.KE_in + energy.E_elastic_start)·(1 + tol), the builder's
    f64 scalars; r_x = dt·(√(2E/m) + |g|·dt); r_r = dt·√(2E/I_min), I_min the smallest
    principal moment (inertia_body: rotation-invariant, no eigensolver)."""
    G = b.n_calls
    call = ph["call"]
    dtype = ph["dx"].dtype
    dev = ph["dx"].device
    dt = b.t["scales"].to(dtype)[:, SCALE_INDEX["dt"]].to(dev)
    ke = b.t["e_ke_in"].to(dtype).to(dev)
    stored = b.t["e_stored_start"].to(dtype).to(dev)
    unbounded = external.to(dev) | ~torch.isfinite(stored) | ~torch.isfinite(ke)
    e = torch.where(unbounded, torch.zeros_like(ke), ke + stored) * (1.0 + tol)
    g = _norm3(b.t["gravity"].to(dtype).to(dev))
    m = ph["mass"]
    i_min = ph["inertia"].to(dtype).min(dim=-1).values
    dtc, ec = dt[call], e[call]
    r_x = dtc * (torch.sqrt(2.0 * ec / m) + g[call] * dtc)
    r_r = dtc * torch.sqrt(2.0 * ec / i_min)
    on = ~ph["anchored"] & ~unbounded[call]
    out, n = dict(ph), torch.zeros(G, dtype=torch.long, device=dev)
    for k, r in (("dx", r_x), ("drot", r_r)):
        norm = _norm3(ph[k])
        hit = on & (norm > r * (1.0 + 1e-9))
        out[k] = torch.where(hit.unsqueeze(-1), ph[k] * (r / torch.where(hit, norm, torch.ones_like(norm))).unsqueeze(-1), ph[k])
        n = n + scatter_sum(hit.to(torch.long), call, G)
    return out, n, unbounded


def envelope_radii(b: Batch, tol: float = None) -> tuple[Tensor, Tensor, Tensor]:
    """(E per call before the tolerance, r_x, r_r per node; NaN where unbounded or anchored)."""
    tol = ENERGY_ENVELOPE_TOL if tol is None else tol
    ph = dict(nodes(b))
    dt = b.t["scales"].double()[:, SCALE_INDEX["dt"]]
    ke, stored = b.t["e_ke_in"].double(), b.t["e_stored_start"].double()
    e = (ke + stored) * (1.0 + tol)
    g = _norm3(b.t["gravity"].double())
    call = ph["call"]
    dtc, ec = dt[call], e[call]
    r_x = dtc * (torch.sqrt(2.0 * ec / ph["mass"]) + g[call] * dtc)
    r_r = dtc * torch.sqrt(2.0 * ec / ph["inertia"].double().min(dim=-1).values)
    off = ph["anchored"] | ~torch.isfinite(ec)
    nan = torch.full_like(r_x, float("nan"))
    # E as the host reports it: (E·(1 + tol))/(1 + tol), not ke + stored (1 ulp apart)
    return e / (1.0 + tol), torch.where(off, nan, r_x), torch.where(off, nan, r_r)


def label_envelope_clamps(b: Batch, tol: float) -> Tensor:
    """The oracle's own frame outcomes (the batch's targets) against the energy envelope:
    (node, dx | drot) pairs per call it would clamp. Recorded data must give 0."""
    t = b.t
    ph = dict(nodes(b))
    ph.update(v_end=torch.cat([t["y_chunk_v_end"], t["y_imp_v_end"]]).double(),
              w_end=torch.cat([t["y_chunk_w_end"], t["y_imp_w_end"]]).double(),
              dx=torch.cat([t["y_chunk_dx"], t["y_imp_dx"]]).double(),
              drot=torch.cat([t["y_chunk_drot"], t["y_imp_drot"]]).double())
    return energy_envelope(ph, b, causal_freedom(b)[1], tol)[1]


def decode(out: dict[str, Tensor], b: Batch, cfg: DecodeConfig | None = None) -> dict:
    cfg = cfg or DecodeConfig()
    ph = to_physical(out, b, cfg.dtype)
    G = b.n_calls
    dev = ph["v_end"].device
    free, external = causal_freedom(b)
    free, external = free.to(dev), external.to(dev)
    if cfg.project_momentum:
        ph = project(ph, b, free)
    clamps = torch.zeros(G, dtype=torch.long, device=dev)
    unbounded = torch.ones(G, dtype=torch.bool, device=dev)
    if cfg.energy_envelope is not None:  # the hard admissibility constraint, after the projection
        ph, clamps, unbounded = energy_envelope(ph, b, external, cfg.energy_envelope)
    active = infeasible = torch.zeros(G, dtype=torch.bool, device=dev)
    if cfg.energy_bound:
        ph, active, infeasible = energy_bound(ph, b, free, cfg.energy_tolerance, external)
    broke = breaks(out, b, cfg)
    d0 = b.edges["bond"]["x"][0::2, S.col("bond_feat", "damage")].to(out["bond_ddamage"].dtype)
    # A survivor's damage stays strictly below 1: only the break decision (q̂ ≥ τ) breaks a
    # bond, and damage 1 would disconnect it without its failure margin (the host enforces
    # the same bound, `SURVIVOR_DAMAGE_MAX`).
    damage_end = torch.clamp(d0 + out["bond_ddamage"], max=1.0 - 1e-6)
    nc = b.t["chunk_node"].shape[0]
    return {
        "chunk_v_end": ph["v_end"][:nc], "chunk_w_end": ph["w_end"][:nc],
        "chunk_dx": ph["dx"][:nc], "chunk_drot": ph["drot"][:nc],
        "imp_v_end": ph["v_end"][nc:], "imp_w_end": ph["w_end"][nc:],
        "imp_dx": ph["dx"][nc:], "imp_drot": ph["drot"][nc:],
        "bond_broke": broke, "bond_damage_end": damage_end,
        "energy_bound_active": active, "energy_bound_infeasible": infeasible,
        "energy_bound_skipped": external,
        # (node, dx | drot) pairs per call scaled into the energy envelope; calls it cannot bound
        "admissibility_clamps": clamps, "admissibility_unbounded": unbounded,
        "free": free, "_phys": ph,
    }
