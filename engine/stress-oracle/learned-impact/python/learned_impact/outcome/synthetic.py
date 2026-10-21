"""Synthetic `learned-impact-outcome/3` events for tests and trainer development.

A toy, *not* the oracle: a free (or bottom-anchored) one-chunk-thick wall struck by one ram.
Bond demand falls off with hops from the struck chunk and scales with the energy ratio;
bonds with demand ≥ 1 break; the ram and the fragment it strikes exchange momentum as a
two-body collision (restitution 0 when the plug breaks free, 0.3 otherwise). Every array
obeys the schema (shapes, twins, frames, ledger sums), and the whole scene is placed under a
random rotation and translation so invariance can be tested. Nothing trained on this says
anything about accuracy on oracle data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import schema as S
import torch

from .geometry import quat_mul_np, quat_to_matrix_np, random_quat, rotvec_to_matrix

G = 9.81


REV = np.array([[-1.0], [1.0], [-1.0]])  # row 2k+1 frame = (−t1, t2, −n)


def _frame_from_normal(n: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2, n])  # rows t1, t2, n


def _rot6(r_body: np.ndarray, r_edge_rows: np.ndarray) -> np.ndarray:
    """First two columns of R_bodyᵀ·R_edge (R_edge columns = t1, t2, n)."""
    m = r_body.T @ r_edge_rows.T
    return np.concatenate([m[:, 0], m[:, 1]])


def _components(n: int, ends: np.ndarray, live: np.ndarray) -> np.ndarray:
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (a, b), ok in zip(ends.tolist(), live.tolist()):
        if ok:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab.astype(np.int32)


def _hops(n: int, ends: np.ndarray, live: np.ndarray, src: list[int]) -> np.ndarray:
    adj: list[list[int]] = [[] for _ in range(n)]
    for (a, b), ok in zip(ends.tolist(), live.tolist()):
        if ok:
            adj[a].append(b)
            adj[b].append(a)
    d = np.full(n, 10**6)
    q = list(src)
    for s in src:
        d[s] = 0
    while q:
        nq = []
        for a in q:
            for b in adj[a]:
                if d[b] > d[a] + 1:
                    d[b] = d[a] + 1
                    nq.append(b)
        q = nq
    return d


def _project_momentum_np(m, inertia_w, pos, v, w, p_target, l_target):
    """Exact minimum-KE-norm projection onto (P, L) (float64). Used only to make the toy
    targets conserve angular momentum too; the model's decode has its own copy in torch."""
    n = len(m)
    a = np.zeros((6, 6 * n))
    for i in range(n):
        a[0:3, 6 * i:6 * i + 3] = m[i] * np.eye(3)
        rx = np.array([[0, -pos[i, 2], pos[i, 1]], [pos[i, 2], 0, -pos[i, 0]],
                       [-pos[i, 1], pos[i, 0], 0]])
        a[3:6, 6 * i:6 * i + 3] = m[i] * rx
        a[3:6, 6 * i + 3:6 * i + 6] = inertia_w[i]
    x = np.concatenate([np.concatenate([v[i], w[i]]) for i in range(n)])
    minv = np.zeros((6 * n, 6 * n))
    for i in range(n):
        minv[6 * i:6 * i + 3, 6 * i:6 * i + 3] = np.eye(3) / m[i]
        minv[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = np.linalg.inv(inertia_w[i])
    b = np.concatenate([p_target, l_target]) - a @ x
    lam = np.linalg.solve(a @ minv @ a.T, b)
    x = x + minv @ a.T @ lam
    x = x.reshape(n, 6)
    return x[:, :3], x[:, 3:]


def make_event(seed: int, *, nx: int | None = None, nz: int | None = None,
               anchored: bool | None = None, split: str = "train", family: str = "synthetic_wall",
               speed: float | None = None, ensemble: tuple[str, int, int] | None = None,
               gravity: bool = False, settle_calls: int = 0
               ) -> tuple[dict, list[S.Call], dict]:
    rng = np.random.default_rng(seed)
    nx = nx or int(rng.integers(4, 9))
    nz = nz or int(rng.integers(4, 8))
    anchored = bool(rng.random() < 0.3) if anchored is None else anchored
    L = 0.1
    rho = 2400.0
    m_c = rho * L ** 3
    f_t, E, GA0 = 3e6, 30e9, 4.0
    c = float(np.sqrt(E / rho))
    v0 = float(speed if speed is not None else rng.uniform(3.0, 60.0))
    m_imp = float(rng.uniform(5.0, 60.0))
    r_imp = 0.1
    strength = float(np.exp(rng.normal(0.0, 0.3)))
    nc = nx * nz
    # grid in the wall plane x–z, the ram along +y
    grid = np.array([[ix * L, 0.0, iz * L] for iz in range(nz) for ix in range(nx)])
    ends = []
    for iz in range(nz):
        for ix in range(nx):
            i = iz * nx + ix
            if ix + 1 < nx:
                ends.append((i, i + 1))
            if iz + 1 < nz:
                ends.append((i, i + nx))
    ends = np.array(ends, dtype=np.int64)
    nb = len(ends)
    is_anchor = np.zeros(nc, dtype=bool)
    if anchored:
        is_anchor[:nx] = True
    struck = int(rng.integers(nc))
    weib = np.exp(rng.normal(0.0, 0.1, size=nb))
    if ensemble is not None:  # members perturb something physical: speed and Weibull draws
        mrng = np.random.default_rng((seed, ensemble[1]))
        v0 *= 1.0 + 1e-3 * mrng.uniform(-1, 1)
        weib = np.exp(mrng.normal(0.0, 0.1, size=nb))

    # ---- toy outcome of the first call ---------------------------------------------
    ke_imp = 0.5 * m_imp * v0 ** 2
    ga = GA0 * strength * weib
    ratio = ke_imp / ga.sum()
    hop = _hops(nc, ends, np.ones(nb, bool), [struck])
    bond_hop = np.minimum(hop[ends[:, 0]], hop[ends[:, 1]])
    r = 0.08 * ratio ** 0.5 * np.exp(-bond_hop / 1.5) / (strength * weib)
    broke = r >= 1.0
    r_rec = np.where(broke, np.maximum(r, 1.0), r).astype(np.float32)  # failure margin q
    # Peak utilisation, as in the oracle: softening starts at r_util = 1 well before failure
    # (q = 1), so bonds with 1 ≤ r_util and q < 1 survive.
    util_rec = np.where(broke, np.maximum(1.6 * r, 1.0), 1.6 * r).astype(np.float32)
    frag = _components(nc, ends, ~broke)
    plug = frag == frag[struck]
    if anchored and np.any(is_anchor & plug):
        plug_mass = np.inf
    else:
        plug_mass = m_c * plug.sum()
    e_rest = 0.0 if broke.any() else 0.3
    if np.isinf(plug_mass):
        v_imp_end = -e_rest * v0
        v_plug = 0.0
    else:
        mt = m_imp + plug_mass
        v_imp_end = (m_imp * v0 - plug_mass * e_rest * v0) / mt
        v_plug = (m_imp * v0 * (1 + e_rest)) / mt
    regime = (S.REGIMES["perforate"] if broke.any() and v_imp_end > 0.5 * v0 * 0.2
              else S.REGIMES["rebound"] if v_imp_end < 0 else S.REGIMES["arrest"])

    # ---- scene placement --------------------------------------------------------------
    q_scene = random_quat(rng, 1)[0]
    rs = quat_to_matrix_np(q_scene)
    shift = rng.normal(0.0, 50.0, size=3)
    ram_pos_local = grid[struck] + np.array([0.0, -(L / 2 + r_imp) - 1e-3, 0.0])
    q_imp = quat_mul_np(q_scene, random_quat(rng, 1)[0])
    pos_all = np.vstack([grid, ram_pos_local[None]])
    mass_all = np.concatenate([np.full(nc, m_c), [m_imp]])
    com_local = (mass_all[:, None] * pos_all).sum(0) / mass_all.sum()
    rel_local = pos_all - com_local
    pos_w = rel_local @ rs.T  # world axes, relative to island COM
    yhat = rs @ np.array([0.0, 1.0, 0.0])
    v_start = np.zeros((nc + 1, 3))
    v_start[nc] = v0 * yhat
    w_start = np.zeros((nc + 1, 3))
    w_start[nc] = rs @ rng.normal(0.0, 2.0, size=3)
    v_end = np.zeros_like(v_start)
    v_end[:nc][plug] = (0.0 if np.isinf(plug_mass) else v_plug) * yhat
    v_end[nc] = v_imp_end * yhat
    w_end = w_start.copy()
    inert_c = np.full((nc, 3), m_c * L ** 2 / 6.0)
    inert_i = np.full(3, 0.4 * m_imp * r_imp ** 2)
    q_c = np.tile(q_scene, (nc, 1))
    inert_w = [quat_to_matrix_np(q_c[i]) @ np.diag(inert_c[i]) @ quat_to_matrix_np(q_c[i]).T
               for i in range(nc)]
    ri = quat_to_matrix_np(q_imp)
    inert_w.append(ri @ np.diag(inert_i) @ ri.T)
    inert_w = np.array(inert_w)
    dt = 1.0 / 60.0
    g_vec = rs @ np.array([0.0, 0.0, -G]) if gravity else np.zeros(3)
    def end_geometry(v1, w1):
        """Frame-end displacement, rotation, COM-relative positions and world inertia."""
        dx_ = 0.5 * (v_start + v1) * dt + 0.5 * g_vec * dt ** 2
        dr_ = 0.5 * (w_start + w1) * dt
        p1 = pos_w + dx_
        p1 = p1 - (mass_all[:, None] * p1).sum(0) / mass_all.sum()
        rd = rotvec_to_matrix(torch.from_numpy(dr_)).numpy()
        return dx_, dr_, p1, rd @ inert_w @ np.transpose(rd, (0, 2, 1))

    if not anchored:
        # Angular momentum is conserved with each instant's own geometry: L(t1) is taken
        # with the frame-end positions and inertia (as the decode does), so iterate the
        # projection with the geometry its own velocities imply (converges in a few steps).
        p0 = (mass_all[:, None] * v_start).sum(0)
        l0 = (np.cross(pos_w, mass_all[:, None] * v_start)).sum(0) + np.einsum(
            "nij,nj->ni", inert_w, w_start).sum(0)
        for _ in range(30):
            _, _, p1, i1 = end_geometry(v_end, w_end)
            v_new, w_new = _project_momentum_np(mass_all, i1, p1, v_end, w_end,
                                                p0 + mass_all.sum() * g_vec * dt, l0)
            done = max(np.abs(v_new - v_end).max(), np.abs(w_new - w_end).max()) < 1e-13
            v_end, w_end = v_new, w_new
            if done:
                break
    else:
        v_end[:nc][is_anchor] = 0.0
        w_end[:nc][is_anchor] = 0.0
    dx, drot, _, inert_end = end_geometry(v_end, w_end)

    def ke(v, w, inertia=None):
        inertia = inert_w if inertia is None else inertia
        return 0.5 * float((mass_all * (v ** 2).sum(1)).sum() + np.einsum(
            "ni,nij,nj->", w, inertia, w))

    ke_in, ke_out = ke(v_start, w_start), ke(v_end, w_end, inert_end)
    w_grav = float((mass_all[:, None] * g_vec[None] * dx).sum())
    d_bond_each = np.where(broke, ga, 0.02 * ga * np.minimum(r, 1.0)).astype(np.float32)
    ddamage = np.where(broke, 0.0, np.clip(0.5 * (r - 0.6), 0, 0.4)).astype(np.float32)
    e_el_end = 0.01 * ke_in
    ke_loss = ke_in + w_grav - ke_out
    d_bond = float(d_bond_each.astype(np.float64).sum())
    energy = {"KE_in": ke_in, "KE_out": ke_out, "W_gravity": w_grav, "W_external": 0.0,
              "E_elastic_start": 0.0, "E_elastic_end": e_el_end, "D_bond": d_bond,
              "D_bond_joint": float(ga[broke].sum()), "D_bond_damping":
              d_bond - float(ga[broke].sum()), "D_fracture_floor": float(ga[broke].sum()),
              "D_contact_booked": max(ke_loss - d_bond - e_el_end, 0.0),
              "D_friction_booked": 0.0, "ledger_residual": 0.0}
    v_ref = v0
    scales = {"L": L, "m": m_c, "v": v_ref, "c": c, "f_t": f_t, "E": E, "GA": GA0, "dt": dt,
              "E_in": max(ke_in + w_grav, 0.5 * m_c * v_ref ** 2)}

    # ---- features ----------------------------------------------------------------------
    def chunk_features(live: np.ndarray, vv: np.ndarray, ww: np.ndarray, damage: np.ndarray,
                       contact_chunks: list[int], t_contact: float) -> np.ndarray:
        f = np.zeros((nc, len(S.CHUNK_FEAT_COLUMNS)), np.float32)
        col = lambda n: S.col("chunk_feat", n)  # noqa: E731
        f[:, col("log_mass")] = 0.0
        for k in range(3):
            f[:, col(f"inertia_{k}")] = inert_c[:, k] / (m_c * L ** 2)
            f[:, col(f"half_extent_{k}")] = 0.5
        f[:, col("support_fixed")] = is_anchor
        deg_live = np.zeros(nc)
        deg_broken = np.zeros(nc)
        dmax = np.zeros(nc)
        dsum = np.zeros(nc)
        for (a, b), ok, d in zip(ends.tolist(), live.tolist(), damage.tolist()):
            for x in (a, b):
                if ok:
                    deg_live[x] += 1
                    dsum[x] += d
                    dmax[x] = max(dmax[x], d)
                else:
                    deg_broken[x] += 1
        f[:, col("bond_damage_mean")] = dsum / np.maximum(deg_live, 1)
        f[:, col("bond_damage_max")] = dmax
        f[:, col("n_live_bonds")] = deg_live / 6
        f[:, col("n_broken_bonds")] = deg_broken / 6
        h = _hops(nc, ends, live, contact_chunks) if contact_chunks else np.full(nc, 10**6)
        f[:, col("hops_to_contact")] = np.minimum(h / 16, 4)
        f[:, col("wave_arrival")] = np.minimum((max(t_contact, 0) * dt + h * L / c) / dt, 4)
        ha = _hops(nc, ends, live, list(np.nonzero(is_anchor)[0])) if anchored else np.full(
            nc, 10**6)
        f[:, col("hops_to_anchor")] = np.minimum(ha / 16, 4)
        vcom = (mass_all[:, None] * np.vstack([vv])).sum(0) / mass_all.sum()
        rinv = rs.T
        gb = (rinv @ g_vec) / G
        for k in range(3):
            f[:, col(f"gravity_body_{k}")] = gb[k]
        f[:, [col(f"v_rel_body_{k}") for k in range(3)]] = ((vv[:nc] - vcom) @ rinv.T) / v_ref
        f[:, [col(f"w_body_{k}") for k in range(3)]] = (ww[:nc] @ rinv.T) * L / v_ref
        f[:, col("punch_in_plug")] = (hop <= 1) if t_contact >= 0 else 0
        f[contact_chunks, col("in_contact_any")] = 1
        return f

    def impactor_features(vv: np.ndarray, in_contact: bool) -> np.ndarray:
        f = np.zeros((1, len(S.IMPACTOR_FEAT_COLUMNS)), np.float32)
        col = lambda n: S.col("impactor_feat", n)  # noqa: E731
        f[0, col("log_mass_island")] = np.log(m_imp / (m_c * nc))
        f[0, col("log_mass_chunk")] = np.log(m_imp / m_c)
        for k in range(3):
            f[0, col(f"inertia_{k}")] = 0.4
        f[0, col("log_radius")] = np.log(r_imp / L)
        f[0, col("log_modulus")] = np.log(200e9 / E)
        vcom = (mass_all[:, None] * vv).sum(0) / mass_all.sum()
        rb = quat_to_matrix_np(q_imp).T
        vrel = rb @ (vv[nc] - vcom)
        f[0, [col(f"v_rel_body_{k}") for k in range(3)]] = vrel / v_ref
        f[0, [col(f"gravity_body_{k}") for k in range(3)]] = rb @ g_vec / G
        f[0, col("speed_over_c")] = np.linalg.norm(vrel) / c
        f[0, col("in_contact_any")] = float(in_contact)
        return f

    r_body_c = rs
    r_body_i = quat_to_matrix_np(q_imp)

    def point_rel(vv, ww, s_, d_, pnt):
        """Row-frame-free relative velocity of dst vs src at world point `pnt`."""
        vs = vv[s_] + np.cross(ww[s_], pnt - pos_w[s_])
        vd = vv[d_] + np.cross(ww[d_], pnt - pos_w[d_])
        return vd - vs

    def bond_arrays(live: np.ndarray, vv: np.ndarray, ww: np.ndarray, damage: np.ndarray,
                    util: np.ndarray, peak_prev: np.ndarray, margin_prev: np.ndarray,
                    qs_log_util: np.ndarray):
        idx = np.nonzero(live)[0]
        rows, frames, src, dst = [], [], [], []
        col = lambda n_: S.col("bond_feat", n_)  # noqa: E731
        for k in idx:
            a, b = ends[k]
            f0 = _frame_from_normal(rs @ ((grid[b] - grid[a]) / L))
            pnt = 0.5 * (pos_w[a] + pos_w[b])
            for s_, d_, fr in ((a, b, f0), (b, a, f0 * REV)):
                f = np.zeros(len(S.BOND_FEAT_COLUMNS), np.float32)
                f[col("log_area")] = 0.0
                f[[col(f"offset_{j}") for j in range(3)]] = fr @ (pnt - pos_w[s_]) / L
                f[[col(f"log_strength_{j}") for j in range(3)]] = np.log(strength * weib[k])
                f[[col(f"log_ga_{j}") for j in range(3)]] = np.log(ga[k] / GA0)
                f[col("weibull")] = weib[k]
                f[col("damage")] = damage[k]
                f[col("fail_margin")] = margin_prev[k]
                f[col("log1p_utilization")] = np.log1p(util[k])
                f[col("log1p_peak_util_prev")] = np.log1p(peak_prev[k])
                f[col("log_qs_hertz_util")] = qs_log_util[k]
                f[col("punch_plug_bond")] = float(bond_hop[k] == 0)
                f[col("log_punch_capacity_ratio")] = (np.log(ratio) if bond_hop[k] == 0
                                                      else 0.0)
                f[[col(f"v_rel_edge_{j}") for j in range(3)]] = fr @ point_rel(
                    vv, ww, s_, d_, pnt) / v_ref
                f[col("rot_src_0"):col("rot_src_0") + 6] = _rot6(r_body_c, fr)
                f[col("rot_dst_0"):col("rot_dst_0") + 6] = _rot6(r_body_c, fr)
                rows.append(f)
                src.append(s_)
                dst.append(d_)
            frames.append(f0)
        return (np.array(src, np.int32), np.array(dst, np.int32),
                np.array(rows, np.float32).reshape(-1, len(S.BOND_FEAT_COLUMNS)),
                np.array(frames, np.float32).reshape(-1, 3, 3), idx)

    def contact_arrays(vv: np.ndarray, ww: np.ndarray, t_contact: float):
        n_w = yhat  # from the impactor (src of row 0) toward the struck chunk
        fr = _frame_from_normal(n_w)
        rows = []
        for self_imp in (True, False):
            f = np.zeros(len(S.IMPACTOR_CONTACT_FEAT_COLUMNS), np.float32)
            col = lambda n_: S.col("impactor_contact_feat", n_)  # noqa: E731
            vrel = vv[nc] - vv[struck]
            sign = 1.0 if self_imp else -1.0
            frr = fr if self_imp else fr * REV
            f[col("v_n")] = float(vrel @ n_w) / v_ref
            f[col("v_t_0")] = float(sign * vrel @ frr[0]) / v_ref
            f[col("v_t_1")] = float(sign * vrel @ frr[1]) / v_ref
            f[col("v_n_over_c")] = float(vrel @ n_w) / c
            f[col("log_mass_ratio")] = np.log(m_c / m_imp) if self_imp else np.log(m_imp / m_c)
            f[col("log_radius")] = np.log(r_imp / L)
            f[col("log_modulus")] = np.log(200e9 / E)
            f[[col(f"spin_{j}") for j in range(3)]] = frr @ ww[nc] * r_imp / v_ref
            f[col("t_contact")] = t_contact
            f[col("log_energy_ratio")] = np.log(ratio)
            f[col("log_hertz_force")] = np.log(1.0 + ratio)
            f[col("hertz_duration")] = 0.01
            f[col("is_self_impactor")] = float(self_imp)
            f[col("friction")] = 0.5
            rb_s, rb_d = (r_body_i, r_body_c) if self_imp else (r_body_c, r_body_i)
            f[col("rot_src_0"):col("rot_src_0") + 6] = _rot6(rb_s, frr)
            f[col("rot_dst_0"):col("rot_dst_0") + 6] = _rot6(rb_d, frr)
            rows.append(f)
        return (np.array([nc, struck], np.int32), np.array([struck, nc], np.int32),
                np.array(rows, np.float32), fr[None].astype(np.float32))

    chunk_key = ((0 << 32) | np.arange(nc)).astype(np.int64)
    bond_key = ((0 << 32) | np.arange(nb)).astype(np.int64)
    imp_key = np.array([1_000_000 + seed % 1000], np.int64)

    def empty(t: str, ncol: int):
        return {f"{t}_src": np.zeros(0, np.int32), f"{t}_dst": np.zeros(0, np.int32),
                f"{t}_key": np.zeros(0, np.int64), f"{t}_feat": np.zeros((0, ncol), np.float32),
                f"{t}_frame": np.zeros((0, 3, 3), np.float32)}

    def raw(vv, ww):
        return {
            "chunk_mass": np.full(nc, m_c, np.float32),
            "chunk_inertia_body": inert_c.astype(np.float32),
            "chunk_q_world": q_c.astype(np.float32), "chunk_pos": pos_w[:nc].astype(np.float32),
            "chunk_v": vv[:nc].astype(np.float32), "chunk_w": ww[:nc].astype(np.float32),
            "chunk_anchored": is_anchor.astype(np.int32),
            "impactor_mass": np.array([m_imp], np.float32),
            "impactor_inertia_body": inert_i[None].astype(np.float32),
            "impactor_q_world": q_imp[None].astype(np.float32),
            "impactor_pos": pos_w[nc:].astype(np.float32),
            "impactor_v": vv[nc:].astype(np.float32), "impactor_w": ww[nc:].astype(np.float32),
        }

    def global_feat(vv) -> np.ndarray:
        g = np.zeros((1, len(S.GLOBAL_FEAT_COLUMNS)), np.float32)
        col = lambda n_: S.col("global_feat", n_)  # noqa: E731
        ke_i = 0.5 * m_imp * float(vv[nc] @ vv[nc])
        g[0, col("log_energy_ratio")] = np.log(max(ke_i, 1e-9) / ga.sum())
        g[0, col("log_energy_ratio_plug")] = np.log(max(ke_i, 1e-9) / ga[bond_hop == 0].sum())
        g[0, col("log_dt_c_over_L")] = np.log(dt * c / L)
        g[0, col("dt_v_over_L")] = dt * v_ref / L
        g[0, col("n_impactors")] = 0.25
        g[0, col("log_n_chunks")] = np.log(nc) / 10
        g[0, col("gravity_mag")] = float(np.linalg.norm(g_vec)) / G
        g[0, col("anchored")] = float(anchored)
        g[0, col("v_over_c")] = v_ref / c
        return g

    calls: list[S.Call] = []
    # call 0: the strike
    live0 = np.ones(nb, bool)
    zeros_b = np.zeros(nb, np.float32)
    qs_log = (np.log(np.maximum(r, 1e-6)) + rng.normal(0, 0.3, nb)).astype(np.float32)
    bs, bd, bf, bfr, bidx = bond_arrays(live0, v_start, w_start, zeros_b,
                                        0.05 * np.minimum(r, 1), zeros_b, zeros_b, qs_log)
    cs, cd, cf, cfr = contact_arrays(v_start, w_start, 0.3)
    arrays0 = {
        "chunk_key": chunk_key, "chunk_feat": chunk_features(live0, v_start, w_start, zeros_b,
                                                            [struck], 0.3),
        "chunk_qs_feat": np.zeros((nc, 6), np.float32),
        "impactor_key": imp_key, "impactor_feat": impactor_features(v_start, True),
        "impactor_qs_feat": np.zeros((1, 6), np.float32),
        "global_feat": global_feat(v_start),
        "bond_qs_feat": np.zeros((len(bs), 3), np.float32),
        "bond_src": bs, "bond_dst": bd, "bond_key": bond_key[bidx], "bond_feat": bf,
        "bond_frame": bfr,
        "impactor_contact_src": cs, "impactor_contact_dst": cd,
        "impactor_contact_key": np.array([(min(imp_key[0], chunk_key[struck]) * 1_000_003)
                                          ^ max(imp_key[0], chunk_key[struck])], np.int64),
        "impactor_contact_feat": cf, "impactor_contact_frame": cfr,
        **empty("cracked", len(S.CRACKED_FEAT_COLUMNS)),
        **empty("structure_contact", len(S.STRUCTURE_CONTACT_FEAT_COLUMNS)),
        **raw(v_start, w_start),
        "y_bond_peak_util": util_rec, "y_bond_fail_margin": r_rec,
        "y_bond_broke": broke.astype(np.int32),
        "y_bond_break_bin": np.where(broke, np.clip(1 + bond_hop, 1, S.N_TIME_BINS), 0
                                     ).astype(np.int32),
        "y_bond_fail_mode": np.where(broke, 1, 0).astype(np.int32),
        "y_bond_ddamage": ddamage, "y_bond_dissipated": d_bond_each,
        "y_chunk_v_end": v_end[:nc].astype(np.float32),
        "y_chunk_w_end": w_end[:nc].astype(np.float32),
        "y_chunk_dx": dx[:nc].astype(np.float32), "y_chunk_drot": drot[:nc].astype(np.float32),
        "y_chunk_fragment": frag,
        "y_impactor_v_end": v_end[nc:].astype(np.float32),
        "y_impactor_w_end": w_end[nc:].astype(np.float32),
        "y_impactor_dx": dx[nc:].astype(np.float32),
        "y_impactor_drot": drot[nc:].astype(np.float32),
        "y_impactor_regime": np.array([regime], np.int32),
        "base_qs_bond_broke": np.zeros(nb, np.int32),
        "base_qs_bond_peak_util": np.exp(qs_log).astype(np.float32),
        "base_qs_chunk_v_end": np.zeros((nc, 3), np.float32),
        "base_qs_chunk_w_end": np.zeros((nc, 3), np.float32),
        "base_qs_impactor_v_end": v_start[nc:].astype(np.float32) * 0.2,
    }
    counts0 = {"chunks": nc, "impactors": 1, "bond": nb, "cracked": 0, "impactor_contact": 1,
               "structure_contact": 0}
    calls.append(S.Call(event_id="", call_index=0, frame=10, island=0, prev_call=None,
                        phase="contact", transition=True, routed=True, anchored=anchored,
                        counts=counts0, scales=scales, energy=energy, arrays=arrays0))

    # call 1: separation (no contact; nothing further breaks; ballistic)
    live1 = ~broke
    damage1 = ddamage
    bs, bd, bf, bfr, bidx = bond_arrays(live1, v_end, w_end, damage1, 0.02 * np.ones(nb),
                                        util_rec, r_rec, np.full(nb, -3.0, np.float32))
    crk = np.nonzero(broke)[0]
    crows, csrc, cdst, cfrs = [], [], [], []
    for k in crk:
        a, b = ends[k]
        f0 = _frame_from_normal(rs @ ((grid[b] - grid[a]) / L))
        pnt = 0.5 * (pos_w[a] + pos_w[b])
        for s_, d_, fr in ((a, b, f0), (b, a, f0 * REV)):
            f = np.zeros(len(S.CRACKED_FEAT_COLUMNS), np.float32)
            col = lambda n_: S.col("cracked_feat", n_)  # noqa: E731
            f[[col(f"offset_{j}") for j in range(3)]] = fr @ (pnt - pos_w[s_]) / L
            rel = point_rel(v_end, w_end, s_, d_, pnt)
            f[col("gap")] = float(rel @ fr[2]) * dt / L
            f[col("friction")] = 0.5
            f[[col(f"v_rel_edge_{j}") for j in range(3)]] = fr @ rel / v_ref
            f[col("frames_since_broken")] = 1 / 8
            f[col("rot_src_0"):col("rot_src_0") + 6] = _rot6(r_body_c, fr)
            f[col("rot_dst_0"):col("rot_dst_0") + 6] = _rot6(r_body_c, fr)
            crows.append(f)
            csrc.append(s_)
            cdst.append(d_)
        cfrs.append(f0)
    v1_end = v_end.copy()
    w1_end = w_end.copy()
    if not anchored and gravity:
        v1_end = v1_end + g_vec * dt
    dx1 = 0.5 * (v_end + v1_end) * dt + 0.5 * g_vec * dt ** 2 * 0
    drot1 = w_end * dt
    nb1 = int(live1.sum())
    ke1 = ke(v_end, w_end)
    ke1o = ke(v1_end, w1_end)
    wg1 = float((mass_all[:, None] * g_vec[None] * dx1).sum())
    d1 = np.zeros(nb1, np.float32)
    energy1 = {"KE_in": ke1, "KE_out": ke1o, "W_gravity": wg1, "W_external": 0.0,
               "E_elastic_start": e_el_end, "E_elastic_end": e_el_end * 0.5, "D_bond": 0.0,
               "D_bond_joint": 0.0, "D_bond_damping": 0.0, "D_fracture_floor": 0.0,
               "D_contact_booked": 0.0, "D_friction_booked": 0.0, "ledger_residual": 0.0}
    scales1 = dict(scales)
    scales1["E_in"] = max(ke1 + wg1, 0.5 * m_c * v_ref ** 2)
    arrays1 = {
        "chunk_key": chunk_key, "chunk_feat": chunk_features(live1, v_end, w_end, damage1, [],
                                                            -1.0),
        "chunk_qs_feat": np.zeros((nc, 6), np.float32),
        "impactor_key": imp_key, "impactor_feat": impactor_features(v_end, False),
        "impactor_qs_feat": np.zeros((1, 6), np.float32),
        "global_feat": global_feat(v_end),
        "bond_qs_feat": np.zeros((len(bs), 3), np.float32),
        "bond_src": bs, "bond_dst": bd, "bond_key": bond_key[bidx], "bond_feat": bf,
        "bond_frame": bfr,
        "cracked_src": np.array(csrc, np.int32), "cracked_dst": np.array(cdst, np.int32),
        "cracked_key": bond_key[crk], "cracked_feat": np.array(crows, np.float32).reshape(
            -1, len(S.CRACKED_FEAT_COLUMNS)),
        "cracked_frame": np.array(cfrs, np.float32).reshape(-1, 3, 3),
        **empty("impactor_contact", len(S.IMPACTOR_CONTACT_FEAT_COLUMNS)),
        **empty("structure_contact", len(S.STRUCTURE_CONTACT_FEAT_COLUMNS)),
        **raw(v_end, w_end),
        "y_bond_peak_util": np.full(nb1, 0.02, np.float32),
        "y_bond_fail_margin": r_rec[live1],  # κ is a history maximum: the margin stays
        "y_bond_broke": np.zeros(nb1, np.int32), "y_bond_break_bin": np.zeros(nb1, np.int32),
        "y_bond_fail_mode": np.zeros(nb1, np.int32), "y_bond_ddamage": np.zeros(nb1, np.float32),
        "y_bond_dissipated": d1,
        "y_chunk_v_end": v1_end[:nc].astype(np.float32),
        "y_chunk_w_end": w1_end[:nc].astype(np.float32),
        "y_chunk_dx": dx1[:nc].astype(np.float32), "y_chunk_drot": drot1[:nc].astype(np.float32),
        "y_chunk_fragment": frag,
        "y_impactor_v_end": v1_end[nc:].astype(np.float32),
        "y_impactor_w_end": w1_end[nc:].astype(np.float32),
        "y_impactor_dx": dx1[nc:].astype(np.float32),
        "y_impactor_drot": drot1[nc:].astype(np.float32),
        "y_impactor_regime": np.array([regime], np.int32),
    }
    counts1 = {"chunks": nc, "impactors": 1, "bond": nb1, "cracked": len(crk),
               "impactor_contact": 0, "structure_contact": 0}
    calls.append(S.Call(event_id="", call_index=1, frame=11, island=0, prev_call=0,
                        phase="separation", transition=True, routed=True, anchored=anchored,
                        counts=counts1, scales=scales1, energy=energy1, arrays=arrays1))

    # extra ballistic settling calls (exact continuations only without gravity)
    if settle_calls and gravity:
        raise ValueError("settle_calls needs gravity=False (the toy copies the ballistic call)")
    for k in range(settle_calls):
        prev = calls[-1]
        arr = dict(prev.arrays)
        bf = arr["bond_feat"].copy()  # carried: last frame's peak demand and damage
        bf[:, S.col("bond_feat", "log1p_peak_util_prev")] = np.repeat(
            np.log1p(prev.arrays["y_bond_peak_util"]), 2)
        bf[:, S.col("bond_feat", "fail_margin")] = np.repeat(prev.arrays["y_bond_fail_margin"], 2)
        bf[:, S.col("bond_feat", "damage")] += np.repeat(prev.arrays["y_bond_ddamage"], 2)
        arr["bond_feat"] = bf
        calls.append(S.Call(**{**prev.__dict__, "call_index": prev.call_index + 1,
                               "frame": prev.frame + 1, "prev_call": prev.call_index,
                               "phase": "settling", "transition": False, "arrays": arr}))

    eid = f"syn_{seed:06d}" + (f"_m{ensemble[1]}" if ensemble else "")
    for cl in calls:
        cl.event_id = eid
        cl.gravity = tuple(float(x) for x in g_vec)
        cl.family, cl.split = family, split
    manifest = {
        "event_id": eid, "recipe_id": f"synthetic/{seed}", "family": family,
        "layout_family": f"grid_{nx}x{nz}", "split_group": f"syn_{seed}" if not ensemble
        else ensemble[0], "split": split,
        "ensemble_id": ensemble[0] if ensemble else None,
        "ensemble_member": ensemble[1] if ensemble else None,
        "ensemble_size": ensemble[2] if ensemble else 1,
        "perturbation": {"offset_m": [0, 0, 0], "speed_scale": 1.0, "weibull_seed": 0},
        "seed": seed, "oracle": {"config": "synthetic", "methods": "toy", "contact": "none",
                                  "hash": "0"},
        "scene_hash": f"{seed:016x}", "recorder_revision": "synthetic",
        "frame_dt": dt, "gravity": g_vec.tolist(), "truncated": False,
        "available": {k: k in ("qs_hertz", "punch", "qs_frame", "cracked_edges",
                               "bond_energy_split") for k in S.AVAILABLE_FLAGS},
        "outcome": {"impactors": [{"key": int(imp_key[0]), "v_exit": v1_end[nc].tolist(),
                                   "speed_exit": float(v_imp_end),
                                   "regime": [k for k, v in S.REGIMES.items() if v == regime][0]}],
                    "n_fragments": int(frag.max() + 1),
                    "largest_fraction": float(np.bincount(frag).max() / nc),
                    "fragment_masses": sorted((np.bincount(frag) * m_c).tolist()),
                    "retained_anchored_fraction": 1.0, "collapsed": False,
                    "sim_time": 2 * dt, "frames": 2},
    }
    structures = {0: {"chunk_key": chunk_key, "bond_key": bond_key,
                      "bond_ends": ends.astype(np.int32),
                      "chunk_rest_pos": (grid @ rs.T + shift).astype(np.float32)}}
    return manifest, calls, structures


def write_dataset(root: Path | str, n_events: int, seed: int = 0, val_fraction: float = 0.2,
                  ensembles: int = 0, ensemble_size: int = 4, **kw) -> list[Path]:
    root = Path(root)
    paths = []
    for i in range(n_events):
        split = "val" if (i % max(1, round(1 / val_fraction)) == 0) else "train"
        m, calls, st = make_event(seed + i, split=split, **kw)
        paths.append(S.write_event(root / m["event_id"], m, calls, st))
    for e in range(ensembles):
        base = seed + 10_000 + e
        for k in range(ensemble_size):
            rng_speed = None
            m, calls, st = make_event(base, split="train", speed=rng_speed,
                                      ensemble=(f"ens_{base}", k, ensemble_size), **kw)
            paths.append(S.write_event(root / m["event_id"], m, calls, st))
    return paths
