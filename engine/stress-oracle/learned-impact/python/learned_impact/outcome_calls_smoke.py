"""SMOKE converter: `learned-impact-frames/1` events → P2's `learned-impact-outcome/1` calls.

    python -m learned_impact.outcome_calls_smoke --p2-python <P2 worktree>/engine/stress-oracle/learned-impact/python \
        --out CALLS_DIR EVENT_RUN_DIR [EVENT_RUN_DIR ...]

**Temporary, for P2's first training smoke test only.** The agreed single source of every
model feature is the Rust call builder (`stress-learned/src/outcome_calls.rs`), which P3
ports; this file exists so that training can be checked end to end before that lands, and
it is to be deleted when it does. Data built with it must not be used for results.

Scope: one call per frame with a routed island (every routed chunk and impactor, island 0,
as P0's harness does). Written in full: chunk, impactor and global features, bond and
impactor-contact edges, the variant-B QS groups, every target, the energies and the
baselines. Written empty and flagged unavailable: cracked-face and structure-contact edges.
Assumptions, stated: chunk rest orientations are the identity (true of every pilot wall);
`v` is the largest predicted contact speed of the island (else the largest impactor speed
relative to the island's centre of mass), floored at 0.05 m/s; contact tangents follow the
source body's x axis (rotation-equivariant).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .outcome_audit import event_dirs, load_event, PHASES as P1_PHASES

G0 = 9.81
TINY = 1e-6


def qmat(q):
    """Rotation matrices of wxyz quaternions [..., 4] → [..., 3, 3] (body → world)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rotvec_between(q0, q1):
    """World rotation vector r with R(q1) = exp(r)·R(q0)."""
    r = qmat(q1) @ np.swapaxes(qmat(q0), -1, -2)
    tr = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    ang = np.arccos(tr)
    axis = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], -1)
    s = np.sin(ang)
    k = np.where(s > 1e-9, ang / (2 * np.where(s > 1e-9, s, 1)), 0.5)
    return axis * k[..., None]


def tangent(n, ref):
    """A unit tangent ⟂ n from `ref` (a body axis), falling back to a second axis."""
    t = ref - n * (ref @ n)
    if np.linalg.norm(t) < 1e-6:
        alt = np.array([ref[1], ref[2], ref[0]])
        t = alt - n * (alt @ n)
    return t / np.linalg.norm(t)


def rot6(r_body, frame_rows):
    """First two columns of R_bodyᵀ·R_edge, R_edge's columns being (t1, t2, n)."""
    m = r_body.T @ frame_rows.T
    return np.concatenate([m[:, 0], m[:, 1]])


def support(r, half, n):
    """Half-thickness of a box (rotation r, half extents) along unit n."""
    return float(np.abs(r.T @ n) @ half)


def bfs(n, adj, sources):
    d = np.full(n, np.inf)
    q = list(sources)
    for s in q:
        d[s] = 0
    i = 0
    while i < len(q):
        u = q[i]
        i += 1
        for v in adj[u]:
            if d[v] == np.inf:
                d[v] = d[u] + 1
                q.append(v)
    return d


def dijkstra(n, adj_w, init):
    import heapq
    d = np.full(n, np.inf)
    h = []
    for s, t0 in init:
        if t0 < d[s]:
            d[s] = t0
            heapq.heappush(h, (t0, s))
    while h:
        t, u = heapq.heappop(h)
        if t > d[u]:
            continue
        for v, w in adj_w[u]:
            if t + w < d[v]:
                d[v] = t + w
                heapq.heappush(h, (t + w, v))
    return d


def convert_event(ev, schema, out_root: Path):
    ctx, o = ev.context, ev.outcome
    scene = json.loads((ev.path / "scene.json").read_text())
    rev = int(ctx.get("recorder_revision", 1))
    if rev < 2:
        raise ValueError(f"{ev.path.name}: recorder revision {rev} has no per-bond energies")
    A = ev.array
    F, N, E, I = ev.n_frames, ctx["n_chunks"], ctx["n_bonds"], ctx["n_impactors"]
    dt = scene["sim"]["frame_dt"]
    scale_k = scene["sim"].get("stiffness_scale", 1.0)
    g = np.array(scene["gravity"], np.float64)
    restitution = scene["sim"]["contact_restitution"]
    mats = scene["materials"]
    # Static
    mass = A("static_chunk_mass")[0].astype(np.float64)
    inertia = A("static_chunk_inertia_diag")[0].astype(np.float64)
    half = A("static_chunk_half_extents")[0].astype(np.float64)
    center_local = A("static_chunk_center_local")[0].astype(np.float64)
    cstruct = A("static_chunk_structure")[0]
    ends = A("static_bond_endpoints")[0]
    geom = A("static_bond_geometry")[0].astype(np.float64)
    stiff = A("static_bond_stiffness")[0].astype(np.float64)
    strength = A("static_bond_strength")[0].astype(np.float64)
    weib = A("static_bond_weibull")[0].astype(np.float64)
    keys_c = ctx["chunk_keys"]
    keys_b = ctx["bond_keys"]
    ckey = np.array([(s << 32) | c for s, c in keys_c], np.int64)
    bkey = np.array([(s << 32) | src for s, src, _, _ in keys_b], np.int64)
    # Per-chunk material, support kind
    chunk_desc = [c for b in scene["bodies"] for c in b["chunks"]]
    bond_desc = [bd for b in scene["bodies"] for bd in b["bonds"]]
    for c in chunk_desc:
        if c["orientation"] != [1.0, 0.0, 0.0, 0.0]:
            raise ValueError(f"{ev.path.name}: rest orientations other than identity are not handled by the smoke converter")
    cmat = [mats[c["material"]] for c in chunk_desc]
    E_c = np.array([m["youngs_modulus"] * scale_k for m in cmat])
    rho_c = np.array([m["density"] for m in cmat])
    fric_c = np.array([m["friction"] for m in cmat])
    fixed = np.array([c["support"] == "fixed" for c in chunk_desc])
    pinned = np.array([c["support"] not in ("none", "fixed") for c in chunk_desc])
    rebar = np.array([bd.get("rebar") is not None for bd in bond_desc])
    imps = scene["impactors"]
    imp_mass = np.array([i["mass"] for i in imps], np.float64)
    imp_E = np.array([mats[i["material"]]["youngs_modulus"] * scale_k for i in imps])
    imp_fric = np.array([mats[i["material"]]["friction"] for i in imps])
    imp_R, imp_inertia, imp_half = [], [], []
    for i in imps:
        sh = i["shape"]
        if sh["type"] == "sphere":
            r = sh["radius"]
            imp_R.append(r)
            imp_inertia.append([0.4 * i["mass"] * r * r] * 3)
            imp_half.append([r, r, r])
        else:
            h = np.array(sh["half_extents"])
            a, b, c = 2 * h
            imp_R.append(float(h.min()))
            imp_inertia.append([i["mass"] / 12 * (b * b + c * c), i["mass"] / 12 * (a * a + c * c), i["mass"] / 12 * (a * a + b * b)])
            imp_half.append(list(h))
    imp_R, imp_inertia, imp_half = np.array(imp_R), np.array(imp_inertia), np.array(imp_half)
    crushable = np.array([i.get("crush") is not None for i in imps])
    # Frames
    pos, vel, ang = A("state_chunk_pos").astype(np.float64), A("state_chunk_vel").astype(np.float64), A("state_chunk_angvel").astype(np.float64)
    qw = A("state_chunk_q_world").astype(np.float64)
    active = A("state_chunk_active")
    cluster = A("state_chunk_cluster")
    joint = A("state_bond_joint").astype(np.float64)
    conn = A("state_bond_connected")
    benergy = A("state_bond_energy").astype(np.float64)
    bmode = A("state_bond_mode")
    impst = A("state_impactor").astype(np.float64)
    routed = A("frame_chunk_routed")
    imp_routed = A("frame_impactor_routed")
    plug = A("frame_chunk_punch_plug")
    prior = A("frame_bond_qs_prior_util").astype(np.float64)
    peak = A("frame_bond_peak_util").astype(np.float64)
    bins = A("frame_bond_break_bin")
    phase = A("frame_phase")
    dis = A("frame_bond_dissipated").astype(np.float64) if "frame_bond_dissipated" in ev.catalog else np.diff(benergy[:, :, :2], axis=0)
    ci = A("contact_i32") if ev.catalog["contact_i32"]["shape"][0] else np.zeros((0, 5), np.int32)
    cf = A("contact_f32").astype(np.float64) if ev.catalog["contact_f32"]["shape"][0] else np.zeros((0, 17))
    qs = {k: A(f"qs_{k}") for k in ("bond_broken", "bond_peak_util", "chunk_vel", "chunk_angvel", "impactor")} if "qs_bond_broken" in ev.catalog else None
    punch_frames = list(A("punch_frame")) if ev.catalog["punch_frame"]["shape"][0] else []
    pu = {k: A(f"punch_{k}") for k in ("bond_broken", "bond_peak_util", "chunk_vel", "chunk_angvel", "impactor")} if punch_frames else None
    frames_json = ev.frames
    regime = {"arrest": 0, "perforate": 1, "rebound": 2, "deflect": 3}
    imp_regime = np.array([regime.get(x["regime"], -1) for x in o["impactors"]], np.int32)
    # Pristine graph per structure
    structures = {}
    for s in sorted(set(cstruct.tolist())):
        cm = np.where(cstruct == s)[0]
        bm = np.array([k for k in range(E) if keys_b[k][0] == s], int)
        local = {int(c): j for j, c in enumerate(cm)}
        structures[int(s)] = {
            "chunk_key": ckey[cm], "bond_key": bkey[bm],
            "bond_ends": np.array([[local[int(ends[k, 0])], local[int(ends[k, 1])]] for k in bm], np.int32).reshape(-1, 2),
            "chunk_rest_pos": center_local[cm].astype(np.float32),
        }
    # Bond adjacency is per frame (live bonds).
    calls, prev_frame_call = [], {}
    broke_frame = np.full(E, -1)
    for f in range(F):
        broke_frame[(conn[f] == 0) & (broke_frame < 0)] = max(f - 1, 0)
        isl = np.where((routed[f] == 1) & (active[f] == 1))[0]
        iimp = np.where(imp_routed[f] == 1)[0]
        if isl.size == 0:
            continue
        nc, ni = isl.size, iimp.size
        node_of = {int(c): j for j, c in enumerate(isl)}
        Rb = qmat(qw[f, isl])  # chunk body → world
        Ri = qmat(impst[f, iimp, 3:7]) if ni else np.zeros((0, 3, 3))
        xi, vi, wi = impst[f, iimp, 0:3], impst[f, iimp, 7:10], impst[f, iimp, 10:13]
        xc, vc, wc = pos[f, isl], vel[f, isl], ang[f, isl]
        mc = mass[isl]
        mall = np.concatenate([mc, imp_mass[iimp]])
        com = (np.concatenate([xc, xi]) * mall[:, None]).sum(0) / mall.sum()
        vcom = (np.concatenate([vc, vi]) * mall[:, None]).sum(0) / mall.sum()
        live = np.where((conn[f] == 1) & np.isin(ends[:, 0], isl) & np.isin(ends[:, 1], isl))[0]
        # contacts of this frame within the island
        crow = np.where((ci[:, 0] == f) & np.isin(ci[:, 3], isl) & np.isin(ci[:, 1], iimp))[0] if len(ci) else np.array([], int)
        # scales
        L = float(np.cbrt(np.median(8 * half[isl].prod(1))))
        m = float(np.median(mc))
        if crow.size:
            v = float(cf[crow, 7].max())
        elif ni:
            v = float(np.linalg.norm(vi - vcom, axis=1).max())
        else:
            v = 5.0
        v = max(v, 0.05)
        dom = np.bincount(np.arange(nc), weights=mc).argmax()
        c_wave = float(np.sqrt(E_c[isl][dom] / rho_c[isl][dom]))
        f_t = float(np.median(strength[live, 0])) if live.size else float(cmat[isl[0]]["tensile_strength"])
        Emed = float(np.median(E_c[isl]))
        GA_b = strength[:, 5] * geom[:, 9]
        GA = float(np.median(GA_b[live])) if live.size else float(cmat[isl[0]]["fracture_energy"]["tension"]) * L * L
        ke_c = 0.5 * mc * (vc * vc).sum(1) + 0.5 * np.einsum("ni,nij,nj->n", wc, Rb @ (inertia[isl][:, :, None] * np.swapaxes(Rb, 1, 2)), wc)
        ke_i = 0.5 * imp_mass[iimp] * (vi * vi).sum(1) + (0.5 * np.einsum("ni,nij,nj->n", wi, Ri @ (imp_inertia[iimp][:, :, None] * np.swapaxes(Ri, 1, 2)), wi) if ni else 0)
        KE_in = float(ke_c.sum() + np.sum(ke_i))
        Wg_est = float((mall * ((np.concatenate([vc, vi]) * dt + 0.5 * g * dt * dt) @ g)).sum())
        E_in = max(KE_in + Wg_est, 0.5 * m * v * v)
        scales = {"L": L, "m": m, "v": v, "c": c_wave, "f_t": f_t, "E": Emed, "GA": GA, "dt": dt, "E_in": E_in}
        # graph distances over live bonds
        adj = [[] for _ in range(nc)]
        adjw = [[] for _ in range(nc)]
        for k in live:
            a, b = node_of[int(ends[k, 0])], node_of[int(ends[k, 1])]
            adj[a].append(b); adj[b].append(a)
            d = float(np.linalg.norm(xc[a] - xc[b]))
            adjw[a].append((b, d / c_wave)); adjw[b].append((a, d / c_wave))
        entry = [node_of[int(ci[r, 3])] for r in crow]
        hops_c = bfs(nc, adj, entry) if entry else np.full(nc, np.inf)
        arr = dijkstra(nc, adjw, [(node_of[int(ci[r, 3])], cf[r, 0]) for r in crow]) if entry else np.full(nc, np.inf)
        anch = [j for j, c in enumerate(isl) if fixed[c] or pinned[c]]
        hops_a = bfs(nc, adj, anch) if anch else np.full(nc, np.inf)
        cap4 = lambda x: np.minimum(np.where(np.isfinite(x), x, 4.0), 4.0)
        # per chunk bond summaries
        dmean, dmax, nlive, nbroken = np.zeros(nc), np.zeros(nc), np.zeros(nc), np.zeros(nc)
        lsum = np.zeros(nc)
        for k in range(E):
            for e_ in ends[k]:
                j = node_of.get(int(e_))
                if j is None:
                    continue
                if conn[f, k]:
                    nlive[j] += 1; lsum[j] += joint[f, k, 0]; dmax[j] = max(dmax[j], joint[f, k, 0])
                else:
                    nbroken[j] += 1
        dmean = np.where(nlive > 0, lsum / np.maximum(nlive, 1), 0)
        contact_nodes = set(entry) | {nc + int(np.where(iimp == ci[r, 1])[0][0]) for r in crow}
        RbT = np.swapaxes(Rb, 1, 2)
        chunk_feat = np.zeros((nc, 33))
        chunk_feat[:, 0] = np.log(mc / m)
        chunk_feat[:, 1:4] = inertia[isl] / (mc[:, None] * L * L)
        chunk_feat[:, 4:7] = half[isl] / L
        chunk_feat[:, 7] = fixed[isl]
        chunk_feat[:, 8] = pinned[isl]
        chunk_feat[:, 9], chunk_feat[:, 10] = dmean, dmax
        chunk_feat[:, 11], chunk_feat[:, 12] = nlive / 6, nbroken / 6
        chunk_feat[:, 13] = cap4(hops_c / 16)
        chunk_feat[:, 14] = cap4(arr / dt)
        chunk_feat[:, 15] = cap4(hops_a / 16)
        chunk_feat[:, 16:19] = (RbT @ g) / G0
        chunk_feat[:, 19:22] = np.einsum("nij,nj->ni", RbT, vc - vcom) / v
        chunk_feat[:, 22:25] = np.einsum("nij,nj->ni", RbT, wc) * L / v
        chunk_feat[:, 25] = plug[f, isl] > 0
        chunk_feat[:, 32] = [j in contact_nodes for j in range(nc)]
        RiT = np.swapaxes(Ri, 1, 2)
        imp_feat = np.zeros((ni, 16))
        if ni:
            imp_feat[:, 0] = np.log(imp_mass[iimp] / mc.sum())
            imp_feat[:, 1] = np.log(imp_mass[iimp] / m)
            imp_feat[:, 2:5] = imp_inertia[iimp] / (imp_mass[iimp, None] * imp_R[iimp, None] ** 2)
            imp_feat[:, 5] = np.log(imp_R[iimp] / L)
            imp_feat[:, 6] = np.log(imp_E[iimp] / Emed)
            imp_feat[:, 7] = crushable[iimp]
            imp_feat[:, 8:11] = (RiT @ g) / G0
            vr = np.einsum("nij,nj->ni", RiT, vi - vcom)
            imp_feat[:, 11:14] = vr / v
            imp_feat[:, 14] = np.linalg.norm(vi - vcom, axis=1) / c_wave
            imp_feat[:, 15] = [(nc + k) in contact_nodes for k in range(ni)]
        sum_ke_imp = float(np.sum(ke_i)) if ni else 0.0
        plug_bonds = [k for k in live if (plug[f, ends[k, 0]] > 0) != (plug[f, ends[k, 1]] > 0)]
        ga_plug = float(GA_b[plug_bonds].sum()) if plug_bonds else 0.0
        anchored_isl = bool((fixed[isl] | pinned[isl]).any())
        glob = np.zeros((1, 12))
        glob[0, 0] = np.log(sum_ke_imp / GA_b[live].sum()) if sum_ke_imp > 0 and live.size else 0.0
        glob[0, 1] = np.log(sum_ke_imp / ga_plug) if sum_ke_imp > 0 and ga_plug > 0 else 0.0
        glob[0, 2] = dt * c_wave / L
        glob[0, 3] = dt * v / L
        glob[0, 4] = ni / 4
        glob[0, 5] = np.log(nc) / 10
        glob[0, 6] = np.linalg.norm(g) / G0
        glob[0, 7] = anchored_isl
        glob[0, 11] = v / c_wave
        # bond edges
        bsrc, bdst, bfeat, bframe = [], [], [], []
        for k in live:
            a, b = int(ends[k, 0]), int(ends[k, 1])
            ja, jb = node_of[a], node_of[b]
            Ra = Rb[ja]
            nrm = Ra @ geom[k, 3:6]
            t1 = Ra @ geom[k, 6:9]
            if nrm @ (xc[jb] - xc[ja]) < 0:
                nrm, t1 = -nrm, -t1
            t2 = np.cross(nrm, t1)
            p = xc[ja] + Ra @ (geom[k, 0:3] - center_local[a])
            area = geom[k, 9]
            base = [np.log(area / (L * L))]
            kref_l, kref_a = Emed * area / L, Emed * area * L
            kn, ks, kb1, kb2, kt = stiff[k]
            lstiff = [np.log(max(kn, TINY) / kref_l), np.log(max(ks, TINY) / kref_l), np.log(max(ks, TINY) / kref_l),
                      np.log(max(kb1, TINY) / kref_a), np.log(max(kb2, TINY) / kref_a), np.log(max(kt, TINY) / kref_a)]
            lstr = [np.log(max(strength[k, 0], TINY) / f_t), np.log(max(strength[k, 2], TINY) / f_t), np.log(max(strength[k, 1], TINY) / f_t)]
            lga = [np.log(max(strength[k, 5 + i] * area, TINY) / GA) for i in range(3)]
            pp = prior[f, k]
            pr = np.log(max(pp, TINY)) if pp >= 0 else 0.0
            in_plug = k in plug_bonds
            ratio = 0.0
            if in_plug:
                pid = max(plug[f, a], plug[f, b])
                rows = [r for r in crow if r == pid - 1] or [pid - 1]
                ratio = np.log(max(cf[rows[0], 13], TINY)) if rows[0] < len(cf) else 0.0
            common = [weib[k], joint[f, k, 0], joint[f, k, 1], joint[f, k, 3], peak[f - 1, k] if f > 0 else 0.0, float(rebar[k]), pr, float(in_plug), ratio]
            for (s_, d_, fr) in ((ja, jb, np.stack([t1, t2, nrm])), (jb, ja, np.stack([-t1, t2, -nrm]))):
                vs = (vc[s_] + np.cross(wc[s_], p - xc[s_]))
                vd = (vc[d_] + np.cross(wc[d_], p - xc[d_]))
                row = base + list(fr @ (p - xc[s_]) / L) + lstiff + lstr + lga + common + list(fr @ (vd - vs) / v) + list(rot6(Rb[s_], fr)) + list(rot6(Rb[d_], fr))
                bsrc.append(s_); bdst.append(d_); bfeat.append(row)
            bframe.append(np.stack([t1, t2, nrm]))
        # impactor contact edges
        isrc, idst, ifeat, iframe, ikeys = [], [], [], [], []
        for r in crow:
            k_imp = int(np.where(iimp == ci[r, 1])[0][0])
            js, jd = nc + k_imp, node_of[int(ci[r, 3])]
            ch = int(ci[r, 3])
            nrm = cf[r, 4:7] / np.linalg.norm(cf[r, 4:7])
            t1 = tangent(nrm, Ri[k_imp][:, 0])
            t2 = np.cross(nrm, t1)
            xs, xd = xi[k_imp], xc[jd]
            gap = (xd - xs) @ nrm - support(Ri[k_imp], imp_half[iimp[k_imp]], nrm) - support(Rb[jd], half[ch], nrm)
            hertz_f = np.log(max(cf[r, 12], TINY) / (f_t * float(np.median(geom[live, 9])) if live.size else f_t * L * L))
            ek = 0.5 * cf[r, 8] * cf[r, 7] ** 2
            ler = np.log(ek / ga_plug) if ga_plug > 0 and cf[r, 15] >= 0 and plug[f].max() > 0 else 0.0
            fr_pair = float(np.sqrt(imp_fric[iimp[k_imp]] * fric_c[ch]))
            tcon = -1.0 if cf[r, 0] <= 0 else cf[r, 0] / dt
            for (s_, d_, fr, self_) in ((js, jd, np.stack([t1, t2, nrm]), 1.0), (jd, js, np.stack([-t1, t2, -nrm]), 0.0)):
                vs = vi[k_imp] if s_ >= nc else vc[s_]
                vd = vi[k_imp] if d_ >= nc else vc[d_]
                rel = fr @ (vs - vd)
                ms = imp_mass[iimp[k_imp]] if s_ >= nc else mc[s_]
                md = imp_mass[iimp[k_imp]] if d_ >= nc else mc[d_]
                Rs = Ri[k_imp] if s_ >= nc else Rb[s_]
                Rd = Ri[k_imp] if d_ >= nc else Rb[d_]
                row = [rel[2] / v, rel[0] / v, rel[1] / v, rel[2] / c_wave, np.log(md / ms), np.log(imp_R[iimp[k_imp]] / L),
                       np.log(imp_E[iimp[k_imp]] / E_c[ch]), *(fr @ wi[k_imp] * imp_R[iimp[k_imp]] / v), tcon, ler, hertz_f,
                       cf[r, 11] / dt, self_, max(0.0, -gap) / L, fr_pair, restitution, *rot6(Rs, fr), *rot6(Rd, fr)]
                isrc.append(s_); idst.append(d_); ifeat.append(row)
            iframe.append(np.stack([t1, t2, nrm]))
            ka, kb = sorted([int(iimp[k_imp]), int(ckey[ch])])
            ikeys.append((ka * 1_000_003) ^ kb)
        def arr_(x, shape, dt_=np.float32):
            return np.array(x, dt_).reshape(shape)
        nb = len(live)
        a = {
            "chunk_key": ckey[isl], "chunk_feat": arr_(chunk_feat, (nc, 33)),
            "chunk_mass": arr_(mc, (nc,)), "chunk_inertia_body": arr_(inertia[isl], (nc, 3)), "chunk_q_world": arr_(qw[f, isl], (nc, 4)),
            "chunk_pos": arr_(xc - com, (nc, 3)), "chunk_v": arr_(vc, (nc, 3)), "chunk_w": arr_(wc, (nc, 3)),
            "chunk_anchored": arr_(fixed[isl] | pinned[isl], (nc,), np.int32),
            "impactor_key": np.array(iimp, np.int64), "impactor_feat": arr_(imp_feat, (ni, 16)),
            "impactor_mass": arr_(imp_mass[iimp], (ni,)), "impactor_inertia_body": arr_(imp_inertia[iimp], (ni, 3)),
            "impactor_q_world": arr_(impst[f, iimp, 3:7], (ni, 4)), "impactor_pos": arr_(xi - com, (ni, 3)),
            "impactor_v": arr_(vi, (ni, 3)), "impactor_w": arr_(wi, (ni, 3)),
            "global_feat": arr_(glob, (1, 12)),
            "bond_src": arr_(bsrc, (2 * nb,), np.int32), "bond_dst": arr_(bdst, (2 * nb,), np.int32), "bond_key": bkey[live],
            "bond_feat": arr_(bfeat, (2 * nb, 40)), "bond_frame": arr_(bframe, (nb, 3, 3)),
            "impactor_contact_src": arr_(isrc, (2 * len(crow),), np.int32), "impactor_contact_dst": arr_(idst, (2 * len(crow),), np.int32),
            "impactor_contact_key": np.array(ikeys, np.int64), "impactor_contact_feat": arr_(ifeat, (2 * len(crow), 30)),
            "impactor_contact_frame": arr_(iframe, (len(crow), 3, 3)),
        }
        for t, nf in (("cracked", 22), ("structure_contact", 22)):
            a[f"{t}_src"] = np.zeros(0, np.int32); a[f"{t}_dst"] = np.zeros(0, np.int32)
            a[f"{t}_key"] = np.zeros(0, np.int64); a[f"{t}_feat"] = np.zeros((0, nf), np.float32); a[f"{t}_frame"] = np.zeros((0, 3, 3), np.float32)
        # QS groups
        cqs, iqs, bqs = np.zeros((nc, 6)), np.zeros((ni, 6)), np.zeros((2 * nb, 3))
        if qs is not None:
            cqs[:, 0:3] = np.einsum("nij,nj->ni", RbT, qs["chunk_vel"][f, isl] - vc) / v
            cqs[:, 3:6] = np.einsum("nij,nj->ni", RbT, qs["chunk_angvel"][f, isl] - wc) * L / v
            if ni:
                iqs[:, 0:3] = np.einsum("nij,nj->ni", RiT, qs["impactor"][f, iimp, 7:10] - vi) / v
                iqs[:, 3:6] = np.einsum("nij,nj->ni", RiT, qs["impactor"][f, iimp, 10:13] - wi) * L / v
            for j, k in enumerate(live):
                row = [np.log(max(qs["bond_peak_util"][f, k], TINY)), float(qs["bond_broken"][f, k]), 0.0]
                bqs[2 * j] = row; bqs[2 * j + 1] = row
            a["base_qs_bond_broke"] = arr_(qs["bond_broken"][f, live], (nb,), np.int32)
            a["base_qs_bond_peak_util"] = arr_(qs["bond_peak_util"][f, live], (nb,))
            a["base_qs_chunk_v_end"] = arr_(qs["chunk_vel"][f, isl], (nc, 3))
            a["base_qs_chunk_w_end"] = arr_(qs["chunk_angvel"][f, isl], (nc, 3))
            a["base_qs_impactor_v_end"] = arr_(qs["impactor"][f, iimp, 7:10], (ni, 3))
        if pu is not None and f in punch_frames:
            pi = punch_frames.index(f)
            a["base_punch_bond_broke"] = arr_(pu["bond_broken"][pi, live], (nb,), np.int32)
            a["base_punch_bond_peak_util"] = arr_(pu["bond_peak_util"][pi, live], (nb,))
            a["base_punch_chunk_v_end"] = arr_(pu["chunk_vel"][pi, isl], (nc, 3))
            a["base_punch_chunk_w_end"] = arr_(pu["chunk_angvel"][pi, isl], (nc, 3))
            a["base_punch_impactor_v_end"] = arr_(pu["impactor"][pi, iimp, 7:10], (ni, 3))
        a["chunk_qs_feat"], a["impactor_qs_feat"], a["bond_qs_feat"] = arr_(cqs, (nc, 6)), arr_(iqs, (ni, 6)), arr_(bqs, (2 * nb, 3))
        # targets
        broke = bins[f, live] >= 0
        dd = np.where(broke, 0.0, np.maximum(joint[f + 1, live, 0] - joint[f, live, 0], 0.0))
        bd = dis[f, live].sum(1)
        a.update({
            "y_bond_peak_util": arr_(peak[f, live], (nb,)), "y_bond_broke": arr_(broke, (nb,), np.int32),
            "y_bond_break_bin": arr_(np.where(broke, bins[f, live] + 1, 0), (nb,), np.int32),
            "y_bond_fail_mode": arr_(np.where(broke, bmode[f + 1, live], 0), (nb,), np.int32),
            "y_bond_ddamage": arr_(dd, (nb,)), "y_bond_dissipated": arr_(bd, (nb,)),
            "y_chunk_v_end": arr_(vel[f + 1, isl], (nc, 3)), "y_chunk_w_end": arr_(ang[f + 1, isl], (nc, 3)),
            "y_chunk_dx": arr_(pos[f + 1, isl] - xc, (nc, 3)), "y_chunk_drot": arr_(rotvec_between(qw[f, isl], qw[f + 1, isl]), (nc, 3)),
            "y_chunk_fragment": arr_(np.unique(cluster[f + 1, isl], return_inverse=True)[1], (nc,), np.int32),
            "y_impactor_v_end": arr_(impst[f + 1, iimp, 7:10], (ni, 3)), "y_impactor_w_end": arr_(impst[f + 1, iimp, 10:13], (ni, 3)),
            "y_impactor_dx": arr_(impst[f + 1, iimp, 0:3] - xi, (ni, 3)),
            "y_impactor_drot": arr_(rotvec_between(impst[f, iimp, 3:7], impst[f + 1, iimp, 3:7]) if ni else np.zeros((0, 3)), (ni, 3)),
            "y_impactor_regime": imp_regime[iimp],
        })
        # energy
        vc1, wc1 = vel[f + 1, isl], ang[f + 1, isl]
        Rb1 = qmat(qw[f + 1, isl])
        ke_c1 = 0.5 * mc * (vc1 * vc1).sum(1) + 0.5 * np.einsum("ni,nij,nj->n", wc1, Rb1 @ (inertia[isl][:, :, None] * np.swapaxes(Rb1, 1, 2)), wc1)
        vi1, wi1 = impst[f + 1, iimp, 7:10], impst[f + 1, iimp, 10:13]
        Ri1 = qmat(impst[f + 1, iimp, 3:7]) if ni else np.zeros((0, 3, 3))
        ke_i1 = 0.5 * imp_mass[iimp] * (vi1 * vi1).sum(1) + (0.5 * np.einsum("ni,nij,nj->n", wi1, Ri1 @ (imp_inertia[iimp][:, :, None] * np.swapaxes(Ri1, 1, 2)), wi1) if ni else 0)
        KE_out = float(ke_c1.sum() + np.sum(ke_i1))
        dx_all = np.concatenate([pos[f + 1, isl] - xc, impst[f + 1, iimp, 0:3] - xi])
        W_g = float((mall * (dx_all @ g)).sum())
        el0, el1 = float(benergy[f, live, 2].sum()), float(benergy[f + 1, live, 2].sum())
        D_bond = float(bd.astype(np.float32).astype(np.float64).sum())
        en = {"KE_in": KE_in, "KE_out": KE_out, "W_gravity": W_g, "W_external": 0.0, "E_elastic_start": el0, "E_elastic_end": el1,
              "D_bond": D_bond, "D_bond_joint": float(dis[f, live, 0].sum()), "D_bond_damping": float(dis[f, live, 1].sum()),
              "D_fracture_floor": float(GA_b[live][broke].sum()),
              # frames.json holds the ledger after each frame.
              "D_contact_booked": float(frames_json[f]["energy"]["contact_dissipated"] - (frames_json[f - 1]["energy"]["contact_dissipated"] if f else 0.0))}
        en["ledger_residual"] = (KE_out + el1 + D_bond - KE_in - el0 - W_g) / E_in
        ph = P1_PHASES[int(phase[f])]
        transition = (f > 0 and phase[f] != phase[f - 1]) or bool(broke.any() and (bins[:f, live] < 0).all())
        prev = prev_frame_call.get(f - 1)
        c = schema.Call(event_id=ev.path.name, call_index=len(calls), frame=f, island=0, prev_call=prev, phase=ph,
                        transition=bool(transition), routed=True, anchored=anchored_isl,
                        counts={"chunks": nc, "impactors": ni, "bond": nb, "cracked": 0, "impactor_contact": len(crow), "structure_contact": 0},
                        scales=scales, energy=en, arrays=a)
        prev_frame_call[f] = c.call_index
        calls.append(c)
    ens = ctx.get("ensemble") or {}
    manifest = {
        "event_id": ev.path.name, "recipe_id": ev.path.name, "family": ctx["family"], "layout_family": ctx["layout_group"],
        "split_group": ctx["split_group"], "split": ctx.get("split") or "unassigned",
        "ensemble_id": ens.get("id"), "ensemble_member": ens.get("member"), "ensemble_size": ens.get("size", 1),
        "perturbation": ens.get("perturbation"), "seed": scene["sim"]["seed"],
        "oracle": {"config": ctx["oracle_config"]["name"], "methods": ctx["oracle_config"]["methods"], "contact": "penalty", "hash": ctx["scene_hash"]},
        "scene_hash": ctx["scene_hash"], "recorder_revision": f"p1-rev{rev}", "builder": "outcome_calls_smoke (Python; NOT the Rust builder)",
        "frame_dt": dt, "gravity": list(map(float, g)), "truncated": bool(o["truncated"]),
        "available": {"qs_hertz": True, "punch": True, "qs_frame": qs is not None, "punch_frame": bool(punch_frames), "blast": False,
                      "cracked_edges": False, "structure_contacts": False, "bond_energy_split": True},
        "outcome": {"impactors": [{"key": k, "v_exit": x["v_end"], "speed_exit": float(np.linalg.norm(x["v_end"])), "regime": x["regime"]} for k, x in enumerate(o["impactors"])],
                    "n_fragments": o["fragments"], "largest_fraction": o["largest_fraction"], "sim_time": o["sim_seconds"], "frames": o["frames"]},
    }
    out = out_root / ev.path.name
    schema.write_event(out, manifest, calls, structures)
    return out, len(calls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--p2-python", type=Path, required=True)
    ap.add_argument("--family", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    # P2's schema.py by path (its package shares this one's name); it needs only numpy.
    import importlib.util
    spec = importlib.util.spec_from_file_location("p2_outcome_schema", a.p2_python / "learned_impact" / "outcome" / "schema.py")
    schema = importlib.util.module_from_spec(spec)
    sys.modules["p2_outcome_schema"] = schema
    spec.loader.exec_module(schema)
    n = ok = 0
    for run in a.runs:
        for p in event_dirs(run):
            ev = load_event(p)
            if a.family and ev.context["family"] != a.family:
                continue
            if a.limit and ok >= a.limit:
                break
            n += 1
            try:
                out, k = convert_event(ev, schema, a.out)
                schema.load_event(out)  # validate
                ok += 1
                print(f"[calls] {ok} {p.name}: {k} calls", flush=True)
            except Exception as e:  # fail loud, per event
                print(f"[calls] FAILED {p.name}: {type(e).__name__}: {e}", flush=True)
    print(f"[calls] {ok}/{n} events converted and validated")


if __name__ == "__main__":
    main()
