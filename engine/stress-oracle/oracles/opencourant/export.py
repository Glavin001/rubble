"""Exporter: stress-scene/1 JSON -> OpenCourant (OpenRadioss) starter + engine decks.

Reads ONLY the scene file (plus CLI overrides) and writes
    <out>/<scene>_0000.rad   starter deck
    <out>/<scene>_0001.rad   engine deck
    <out>/map.json           node/element bookkeeping for observe.py

Model (see README.md for the justification):
  * every chunk is an independent block of 8-node hexahedra (/PROP/SOLID Isolid=24),
    linear elastic (/MAT/LAW1) with the chunk material's E, nu, rho;
  * every scene bond becomes a layer of 8-node cohesive elements (/PROP/TYPE43) with
    /MAT/LAW169 (ARUP adhesive): tension cutoff TENMAX = f_t * weibull, Mohr-Coulomb shear
    SHRMAX + SHT_SL * compression with SHRMAX = c * weibull and SHT_SL = friction, linear
    softening to zero with the fracture energies G_f (tension, shear).  LAW169 needs a
    finite geometric thickness, so the faces of a chunk that carry a bond are pulled in by
    t_c / 2 and the cohesive layer (thickness t_c, density of the chunks) fills the gap;
    its modulus is chosen so that the series compliance of chunk + layer equals the scene's
    bond compliance L / E_bond;
  * chunk faces without bonds are not connected; a node-to-surface contact (/INTER/TYPE7)
    with gap t_c acts between all chunks of a body (cracked joints close at their original
    position), and separate TYPE7 interfaces couple different bodies and impactors;
  * rigid impactors are /RBODY with the scene mass/inertia, meshed as a box of hexahedra
    (removed from the element loop) that provides the contact surface;
  * fixed chunks: all nodes fixed (/BCS 111 111); gravity /GRAV on all nodes (applied at
    t = 0, no prestress phase); point forces are distributed over the chunk's nodes like a
    rigid-body load (mass-proportional + moment); pressures act on exposed chunk faces.

Units in the deck: g, cm, s (CGS; see the comment at the unit constants).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict

# ----------------------------------------------------------------------------- units
# Deck units: g, cm, s (CGS).  LAW169 adds `E * area` to the nodal stiffness and checks
# G_c >= sigma^2 / E, i.e. it implicitly assumes a layer thickness of one length unit;
# with centimetres that matches the cohesive layer thickness (~1 cm for 10 cm chunks).
L = 1e2  # m -> cm
M = 1e3  # kg -> g
RHO = M / L ** 3  # kg/m^3 -> g/cm^3
F = 1e5  # N -> dyn
P = F / L ** 2  # Pa -> dyn/cm^2
GF = F / L  # J/m^2 -> erg/cm^2 = dyn/cm
INERTIA = M * L * L  # kg m^2 -> g cm^2
VEL = L  # m/s -> cm/s

ORACLE = "opencourant"


# ----------------------------------------------------------------------------- helpers

def quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def is_identity(q, tol=1e-12):
    return abs(abs(q[0]) - 1.0) < tol and all(abs(c) < tol for c in q[1:])


def mat_vec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def to_world(body, p):
    r = quat_to_mat(body.get("orientation", [1, 0, 0, 0]))
    pv = mat_vec(r, p)
    pos = body.get("position", [0, 0, 0])
    return [pv[i] + pos[i] for i in range(3)]


def fnum(x):
    """20-char float field."""
    return f"{x:20.12g}"


def inum(i):
    return f"{int(i):10d}"


def eval_tf(tf, t):
    k = tf["type"]
    if k == "constant":
        return tf["value"]
    if k == "ramp":
        if t <= tf["t0"]:
            return 0.0
        if t >= tf["t1"]:
            return tf["value"]
        return tf["value"] * (t - tf["t0"]) / (tf["t1"] - tf["t0"])
    if k == "half_sine":
        if t < tf["start"] or t > tf["start"] + tf["duration"]:
            return 0.0
        return tf["peak"] * math.sin(math.pi * (t - tf["start"]) / tf["duration"])
    if k == "friedlander":
        s = (t - tf["arrival"]) / tf["duration"]
        if s < 0 or s > 1:
            return 0.0
        return tf["peak"] * (1 - s) * math.exp(-tf["decay"] * s)
    if k == "table":
        pts = tf["points"]
        if not pts:
            return 0.0
        if t <= pts[0][0]:
            return pts[0][1]
        for a, b in zip(pts, pts[1:]):
            if t <= b[0]:
                f = (t - a[0]) / max(b[0] - a[0], 1e-300)
                return a[1] + f * (b[1] - a[1])
        return pts[-1][1]
    raise ValueError(f"unknown time function {k}")


def tf_points(tf, t_end, n_smooth=200):
    """Piecewise-linear samples of a TimeFunction, flat beyond the ends."""
    k = tf["type"]
    far = 10.0 * max(t_end, 1.0) + 1.0
    if k == "constant":
        pts = [(0.0, tf["value"]), (far, tf["value"])]
    elif k == "ramp":
        pts = [(0.0, 0.0), (tf["t0"], 0.0), (tf["t1"], tf["value"]), (far, tf["value"])]
    elif k == "table":
        pts = [(p[0], p[1]) for p in tf["points"]]
        if pts[0][0] > 0.0:
            pts.insert(0, (0.0, pts[0][1]))
        pts.append((far, pts[-1][1]))
    else:
        if k == "half_sine":
            a, d = tf["start"], tf["duration"]
        elif k == "friedlander":
            a, d = tf["arrival"], tf["duration"]
        else:
            raise ValueError(k)
        pts = [(0.0, 0.0)] if a > 0 else []
        for i in range(n_smooth + 1):
            t = a + d * i / n_smooth
            pts.append((t, eval_tf(tf, t)))
        # step back to zero just after the pulse (friedlander ends at 0 anyway)
        pts.append((a + d * (1 + 1e-6), 0.0))
        pts.append((far, 0.0))
    # strictly increasing abscissae
    out = []
    for t, v in pts:
        if out and t <= out[-1][0]:
            t = out[-1][0] + 1e-12 * max(1.0, abs(t))
        out.append((t, v))
    return out


# ----------------------------------------------------------------------------- model

class Model:
    def __init__(self):
        self.nodes = []  # mm
        self.hex_by_part = defaultdict(list)  # part id -> [(eid, n1..n8)]
        self.next_eid = 1
        self.parts = OrderedDict()  # pid -> dict(title, prop, mat)
        self.mats = OrderedDict()  # mid -> text
        self.props = OrderedDict()  # pid -> text
        self.groups = []  # text blocks (GRNOD, GRBRIC, SURF)
        self.loads = []  # text blocks
        self.functs = []  # text blocks
        self.inter = []
        self.misc = []  # RBODY, BCS, INIVEL, SECT, TH
        self.next_id = defaultdict(lambda: 1)

    def nid(self):
        return len(self.nodes)

    def add_node(self, xyz_m):
        self.nodes.append([c * L for c in xyz_m])
        return len(self.nodes)

    def new(self, kind):
        i = self.next_id[kind]
        self.next_id[kind] += 1
        return i

    def add_hex(self, pid, conn):
        eid = self.next_eid
        self.next_eid += 1
        self.hex_by_part[pid].append((eid, conn))
        return eid


def grnod_block(gid, title, ids):
    lines = [f"/GRNOD/NODE/{gid}", title[:100]]
    ids = list(ids)
    for i in range(0, len(ids), 10):
        lines.append("".join(inum(x) for x in ids[i:i + 10]))
    return "\n".join(lines)


def grbric_block(gid, title, ids):
    lines = [f"/GRBRIC/BRIC/{gid}", title[:100]]
    ids = list(ids)
    for i in range(0, len(ids), 10):
        lines.append("".join(inum(x) for x in ids[i:i + 10]))
    return "\n".join(lines)


def surf_block(sid, title, segs):
    lines = [f"/SURF/SEG/{sid}", title[:100]]
    for k, s in enumerate(segs, 1):
        lines.append(inum(k) + "".join(inum(n) for n in s))
    return "\n".join(lines)


def funct_block(fid, title, pts, yscale=1.0):
    lines = [f"/FUNCT/{fid}", title[:100]]
    for t, v in pts:
        lines.append(fnum(t) + fnum(v * yscale))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- exporter

def options_from(scene, cli):
    hints = scene.get("oracle", {}).get(ORACLE, {}) or {}
    o = {
        "elements_per_chunk": 2,
        "cohesive_thickness": 0.2,  # t_c as a fraction of the element size
        "self_contact": True,
        "contact_gap_fraction": 0.5,  # self-contact gap / t_c (contact only once a joint has closed by t_c/2)
        "impact_gap": 2e-4,  # m, gap of body-body / impactor contacts
        "dt_safety": 0.6,  # on the cohesive-layer stability estimate
        "threads": 1,
        "anim_frames": 10,
        "th_dt": None,  # default: half the scene sample interval
        "isolid": 24,
        "body_contact_stfac": 1.0,  # TYPE7 Stfac of body-body contacts
        "contact_damping": 0.05,  # TYPE7 VIS_S (fraction of critical damping, the Radioss default)
        # Chunk solids crush (/MAT/LAW2, J2 plasticity at the compressive strength, eroded
        # at the plastic strain that dissipates G_c per area over one element):
        # off reproduces the committed goldens (elastic chunks, /MAT/LAW1).
        "chunk_crushing": False,
        # Chunk solids as concrete (/MAT/LAW124, CDPM2: damage-plasticity with tensile and
        # compressive softening regularised by the fracture energies): off by default.
        "chunk_cdpm2": False,
    }
    for k, v in hints.items():
        if k in o:
            o[k] = v
    for k, v in (cli or {}).items():
        if v is not None:
            o[k] = v
    return o


def export(scene, outdir, opts):
    m = Model()
    sim = scene["sim"]
    duration = sim["duration"]
    mats = scene["materials"]
    scale = sim.get("stiffness_scale", 1.0)
    notes = []
    if scene.get("events"):
        raise NotImplementedError("scripted events (remove_chunks/remove_supports) are not exported")
    if scene.get("ground"):
        raise NotImplementedError("ground plane not exported yet")
    if scale != 1.0:
        notes.append(f"stiffness_scale {scale} applied to every Young's modulus")

    K = int(opts["elements_per_chunk"])
    tc_frac = float(opts["cohesive_thickness"])
    meta = {"units": {"length_to_m": 1 / L, "mass_to_kg": 1 / M, "force_to_N": 1 / F, "energy_to_J": 1 / (F * L),
                      "stress_to_Pa": 1 / P},
            "options": opts, "bodies": {}, "impactors": {}, "probes": {}, "th": [], "notes": notes}

    # Material / part bookkeeping
    solid_prop = m.new("prop")
    m.props[solid_prop] = "\n".join([
        f"/PROP/SOLID/{solid_prop}", "chunk solids",
        "#   Isolid    Ismstr      Iale     Icpre  Itetra10     Inpts   Itetra4    Iframe                  Dn",
        f"{int(opts['isolid']):10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0:10d}{0.0:20.6g}",
        "#                 qa                  qb                   h              Lambda                  Mu",
        f"{0.0:20.6g}{0.0:20.6g}{0.0:20.6g}{0.0:20.6g}{0.0:20.6g}",
        "#         deltaT_min            vdef_min            vdef_max             ASP_max             COL_min",
        f"{0.0:20.6g}{0.0:20.6g}{0.0:20.6g}{0.0:20.6g}{0.0:20.6g}",
        "#     Ndir sphpartID  Icontrol",
        f"{0:10d}{0:10d}{0:10d}",
    ])
    coh_prop = m.new("prop")
    m.props[coh_prop] = "\n".join([
        f"/PROP/TYPE43/{coh_prop}", "cohesive layers",
        "#   Ismstr                                                                            True_thickness",
        f"{0:10d}{'':70s}{0.0:20.6g}",
    ])
    elastic_mat = {}

    def solid_mat(name, element_size):
        key = (name, element_size if (opts.get("chunk_crushing") or opts.get("chunk_cdpm2")) else None)
        if key in elastic_mat:
            return elastic_mat[key]
        mt = mats[name]
        mid = m.new("mat")
        if opts.get("chunk_cdpm2"):
            # CDPM2 (Grassl et al. 2013): the scene's E, nu, f_t, f_c; tensile softening
            # linear to the crack opening 2 G_f / f_t; compressive softening at the
            # inelastic strain G_c / (f_c h) (the compressive fracture energy over one
            # element, as the reference's crushing); shape, hardening and damage
            # constants at the law's defaults (zeros); damaged elements deleted.
            ft, fc = mt["tensile_strength"], mt["compressive_strength"]
            wf = 2.0 * mt["fracture_energy"]["tension"] / ft
            efc = mt["fracture_energy"]["compression"] / (fc * element_size)
            m.mats[mid] = "\n".join([
                f"/MAT/LAW124/{mid}", f"{name} (concrete, CDPM2)",
                "#              RHO_I", fnum(mt["density"] * RHO),
                "#                  E                  NU                IDEL               IRATE                FCUT",
                fnum(mt["youngs_modulus"] * scale * P) + fnum(mt["poisson_ratio"]) + f"{'':10s}{1:10d}{'':10s}{0:10d}" + fnum(0.0),
                "#                ECC                 QH0                  FT                  FC                  HP",
                fnum(0.0) + fnum(0.0) + fnum(ft * P) + fnum(fc * P) + fnum(0.0),
                "#                 AH                  BH                  CH                  DH",
                fnum(0.0) + fnum(0.0) + fnum(0.0) + fnum(0.0),
                "#                 AS                  BS                  DF               DFLAG     DTYPE      IREG",
                fnum(0.0) + fnum(0.0) + fnum(0.0) + f"{'':10s}{0:10d}{0:10d}{0:10d}",
                "#                 WF                 WF1                 FT1                 EFC",
                fnum(wf * L) + fnum(0.0) + fnum(0.0) + fnum(efc),
            ])
            notes.append(f"{name}: chunk solids CDPM2 (f_t {ft:g}, f_c {fc:g} Pa, w_f {wf:.3g} m, e_fc {efc:.4g}, h {element_size:g} m)")
        elif opts.get("chunk_crushing"):
            # Crushing: von Mises plasticity at the uniaxial compressive strength (no
            # hardening, no rate effect), the element eroded once its plastic strain
            # reaches G_c / (f_c h): a crushed band one element thick has then dissipated
            # the compressive fracture energy per area, as the reference's crushing does.
            fc = mt["compressive_strength"]
            gc = mt["fracture_energy"]["compression"]
            eps_max = gc / (fc * element_size)
            m.mats[mid] = "\n".join([
                f"/MAT/LAW2/{mid}", f"{name} (crushing at f_c)",
                "#              RHO_I", fnum(mt["density"] * RHO),
                "#                  E                  Nu     Iflag    flagVP                Pmin",
                fnum(mt["youngs_modulus"] * scale * P) + fnum(mt["poisson_ratio"]) + f"{0:10d}{0:10d}" + fnum(0.0),
                "#                  a                   b                   n           EPS_p_max            SIG_max0",
                fnum(fc * P) + fnum(0.0) + fnum(1.0) + fnum(eps_max) + fnum(0.0),
                "#                  c           EPS_DOT_0       ICC   Fsmooth               F_cut               Chard",
                fnum(0.0) + fnum(0.0) + f"{0:10d}{0:10d}" + fnum(0.0) + fnum(0.0),
                "#                  m              T_melt              rhoC_p                 T_r               T_max",
                fnum(0.0) + fnum(0.0) + fnum(0.0) + fnum(0.0) + fnum(0.0),
            ])
            notes.append(f"{name}: chunk solids crush at f_c = {fc:g} Pa (J2), eroded at plastic strain {eps_max:.4g} (G_c / (f_c h), h = {element_size:g} m)")
        else:
            m.mats[mid] = "\n".join([
                f"/MAT/LAW1/{mid}", f"{name} (elastic)",
                "#              RHO_I", fnum(mt["density"] * RHO),
                "#                  E                  nu",
                fnum(mt["youngs_modulus"] * scale * P) + fnum(mt["poisson_ratio"]),
            ])
        pid = m.new("part")
        m.parts[pid] = {"title": f"chunks {name}", "prop": solid_prop, "mat": mid}
        elastic_mat[key] = pid
        return pid

    coh_parts = {}

    def coh_part(key):
        if key in coh_parts:
            return coh_parts[key]
        rho, e, nu, sht, ten, gct, shr, gcs = key
        mid = m.new("mat")
        m.mats[mid] = "\n".join([
            f"/MAT/LAW169/{mid}", f"cohesive {len(coh_parts) + 1}",
            "#              Rho_I", fnum(rho),
            "#                  E                  PR              SHT_SL              TENMAX               GCTEN",
            fnum(e) + fnum(nu) + fnum(sht) + fnum(ten) + fnum(gct),
            "#             SHRMAX               GCSHR      PWRT      PWRS                SHRP",
            fnum(shr) + fnum(gcs) + f"{2:10d}{2:10d}" + fnum(0.0),
        ])
        pid = m.new("part")
        m.parts[pid] = {"title": f"cohesive {len(coh_parts) + 1}", "prop": coh_prop, "mat": mid}
        coh_parts[key] = pid
        return pid

    # ---------------------------------------------------------------- bodies
    body_surfaces = {}  # body name -> list of segments (all chunk faces)
    body_nodes = {}
    min_tc = float("inf")
    dt_limit = float("inf")
    fixed_nodes = []
    hex_dt = float("inf")
    for bi, body in enumerate(scene["bodies"]):
        bname = body["name"]
        chunks = body["chunks"]
        bonds = body["bonds"]
        if not is_identity(body.get("orientation", [1, 0, 0, 0])) and False:
            pass
        for c in chunks:
            if not is_identity(c.get("orientation", [1, 0, 0, 0])):
                raise NotImplementedError("rotated chunks are not supported by the exporter")
            if c.get("level", 0) != 0:
                raise NotImplementedError("pre-fractured (level > 0) chunks are not exported")
        min_edge = min(2 * h for c in chunks for h in c["half_extents"])
        # element size per axis: K elements along the body's smallest chunk extent on that
        # axis (conforming grids across bonds; flat chunks get flat elements)
        he_ax = [min(2 * c["half_extents"][a] for c in chunks) / K for a in range(3)]
        he = min(he_ax)
        tc = tc_frac * he
        min_tc = min(min_tc, tc)
        # bonded faces (chunk, axis, side)
        bond_axis = []
        bonded = set()
        fixed_pair = set()
        for b in bonds:
            n = b["normal"]
            ax = max(range(3), key=lambda i: abs(n[i]))
            if abs(abs(n[ax]) - 1.0) > 1e-9:
                raise NotImplementedError("non axis-aligned bond normals are not supported")
            s = 1 if n[ax] > 0 else -1
            bond_axis.append((ax, s))
            bonded.add((b["a"], ax, s))
            bonded.add((b["b"], ax, -s))
            if chunks[b["a"]].get("support", "none") != "none" and chunks[b["b"]].get("support", "none") != "none":
                fixed_pair.add((b["a"], ax, s))
                fixed_pair.add((b["b"], ax, -s))
        cinfo = []
        all_nodes = []
        segs = []
        for ci, c in enumerate(chunks):
            ctr, h = c["center"], c["half_extents"]
            pid = solid_mat(c["material"], min(he_ax))
            n = [max(1, int(round(2 * h[a] / he_ax[a]))) for a in range(3)]
            lo = [ctr[a] - h[a] for a in range(3)]
            hi = [ctr[a] + h[a] for a in range(3)]
            dlo = [tc / 2 if (ci, a, -1) in bonded else 0.0 for a in range(3)]
            dhi = [tc / 2 if (ci, a, 1) in bonded else 0.0 for a in range(3)]
            nominal = [[lo[a] + (hi[a] - lo[a]) * i / n[a] for i in range(n[a] + 1)] for a in range(3)]
            actual = [[(lo[a] + dlo[a]) + (hi[a] - dhi[a] - lo[a] - dlo[a]) * i / n[a] for i in range(n[a] + 1)]
                      for a in range(3)]
            ids = {}
            for k in range(n[2] + 1):
                for j in range(n[1] + 1):
                    for i in range(n[0] + 1):
                        ids[(i, j, k)] = m.add_node(to_world(body, [actual[0][i], actual[1][j], actual[2][k]]))
            weight = defaultdict(int)
            hexes = []
            for k in range(n[2]):
                for j in range(n[1]):
                    for i in range(n[0]):
                        conn = [ids[(i, j, k)], ids[(i + 1, j, k)], ids[(i + 1, j + 1, k)], ids[(i, j + 1, k)],
                                ids[(i, j, k + 1)], ids[(i + 1, j, k + 1)], ids[(i + 1, j + 1, k + 1)],
                                ids[(i, j + 1, k + 1)]]
                        hexes.append(m.add_hex(pid, conn))
                        for nd in conn:
                            weight[nd] += 1
            # boundary faces as outward segments
            for ax in range(3):
                t1, t2 = (ax + 1) % 3, (ax + 2) % 3
                for side in (-1, 1):
                    fixed_idx = 0 if side < 0 else n[ax]
                    for u in range(n[t1]):
                        for v in range(n[t2]):
                            def nd(uu, vv):
                                idx = [0, 0, 0]
                                idx[ax], idx[t1], idx[t2] = fixed_idx, uu, vv
                                return ids[tuple(idx)]
                            quad = [nd(u, v), nd(u + 1, v), nd(u + 1, v + 1), nd(u, v + 1)]  # normal +ax
                            if side < 0:
                                quad = quad[::-1]
                            segs.append({"chunk": ci, "axis": ax, "side": side, "nodes": quad,
                                         "center": [0.5 * (nominal[t1][u] + nominal[t1][u + 1]),
                                                    0.5 * (nominal[t2][v] + nominal[t2][v + 1])]})
            mt = mats[c["material"]]
            vol_e = 1.0
            for a in range(3):
                vol_e *= (hi[a] - dhi[a] - lo[a] - dlo[a]) / n[a]
            emin = min((hi[a] - dhi[a] - lo[a] - dlo[a]) / n[a] for a in range(3))
            e = mt["youngs_modulus"] * scale
            nu = mt["poisson_ratio"]
            cm = math.sqrt(e * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) / mt["density"])
            hex_dt = min(hex_dt, emin / cm)
            cinfo.append({"ids": ids, "n": n, "nominal": nominal, "weights": dict(weight), "hexes": hexes,
                          "node_mass": mt["density"] * vol_e / 8.0, "emin": emin, "cm": cm})
            all_nodes.extend(ids.values())
            if c.get("support", "none") in ("fixed", "pinned"):
                fixed_nodes.extend(ids.values())
        body_nodes[bname] = all_nodes
        body_surfaces[bname] = [s["nodes"] for s in segs]

        # ------------------------------------------------ cohesive layers
        bond_elems = []
        for bi2, (b, (ax, s)) in enumerate(zip(bonds, bond_axis)):
            a, bb = b["a"], b["b"]
            ca, cb = chunks[a], chunks[bb]
            t1, t2 = (ax + 1) % 3, (ax + 2) % 3
            plo = [max(ca["center"][o] - ca["half_extents"][o], cb["center"][o] - cb["half_extents"][o]) for o in (t1, t2)]
            phi = [min(ca["center"][o] + ca["half_extents"][o], cb["center"][o] + cb["half_extents"][o]) for o in (t1, t2)]
            ia, ib = cinfo[a], cinfo[bb]
            fa = 0 if s < 0 else ia["n"][ax]  # face index of a facing b
            fb = ib["n"][ax] if s < 0 else 0
            na, nb = ia["nominal"], ib["nominal"]
            tol = 1e-9 * max(1.0, min_edge)

            def find(arr, x):
                for i, y in enumerate(arr):
                    if abs(x - y) <= tol:
                        return i
                return None

            # bond material and layer modulus (series compliance)
            bm = mats[b["material"]]
            if (b.get("derived") or {}).get("weibull") is None:
                if bm.get("weibull_modulus") is not None:
                    raise RuntimeError("bond without derived.weibull in a Weibull scene (regenerate the scene)")
            w = (b.get("derived") or {}).get("weibull", 1.0)
            length = abs(ca["center"][ax] - cb["center"][ax])
            e_bond = bm["youngs_modulus"] * scale
            ea = mats[ca["material"]]["youngs_modulus"] * scale
            eb = mats[cb["material"]]["youngs_modulus"] * scale
            chunk_compliance = (ca["half_extents"][ax] - tc / 2) / ea + (cb["half_extents"][ax] - tc / 2) / eb
            layer_compliance = length / e_bond - chunk_compliance
            if layer_compliance <= tc / (20.0 * max(ea, eb)):
                layer_compliance = tc / (20.0 * max(ea, eb))
                notes.append(f"body {bname} bond {bi2}: bond stiffer than chunks; layer modulus capped at 20 E")
            e_layer = tc / layer_compliance
            nu_layer = bm["poisson_ratio"]
            rho_layer = 0.5 * (mats[ca["material"]]["density"] + mats[cb["material"]]["density"])
            fric = bm["friction"]
            key = (round(rho_layer * RHO, 18), float(f"{e_layer * P:.10g}"), nu_layer, fric,
                   float(f"{bm['tensile_strength'] * w * P:.10g}"), bm["fracture_energy"]["tension"] * GF,
                   float(f"{bm['cohesion'] * w * P:.10g}"), bm["fracture_energy"]["shear"] * GF)
            if bm.get("kind", "brittle") == "ductile":
                notes.append(f"bond material '{b['material']}' is ductile (plateau then snap); LAW169 softens "
                             "linearly with the same fracture energies")
            if bm.get("shear_cap") is not None:
                notes.append("shear_cap is not represented by LAW169 (Mohr-Coulomb shear is uncapped)")
            pid = coh_part(key)
            elems = []
            for u in range(len(na[t1]) - 1):
                cu = 0.5 * (na[t1][u] + na[t1][u + 1])
                if not (plo[0] - tol <= cu <= phi[0] + tol):
                    continue
                for v in range(len(na[t2]) - 1):
                    cv = 0.5 * (na[t2][v] + na[t2][v + 1])
                    if not (plo[1] - tol <= cv <= phi[1] + tol):
                        continue
                    ub0, ub1 = find(nb[t1], na[t1][u]), find(nb[t1], na[t1][u + 1])
                    vb0, vb1 = find(nb[t2], na[t2][v]), find(nb[t2], na[t2][v + 1])
                    if None in (ub0, ub1, vb0, vb1):
                        raise RuntimeError(f"body {bname} bond {bi2}: non-conforming chunk meshes")

                    def na_(uu, vv):
                        idx = [0, 0, 0]
                        idx[ax], idx[t1], idx[t2] = fa, uu, vv
                        return ia["ids"][tuple(idx)]

                    def nb_(uu, vv):
                        idx = [0, 0, 0]
                        idx[ax], idx[t1], idx[t2] = fb, uu, vv
                        return ib["ids"][tuple(idx)]

                    bot = [(u, v, ub0, vb0), (u + 1, v, ub1, vb0), (u + 1, v + 1, ub1, vb1), (u, v + 1, ub0, vb1)]
                    if s < 0:
                        bot = bot[::-1]
                    conn = [na_(p[0], p[1]) for p in bot] + [nb_(p[2], p[3]) for p in bot]
                    elems.append(m.add_hex(pid, conn))
            if not elems:
                raise RuntimeError(f"body {bname} bond {bi2}: no cohesive elements generated")
            bond_elems.append(elems)
            # stability of the layer (two-node oscillator on the lightest face node)
            mn = min(ia["node_mass"], ib["node_mass"])
            area_e = (phi[0] - plo[0]) * (phi[1] - plo[1]) / len(elems)
            m_layer = e_layer * (1 - nu_layer) / ((1 + nu_layer) * (1 - 2 * nu_layer))
            k = m_layer * area_e / (4.0 * tc)
            w2 = 2.0 * k / mn + (2.0 * max(ia["cm"], ib["cm"]) / min(ia["emin"], ib["emin"])) ** 2
            dt_limit = min(dt_limit, 2.0 / math.sqrt(w2))

        meta["bodies"][bname] = {
            "chunks": [{"nodes": [int(x) for x in ci_["ids"].values()],
                        "weights": [int(ci_["weights"].get(x, 0)) for x in ci_["ids"].values()],
                        "hexes": ci_["hexes"]} for ci_ in cinfo],
            "bonds": bond_elems,
            "element_size": he, "cohesive_thickness": tc,
            "supported": [i for i, c in enumerate(chunks) if c.get("support", "none") != "none"],
        }
        cinfo_all = meta.setdefault("_cinfo", {})
        cinfo_all[bname] = cinfo
        body["_segs"] = segs
        body["_fixed_pair"] = fixed_pair

    # ---------------------------------------------------------------- impactors
    rb_texts = []
    rigid_part = None
    imp_surfaces = {}
    for imp in scene.get("impactors", []):
        shp = imp["shape"]
        if shp["type"] not in ("box", "sphere"):
            raise NotImplementedError(f"impactor shape {shp['type']}")
        sphere = shp["type"] == "sphere"
        if not is_identity(imp.get("orientation", [1, 0, 0, 0])):
            raise NotImplementedError("rotated impactors not supported")
        if imp.get("crush"):
            raise NotImplementedError("crushable impactors not supported")
        if rigid_part is None:
            mid = m.new("mat")
            steel = mats[imp["material"]]
            # nominal density: the rigid body's mass/inertia are set on the main node
            m.mats[mid] = "\n".join([
                f"/MAT/LAW1/{mid}", "impactor (rigid, nominal)",
                "#              RHO_I", fnum(steel["density"] * RHO * 1e-6),
                "#                  E                  nu",
                fnum(steel["youngs_modulus"] * P) + fnum(steel["poisson_ratio"]),
            ])
            rigid_part = m.new("part")
            m.parts[rigid_part] = {"title": "impactors (rigid)", "prop": solid_prop, "mat": mid}
        he_imp = min(meta["bodies"][b]["element_size"] for b in meta["bodies"]) if meta["bodies"] else 0.05
        hh = [shp["radius"]] * 3 if sphere else shp["half_extents"]
        n = [max(4 if sphere else 1, int(round(2 * hh[a] / he_imp))) for a in range(3)]
        p0 = imp["position"]
        ids = {}
        for k in range(n[2] + 1):
            for j in range(n[1] + 1):
                for i in range(n[0] + 1):
                    q = [-hh[0] + 2 * hh[0] * i / n[0], -hh[1] + 2 * hh[1] * j / n[1], -hh[2] + 2 * hh[2] * k / n[2]]
                    if sphere:
                        # cube -> ball map (radial scaling by |q|_inf / |q|_2): the outer
                        # faces land on the sphere; inner hexes are only a rigid filler
                        n2 = math.sqrt(sum(x * x for x in q))
                        ninf = max(abs(x) for x in q)
                        if n2 > 0:
                            q = [x * ninf / n2 for x in q]
                    ids[(i, j, k)] = m.add_node([p0[a] + q[a] for a in range(3)])
        for k in range(n[2]):
            for j in range(n[1]):
                for i in range(n[0]):
                    m.add_hex(rigid_part, [ids[(i, j, k)], ids[(i + 1, j, k)], ids[(i + 1, j + 1, k)], ids[(i, j + 1, k)],
                                           ids[(i, j, k + 1)], ids[(i + 1, j, k + 1)], ids[(i + 1, j + 1, k + 1)],
                                           ids[(i, j + 1, k + 1)]])
        segs = []
        for ax in range(3):
            t1, t2 = (ax + 1) % 3, (ax + 2) % 3
            for side in (-1, 1):
                fi = 0 if side < 0 else n[ax]
                for u in range(n[t1]):
                    for v in range(n[t2]):
                        def nd(uu, vv):
                            idx = [0, 0, 0]
                            idx[ax], idx[t1], idx[t2] = fi, uu, vv
                            return ids[tuple(idx)]
                        q = [nd(u, v), nd(u + 1, v), nd(u + 1, v + 1), nd(u, v + 1)]
                        segs.append(q[::-1] if side < 0 else q)
        main = m.add_node(p0)
        mass = imp["mass"]
        if sphere:
            jxx = jyy = jzz = 0.4 * mass * hh[0] ** 2
        else:
            a2 = [(2 * hh[i]) ** 2 for i in range(3)]
            jxx = mass / 12 * (a2[1] + a2[2])
            jyy = mass / 12 * (a2[0] + a2[2])
            jzz = mass / 12 * (a2[0] + a2[1])
        dep = m.new("grnod")
        dep_nodes = list(ids.values())
        m.groups.append(grnod_block(dep, f"impactor {imp['name']} nodes", dep_nodes))
        rid = m.new("rbody")
        rb_texts.append("\n".join([
            f"/RBODY/{rid}", f"impactor {imp['name']}",
            "#  node_ID   sens_ID   Skew_ID    Ispher                Mass   grnd_ID     Ikrem      ICoG   surf_ID",
            f"{main:10d}{0:10d}{0:10d}{0:10d}" + fnum(mass * M) + f"{dep:10d}{1:10d}{3:10d}{0:10d}",
            "#                Jxx                 Jyy                 Jzz",
            fnum(jxx * INERTIA) + fnum(jyy * INERTIA) + fnum(jzz * INERTIA),
            "#                Jxy                 Jyz                 Jxz",
            fnum(0.0) + fnum(0.0) + fnum(0.0),
            "#  Ioptoff   Iexpams     Ifail",
            f"{0:10d}{0:10d}{0:10d}",
        ]))
        all_imp = dep_nodes + [main]
        g = m.new("grnod")
        m.groups.append(grnod_block(g, f"impactor {imp['name']} all", all_imp))
        v = imp["velocity"]
        if any(abs(x) > 0 for x in v):
            iv = m.new("inivel")
            m.misc.append("\n".join([f"/INIVEL/TRA/{iv}", f"impactor {imp['name']}",
                                     "#                 Vx                  Vy                  Vz   Gnod_id   Skew_id",
                                     fnum(v[0] * VEL) + fnum(v[1] * VEL) + fnum(v[2] * VEL) + f"{g:10d}{0:10d}",
                                     "#             tstart   sens_ID", fnum(0.0) + f"{0:10d}"]))
        if any(abs(x) > 0 for x in imp.get("angular_velocity", [0, 0, 0])):
            raise NotImplementedError("impactor angular velocity")
        meta["impactors"][imp["name"]] = {"rbody": rid, "main_node": main, "position": p0, "mass": mass}
        imp_surfaces[imp["name"]] = {"segs": segs, "nodes": dep_nodes, "material": imp["material"]}

    # ---------------------------------------------------------------- body initial velocities
    for body in scene["bodies"]:
        v = body.get("linear_velocity", [0, 0, 0])
        if any(abs(x) > 0 for x in body.get("angular_velocity", [0, 0, 0])):
            raise NotImplementedError("body angular velocity")
        if any(abs(x) > 0 for x in v):
            g = m.new("grnod")
            m.groups.append(grnod_block(g, f"body {body['name']} nodes", body_nodes[body["name"]]))
            iv = m.new("inivel")
            m.misc.append("\n".join([f"/INIVEL/TRA/{iv}", f"body {body['name']}",
                                     "#                 Vx                  Vy                  Vz   Gnod_id   Skew_id",
                                     fnum(v[0] * VEL) + fnum(v[1] * VEL) + fnum(v[2] * VEL) + f"{g:10d}{0:10d}",
                                     "#             tstart   sens_ID", fnum(0.0) + f"{0:10d}"]))

    # ---------------------------------------------------------------- supports
    if fixed_nodes:
        g = m.new("grnod")
        m.groups.append(grnod_block(g, "fixed chunks", fixed_nodes))
        bid = m.new("bcs")
        m.misc.append("\n".join([f"/BCS/{bid}", "fixed chunks", "#  Tra rot   skew_ID  grnod_ID",
                                 f"   111 111{0:10d}{g:10d}"]))
    for body in scene["bodies"]:
        if any(c.get("support") == "pinned" for c in body["chunks"]):
            notes.append("pinned chunks are held like fixed ones (continuum chunks have no rotational DOF)")

    # ---------------------------------------------------------------- functions
    def add_funct(title, pts, yscale=1.0):
        fid = m.new("funct")
        m.functs.append(funct_block(fid, title, pts, yscale))
        return fid

    one = add_funct("unit constant", [(0.0, 1.0), (10.0 * max(duration, 1.0) + 1.0, 1.0)])

    # ---------------------------------------------------------------- gravity
    grav = scene.get("gravity", [0.0, 0.0, -9.81])
    if any(abs(x) > 0 for x in grav):
        g = m.new("grnod")
        all_ids = list(range(1, m.nid() + 1))
        m.groups.append(grnod_block(g, "all nodes", all_ids))
        for ax, name in enumerate("XYZ"):
            if abs(grav[ax]) > 0:
                gid = m.new("grav")
                m.loads.append("\n".join([f"/GRAV/{gid}", f"gravity {name}",
                                          "#funct_IDT       DIR   skew_ID sensor_ID  grnod_ID                      Ascale_x            Fscale_Y",
                                          f"{one:10d}{name:>10s}{0:10d}{0:10d}{g:10d}{'':10s}" + fnum(1.0) + fnum(grav[ax] * L)]))
        if sim.get("gravity_prestress", True):
            notes.append("gravity applied suddenly at t = 0 (no static prestress phase); self-weight stresses "
                         "are far below the strengths in these scenes, the dynamic overshoot is neglected")

    # ---------------------------------------------------------------- loads
    for load in scene.get("loads", []):
        typ = load["type"]
        if typ == "point_force":
            body = next(b for b in scene["bodies"] if b["name"] == load["body"])
            ci = meta["_cinfo"][body["name"]][load["chunk"]]
            c = body["chunks"][load["chunk"]]
            d = load["direction"]
            dn = math.sqrt(sum(x * x for x in d))
            d = [x / dn for x in d]
            fid = add_funct(f"point force {load['chunk']}", tf_points(load["magnitude"], duration))
            ids = ci["ids"]
            wts = ci["weights"]
            tot = sum(wts.values())
            # rigid-body distribution: f_i = m_i (a + alpha x r_i)
            ctr = c["center"]
            point = load.get("point") or ctr
            arm = [point[i] - ctr[i] for i in range(3)]
            torque = [arm[1] * d[2] - arm[2] * d[1], arm[2] * d[0] - arm[0] * d[2], arm[0] * d[1] - arm[1] * d[0]]
            pos = {}
            for key, nid in ids.items():
                pos[nid] = [x / L for x in m.nodes[nid - 1]]
            # inertia of the lumped nodes about the centre
            I = [[0.0] * 3 for _ in range(3)]
            for nid, w in wts.items():
                r = [pos[nid][i] - ctr[i] for i in range(3)]
                rr = sum(x * x for x in r)
                for i in range(3):
                    for j in range(3):
                        I[i][j] += w / tot * ((rr if i == j else 0.0) - r[i] * r[j])
            has_t = any(abs(x) > 0 for x in torque)
            if has_t:
                import numpy as np  # noqa: local import (only needed for eccentric loads)
                alpha = list(np.linalg.solve(np.array(I), np.array(torque)))
            else:
                alpha = [0.0, 0.0, 0.0]
            byw = defaultdict(list)
            for nid, w in wts.items():
                r = [pos[nid][i] - ctr[i] for i in range(3)]
                ax_ = [alpha[1] * r[2] - alpha[2] * r[1], alpha[2] * r[0] - alpha[0] * r[2], alpha[0] * r[1] - alpha[1] * r[0]]
                fv = tuple(round(w / tot * (d[i] + ax_[i]), 14) for i in range(3))
                byw[fv].append(nid)
            for fv, nlist in byw.items():
                g = m.new("grnod")
                m.groups.append(grnod_block(g, f"load nodes chunk {load['chunk']}", nlist))
                for ax, name in enumerate("XYZ"):
                    if abs(fv[ax]) > 0:
                        cid = m.new("cload")
                        m.loads.append("\n".join([f"/CLOAD/{cid}", f"point force chunk {load['chunk']}",
                                                  "#funct_IDT       Dir   skew_ID sensor_ID  grnod_ID   Itypfun             Ascalex             Fscaley",
                                                  f"{fid:10d}{name:>10s}{0:10d}{0:10d}{g:10d}{0:10d}" + fnum(1.0) + fnum(fv[ax] * F)]))
        elif typ == "pressure":
            body = next(b for b in scene["bodies"] if b["name"] == load["body"])
            sel = load["chunks"]
            fn = load["face_normal"]
            ax = max(range(3), key=lambda i: abs(fn[i]))
            side = 1 if fn[ax] > 0 else -1
            chosen = select_chunks(body, sel)
            segs = []
            for sg in body["_segs"]:
                if sg["chunk"] not in chosen or sg["axis"] != ax or sg["side"] != side:
                    continue
                if not covered(body, sg):
                    # /PLOAD pushes along the segment normal: orient it into the chunk
                    segs.append(sg["nodes"][::-1])
            if not segs:
                raise RuntimeError("pressure load selects no exposed face")
            sid = m.new("surf")
            m.groups.append(surf_block(sid, "pressure faces", segs))
            fid = add_funct("pressure", tf_points(load["pressure"], duration))
            pl = m.new("pload")
            m.loads.append("\n".join([f"/PLOAD/{pl}", "pressure",
                                      "#  surf_ID  functIDT sensor_ID    Ipinch      Idel   Itypfun            Ascale_x            Fscale_y",
                                      f"{sid:10d}{fid:10d}{0:10d}{0:10d}{0:10d}{0:10d}" + fnum(1.0) + fnum(P)]))
            meta.setdefault("pressure_segments", []).append(len(segs))
        else:
            raise NotImplementedError(f"load type {typ}")

    # ---------------------------------------------------------------- contacts
    def add_surf(title, segs):
        sid = m.new("surf")
        m.groups.append(surf_block(sid, title, segs))
        return sid

    def add_grnod(title, ids):
        g = m.new("grnod")
        m.groups.append(grnod_block(g, title, ids))
        return g

    def type7(title, gnod, surf, gap, fric, stfac=1.0, istf=0, vis=None):
        vis = float(opts["contact_damping"]) if vis is None else vis
        iid = m.new("inter")
        m.inter.append("\n".join([
            f"/INTER/TYPE7/{iid}", title,
            "# grnod_id   surf_id      Istf      Ithe      Igap                Ibag      Idel     Icurv      Iadm",
            f"{gnod:10d}{surf:10d}{istf:10d}{0:10d}{0:10d}{'':10s}{0:10d}{0:10d}{0:10d}{0:10d}",
            "#          Fscalegap             Gap_max             Fpenmax                         Itied      Ists",
            fnum(0.0) + fnum(0.0) + fnum(0.0) + f"{'':20s}{0:10d}{0:10d}",
            "#              Stmin               Stmax   Percent_mesh_size               dtmin  Irem_gap   Irem_i2",
            fnum(0.0) + fnum(0.0) + fnum(0.0) + fnum(0.0) + f"{0:10d}{0:10d}",
            "#              Stfac                Fric              GAPmin              Tstart               Tstop",
            fnum(stfac) + fnum(fric) + fnum(gap * L) + fnum(0.0) + fnum(0.0),
            "#      IBC                        Inacti               VIS_S               VIS_F              Bumult",
            f"{'':7s}000{'':20s}{0:10d}" + fnum(vis) + fnum(0.0) + fnum(0.0),
            "#    Ifric    Ifiltr               Xfreq     Iform   sens_ID   fct_IDF             AscaleF   fric_ID",
            f"{0:10d}{0:10d}" + fnum(0.0) + f"{0:10d}{0:10d}{0:10d}" + fnum(0.0) + f"{0:10d}",
        ]))
        return iid

    body_surf_id = {}
    body_grnod_id = {}
    for body in scene["bodies"]:
        bn = body["name"]
        body_surf_id[bn] = add_surf(f"body {bn} chunk faces", body_surfaces[bn])
        body_grnod_id[bn] = add_grnod(f"body {bn} nodes", body_nodes[bn])
        if opts["self_contact"]:
            mus = [mats[b["material"]]["friction"] for b in body["bonds"]] or [mats[body["chunks"][0]["material"]]["friction"]]
            tcb = meta["bodies"][bn]["cohesive_thickness"]
            type7(f"self contact {bn}", body_grnod_id[bn], body_surf_id[bn], opts["contact_gap_fraction"] * tcb, min(mus))
    gap_i = float(opts["impact_gap"])

    def aabb(boxes):
        return ([min(c[a] - h[a] for c, h in boxes) for a in range(3)], [max(c[a] + h[a] for c, h in boxes) for a in range(3)])

    def clearance(b1, b2):
        d2 = 0.0
        for a in range(3):
            g = max(b1[0][a] - b2[1][a], b2[0][a] - b1[1][a], 0.0)
            d2 += g * g
        return math.sqrt(d2)

    boxes = {b["name"]: aabb([(to_world(b, c["center"]), c["half_extents"]) for c in b["chunks"]]) for b in scene["bodies"]}
    for imp in scene.get("impactors", []):
        hh = [imp["shape"]["radius"]] * 3 if imp["shape"]["type"] == "sphere" else imp["shape"]["half_extents"]
        boxes["impactor " + imp["name"]] = aabb([(imp["position"], hh)])
    keys = list(boxes)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cl = clearance(boxes[keys[i]], boxes[keys[j]])
            if 0.0 < cl < 2.0 * gap_i:
                gap_i = 0.5 * cl
                notes.append(f"body/impactor contact gap reduced to {gap_i:.3g} m (initial clearance {cl:.3g} m)")
    # elastic contact (scene contact_restitution ~ 1): no contact damping between bodies
    rest = sim.get("contact_restitution")
    vis_i = 1e-6 if (rest is not None and rest >= 0.99) else None
    if vis_i is not None:
        notes.append("contact_restitution = 1: body-body/impactor contacts without damping (VIS_S = 1e-6)")
    stf_i = float(opts["body_contact_stfac"])
    fric_override = sim.get("contact_friction")
    names = [b["name"] for b in scene["bodies"]]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            bi_, bj_ = scene["bodies"][i], scene["bodies"][j]
            mu = min(mats[bi_["chunks"][0]["material"]]["friction"], mats[bj_["chunks"][0]["material"]]["friction"])
            if fric_override is not None:
                mu = fric_override
            type7(f"contact {names[i]} on {names[j]}", body_grnod_id[names[i]], body_surf_id[names[j]], gap_i, mu,
                  stfac=stf_i, vis=vis_i)
            type7(f"contact {names[j]} on {names[i]}", body_grnod_id[names[j]], body_surf_id[names[i]], gap_i, mu,
                  stfac=stf_i, vis=vis_i)
    for iname, isf in imp_surfaces.items():
        s_imp = add_surf(f"impactor {iname} faces", isf["segs"])
        g_imp = add_grnod(f"impactor {iname} surface nodes", isf["nodes"])
        for body in scene["bodies"]:
            bn = body["name"]
            mu = min(mats[isf["material"]]["friction"], mats[body["chunks"][0]["material"]]["friction"])
            if fric_override is not None:
                mu = fric_override
            type7(f"{bn} on impactor {iname}", body_grnod_id[bn], s_imp, gap_i, mu, vis=vis_i)
            type7(f"impactor {iname} on {bn}", g_imp, body_surf_id[bn], gap_i, mu, vis=vis_i)

    # ---------------------------------------------------------------- probes / time history
    th = []
    th_nodes_disp = []  # node ids needing DX..VZ
    th_nodes_reac = []
    sects = []
    for p in scene.get("probes", []):
        typ = p["type"]
        if typ in ("chunk_displacement", "chunk_velocity"):
            bmeta = meta["bodies"][p["body"]]["chunks"][p["chunk"]]
            th_nodes_disp.extend(bmeta["nodes"])
            meta["probes"][p["name"]] = {"type": typ, "body": p["body"], "chunk": p["chunk"], "axis": p["axis"]}
        elif typ == "reaction":
            body = next(b for b in scene["bodies"] if b["name"] == p["body"])
            chosen = select_chunks(body, p["chunks"])
            ns = []
            # only nodes that can carry a reaction: faces of supported chunks that are not
            # glued to another supported chunk (interior nodes of a fully held block carry none)
            for sg in body["_segs"]:
                c = sg["chunk"]
                if c not in chosen or body["chunks"][c].get("support", "none") == "none":
                    continue
                if (c, sg["axis"], sg["side"]) in body["_fixed_pair"]:
                    continue
                ns.extend(sg["nodes"])
            ns = list(dict.fromkeys(ns))
            th_nodes_reac.extend(ns)
            meta["probes"][p["name"]] = {"type": typ, "nodes": ns, "axis": p["axis"]}
        elif typ in ("impactor_velocity", "impactor_position"):
            meta["probes"][p["name"]] = {"type": typ, "impactor": p["impactor"], "axis": p["axis"],
                                         "rbody": meta["impactors"][p["impactor"]]["rbody"]}
        elif typ == "section_force":
            body = next(b for b in scene["bodies"] if b["name"] == p["body"])
            pt, nrm = p["point"], p["normal"]
            nn = math.sqrt(sum(x * x for x in nrm))
            nrm = [x / nn for x in nrm]
            reg = p.get("region")
            elems, plus_nodes = [], []
            for b, els in zip(body["bonds"], meta["bodies"][p["body"]]["bonds"]):
                if reg and not all(reg["min"][i] <= b["centroid"][i] <= reg["max"][i] for i in range(3)):
                    continue
                sa = sum((body["chunks"][b["a"]]["center"][i] - pt[i]) * nrm[i] for i in range(3))
                sb = sum((body["chunks"][b["b"]]["center"][i] - pt[i]) * nrm[i] for i in range(3))
                if sa * sb >= 0:
                    continue
                elems.extend(els)
                plus = "a" if sa > 0 else "b"
                plus_nodes.append((els, plus))
            if not elems:
                raise RuntimeError(f"section probe {p['name']} crosses no bond")
            sects.append((p, elems, plus_nodes))
        else:
            raise NotImplementedError(f"probe type {typ}")

    # cohesive element connectivity lookup
    conn_of = {}
    for pid, lst in m.hex_by_part.items():
        for eid, conn in lst:
            conn_of[eid] = conn

    th_blocks = []
    if th_nodes_disp:
        uniq = list(dict.fromkeys(th_nodes_disp))
        tid = m.new("th")
        th_blocks.append("\n".join([f"/TH/NODE/{tid}", "probe nodes",
                                    "#      var       var       var       var       var       var       var       var       var       var",
                                    "".join(f"{v:<10s}" for v in ["DX", "DY", "DZ", "VX", "VY", "VZ"]),
                                    "#  node_ID   skew_ID node_name"] + [f"{n:10d}{0:10d}" for n in uniq]))
        meta["th"].append({"kind": "node", "vars": ["DX", "DY", "DZ", "VX", "VY", "VZ"], "ids": uniq})
    if th_nodes_reac:
        uniq = list(dict.fromkeys(th_nodes_reac))
        tid = m.new("th")
        th_blocks.append("\n".join([f"/TH/NODE/{tid}", "reaction nodes",
                                    "#      var       var       var       var       var       var       var       var       var       var",
                                    "".join(f"{v:<10s}" for v in ["REACX", "REACY", "REACZ"]),
                                    "#  node_ID   skew_ID node_name"] + [f"{n:10d}{0:10d}" for n in uniq]))
        meta["th"].append({"kind": "node", "vars": ["REACX", "REACY", "REACZ"], "ids": uniq})
    if meta["impactors"]:
        tid = m.new("th")
        rids = [v["rbody"] for v in meta["impactors"].values()]
        th_blocks.append("\n".join([f"/TH/RBODY/{tid}", "impactors",
                                    "#      var       var       var       var       var       var       var       var       var       var",
                                    "".join(f"{v:<10s}" for v in ["X", "Y", "Z", "VX", "VY", "VZ"]),
                                    "#      Obj       Obj       Obj       Obj       Obj       Obj       Obj       Obj       Obj       Obj",
                                    "".join(f"{r:10d}" for r in rids)]))
        meta["th"].append({"kind": "rbody", "vars": ["X", "Y", "Z", "VX", "VY", "VZ"], "ids": rids})
    # Section forces: OpenRadioss /SECT does not accumulate forces of connection solids
    # (TYPE43), so the crossing cohesive elements' mean stresses are recorded (/TH/BRIC)
    # and integrated by observe.py (force on the +normal side = -/+ traction x area).
    sec_elems = []
    for p, elems, plus in sects:
        info = []
        for (els, side), b in zip(plus, [None] * len(plus)):
            for e in els:
                c = conn_of[e]
                pts = [[x / L for x in m.nodes[n - 1]] for n in c[:4]]
                d1 = [pts[2][i] - pts[0][i] for i in range(3)]
                d2 = [pts[3][i] - pts[1][i] for i in range(3)]
                cr = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]]
                area = 0.5 * math.sqrt(sum(x * x for x in cr))
                nrm_e = [x / (2 * area) for x in cr]  # bottom-face normal = a -> b
                ctr = [sum(q[i] for q in [[x / L for x in m.nodes[n - 1]] for n in c]) / 8 for i in range(3)]
                info.append({"eid": e, "area": area, "centroid": ctr, "normal": nrm_e, "plus": side})
                sec_elems.append(e)
        meta["probes"][p["name"]] = {"type": "section_force", "elements": info, "normal": p["normal"],
                                     "component": p["component"], "point": p["point"]}
    if sec_elems:
        uniq = list(dict.fromkeys(sec_elems))
        tid = m.new("th")
        th_blocks.append("\n".join([f"/TH/BRIC/{tid}", "section cohesive elements",
                                    "#      var       var       var       var       var       var       var       var       var       var",
                                    "".join(f"{v:<10s}" for v in ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ", "OFF"]),
                                    "#  elem_ID          elem_name"] + [f"{e:10d}" for e in uniq]))
        meta["th"].append({"kind": "bric", "vars": ["SX", "SY", "SZ", "SXY", "SYZ", "SXZ", "OFF"], "ids": uniq})
    # part energies of cohesive parts (dissipation bookkeeping)
    coh_pids = list(coh_parts.values())
    if coh_pids:
        tid = m.new("th")
        th_blocks.append("\n".join([f"/TH/PART/{tid}", "cohesive parts",
                                    "#      var       var       var       var       var       var       var       var       var       var",
                                    "".join(f"{v:<10s}" for v in ["IE", "KE"]),
                                    "#      Obj       Obj       Obj       Obj       Obj       Obj       Obj       Obj       Obj       Obj"]
                                   + ["".join(f"{p_:10d}" for p_ in coh_pids[i:i + 10]) for i in range(0, len(coh_pids), 10)]))
        meta["th"].append({"kind": "part", "vars": ["IE", "KE"], "ids": coh_pids})
    meta["cohesive_parts"] = coh_pids

    # ---------------------------------------------------------------- write starter
    name = scene["name"]
    os.makedirs(outdir, exist_ok=True)
    st = []
    st.append("#RADIOSS STARTER")
    st.append("/BEGIN")
    st.append(f"{name}"[:100])
    st.append(f"{2026:10d}{0:10d}")
    st.append(f"{'g':>20s}{'cm':>20s}{'s':>20s}")
    st.append(f"{'g':>20s}{'cm':>20s}{'s':>20s}")
    st.append("/TITLE")
    st.append(f"stress-oracle {name} (exported by oracles/opencourant/export.py)")
    st.append("/NODE")
    for i, (x, y, z) in enumerate(m.nodes, 1):
        st.append(f"{i:10d}{x:20.12g}{y:20.12g}{z:20.12g}")
    for mid, txt in m.mats.items():
        st.append(txt)
    for pid, txt in m.props.items():
        st.append(txt)
    for pid, pdef in m.parts.items():
        st.append(f"/PART/{pid}")
        st.append(pdef["title"])
        st.append("#  prop_ID    mat_ID subset_ID")
        st.append(f"{pdef['prop']:10d}{pdef['mat']:10d}{0:10d}")
        if m.hex_by_part.get(pid):
            st.append(f"/BRICK/{pid}")
            for eid, conn in m.hex_by_part[pid]:
                st.append(f"{eid:10d}" + "".join(f"{n:10d}" for n in conn))
    st.extend(m.groups)
    st.extend(m.functs)
    st.extend(m.loads)
    st.extend(m.inter)
    st.extend(rb_texts)
    st.extend(m.misc)
    st.extend(th_blocks)
    st.append("/END")
    starter_text = "\n".join(st) + "\n"

    # ---------------------------------------------------------------- engine
    th_dt = opts["th_dt"] or 0.5 * (sim.get("sample_interval") or sim.get("frame_dt", 1 / 60))
    th_dt = min(th_dt, duration / 2000.0) if duration < 1.0 and th_dt > duration / 2000.0 and not sim.get("sample_interval") else min(th_dt, duration / 200.0)
    dt_max = float(opts["dt_safety"]) * dt_limit
    frames = max(1, int(opts["anim_frames"]))
    en = [
        "#RADIOSS ENGINE",
        f"/RUN/{name}/1",
        f"{duration:20.12g}",
        "/PRINT/-1000",
        "/TFILE/0",
        f"{th_dt:20.12g}",
        "/ANIM/DT",
        f"{0.0:20.12g}{duration / frames * (1 - 1e-9):20.12g}",
        "/ANIM/VECT/VEL",
        "/ANIM/VECT/DISP",
        "/DTIX",
        f"{0.0:20.12g}{dt_max:20.12g}",
        "/PARITH/ON",
    ]
    if opts.get("chunk_crushing") or opts.get("chunk_cdpm2"):
        # Crushed elements collapse volumetrically (J2 plasticity does not resist it) and
        # invert before their plastic strain reaches the erosion limit at high impact
        # speed (b5 at 40 m/s stopped with an inverted element in the contact). Standard
        # time-step erosion deletes an element once its distortion has cut its stable
        # step to a tenth of the run's: a numerical safeguard for elements already crushed.
        en += ["/DT/BRICK/DEL", f"{0.0:20.12g}{0.1 * dt_max:20.12g}"]
        notes.append(f"crushing: bricks deleted once their stable step falls below {0.1 * dt_max:.3g} s (a tenth of the run's)")
    engine_text = "\n".join(en) + "\n"
    with open(os.path.join(outdir, f"{name}_0000.rad"), "w") as f:
        f.write(starter_text)
    with open(os.path.join(outdir, f"{name}_0001.rad"), "w") as f:
        f.write(engine_text)

    meta["notes"] = list(dict.fromkeys(notes))
    meta.pop("_cinfo", None)
    for body in scene["bodies"]:
        body.pop("_segs", None)
        body.pop("_fixed_pair", None)
    meta["scene"] = name
    meta["duration"] = duration
    meta["th_dt"] = th_dt
    meta["dt_max_cohesive"] = dt_max
    meta["hex_dt_estimate"] = hex_dt
    meta["n_nodes"] = m.nid()
    meta["n_elements"] = m.next_eid - 1
    meta["n_cohesive"] = sum(len(e) for b in meta["bodies"].values() for e in b["bonds"])
    meta["deck_sha256"] = {
        "starter": hashlib.sha256(starter_text.encode()).hexdigest(),
        "engine": hashlib.sha256(engine_text.encode()).hexdigest(),
    }
    with open(os.path.join(outdir, "map.json"), "w") as f:
        json.dump(meta, f)
    return meta


def select_chunks(body, sel):
    if sel == "all":
        return set(range(len(body["chunks"])))
    if isinstance(sel, dict):
        if "group" in sel:
            return {i for i, c in enumerate(body["chunks"]) if sel["group"] in c.get("groups", [])}
        if "indices" in sel:
            return set(sel["indices"])
        if "region" in sel:
            r = sel["region"]
            return {i for i, c in enumerate(body["chunks"])
                    if all(r["min"][k] <= c["center"][k] <= r["max"][k] for k in range(3))}
    raise ValueError(f"bad chunk selector {sel}")


def covered(body, seg):
    """Whether a chunk face segment (nominal centre) lies on a face of another chunk."""
    c = body["chunks"][seg["chunk"]]
    ax, side = seg["axis"], seg["side"]
    t1, t2 = (ax + 1) % 3, (ax + 2) % 3
    plane = c["center"][ax] + side * c["half_extents"][ax]
    u, v = seg["center"]
    tol = 1e-9
    for j, o in enumerate(body["chunks"]):
        if j == seg["chunk"]:
            continue
        if abs((o["center"][ax] - side * o["half_extents"][ax]) - plane) > tol:
            continue
        if (abs(u - o["center"][t1]) < o["half_extents"][t1] - tol and
                abs(v - o["center"][t2]) < o["half_extents"][t2] - tol):
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene")
    ap.add_argument("outdir")
    ap.add_argument("--elements-per-chunk", type=int)
    ap.add_argument("--cohesive-thickness", type=float, help="t_c / element size")
    ap.add_argument("--no-self-contact", action="store_true")
    a = ap.parse_args(argv)
    with open(a.scene) as f:
        scene = json.load(f)
    cli = {"elements_per_chunk": a.elements_per_chunk, "cohesive_thickness": a.cohesive_thickness}
    if a.no_self_contact:
        cli["self_contact"] = False
    meta = export(scene, a.outdir, options_from(scene, cli))
    print(json.dumps({k: meta[k] for k in ("n_nodes", "n_elements", "n_cohesive", "dt_max_cohesive", "hex_dt_estimate")}))


if __name__ == "__main__":
    main()
