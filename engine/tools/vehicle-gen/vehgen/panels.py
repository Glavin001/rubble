"""Turning welded skin patches into deliverable panels: shut-line insets,
rolled hems, inner panels, and glass made from the openings."""
import numpy as np
from .mesh import Mesh, MAT

HEM_OUT = 0.0025     # how far the roll's apex reaches past the skin edge
HEM_DROP = 0.0040    # depth of the apex below the skin
FLANGE_IN = 0.0070   # flange returns this far under the skin
FLANGE_DROP = 0.0085


def faces_list(m):
    return [f for f in m.q] + [f for f in m.t]


def boundary_loops(m):
    """Directed boundary edges (face on the left seen from the front) chained
    into loops."""
    cnt = {}
    for f in faces_list(m):
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i + 1) % k])
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
    bnd = [(a, b) for (a, b) in cnt if (b, a) not in cnt]
    nxt = {}
    for a, b in bnd:
        nxt.setdefault(a, []).append(b)
    used = set()
    loops = []
    for a, b in bnd:
        if (a, b) in used:
            continue
        loop = [a]
        used.add((a, b))
        cur = b
        guard = 0
        while cur != a and guard < 200000:
            loop.append(cur)
            cands = [x for x in nxt.get(cur, []) if (cur, x) not in used]
            if not cands:
                break
            x = cands[0]
            used.add((cur, x))
            cur = x
            guard += 1
        loops.append(loop)
    return loops


def neighbours(m):
    nb = [set() for _ in range(len(m.v))]
    for f in faces_list(m):
        k = len(f)
        for i in range(k):
            a, b = int(f[i]), int(f[(i + 1) % k])
            nb[a].add(b)
            nb[b].add(a)
    return nb


def loop_dirs(m, loop, n, smooth=2):
    """Inward in-surface direction per loop vertex (miter-scaled, <= 1.6),
    optionally smoothed along the loop."""
    P = m.v[loop]
    e = np.roll(P, -1, 0) - P
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    nin = np.cross(n[loop], e)
    nin /= np.maximum(np.linalg.norm(nin, axis=1, keepdims=True), 1e-12)
    w = nin + np.roll(nin, 1, 0)
    w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
    cosh = np.clip(np.einsum('ij,ij->i', w, nin), 0.62, 1.0)
    for _ in range(smooth):
        w = 0.25 * np.roll(w, 1, 0) + 0.5 * w + 0.25 * np.roll(w, -1, 0)
        w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
    return w / cosh[:, None]


def inset_boundary(m, d, falloff=None, loops=None):
    """Move every boundary loop by d into the panel (negative d grows it),
    spreading the move over two rings so thin rows keep their shape. d may be
    a per-vertex array."""
    if loops is None:
        loops = boundary_loops(m)
    n = m.vertex_normals()
    dv = np.broadcast_to(np.asarray(d, float), (len(m.v),)).copy()
    W = np.zeros_like(m.v)
    isb = np.zeros(len(m.v), bool)
    for loop in loops:
        if len(loop) < 3:
            continue
        W[loop] = loop_dirs(m, loop, n)
        isb[loop] = True
    nb = neighbours(m)
    if falloff is None:
        # spread the move over enough rings that no row loses over 35 % of
        # its height: linear falloff over K rings
        h1 = []
        for vi in np.where(isb)[0]:
            for u in nb[vi]:
                if not isb[u]:
                    h1.append(np.linalg.norm(m.v[u] - m.v[vi]))
        hmin = np.percentile(h1, 1) if h1 else 0.02
        K = int(np.clip(np.ceil(1.6 * np.abs(dv).max() / (0.35 * max(hmin, 1e-4))), 2, 12))
        falloff = [1 - k / K for k in range(1, K)]
    move = np.zeros_like(m.v)
    move[isb] = W[isb] * dv[isb, None]
    ring = isb.copy()
    frontier = np.where(isb)[0]
    srcW = W * dv[:, None]
    for fac in falloff:
        newf = []
        acc = {}
        for vi in frontier:
            for u in nb[vi]:
                if ring[u]:
                    continue
                acc.setdefault(u, []).append(srcW[vi])
        for u, ws in acc.items():
            w = np.mean(ws, axis=0)
            srcW[u] = w
            move[u] = w * fac
            ring[u] = True
            newf.append(u)
        frontier = newf
    m.v = m.v + move
    return loops, isb


def add_hem(m, loops, island_base=1000):
    """Roll every border under the skin: apex row and flange row."""
    n = smooth_normals(m, 2)
    hem = Mesh()
    island = island_base
    for loop in loops:
        L = len(loop)
        if L < 3:
            continue
        P = m.v[loop]
        w = loop_dirs(m, loop, n, smooth=0)
        N = n[loop]
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        local = np.minimum(seg, np.roll(seg, 1))
        fin = np.minimum(FLANGE_IN, 0.85 * local)[:, None]
        A = P - w * HEM_OUT - N * HEM_DROP
        B = P + w * fin - N * FLANGE_DROP
        base = len(m.v) + len(hem.v)
        ia = base + np.arange(L)
        ib = base + L + np.arange(L)
        hem.v = np.concatenate([hem.v, A, B])
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        v1 = np.hypot(HEM_OUT, HEM_DROP)
        v2 = v1 + np.hypot(HEM_OUT + FLANGE_IN, FLANGE_DROP - HEM_DROP)
        quads, quv, isl = [], [], []
        chunk = 1.2
        for i in range(L):
            j = (i + 1) % L
            a, b = loop[i], loop[j]
            ua, ub = s[i], s[i + 1]
            c = int(ua // chunk)
            ua0, ub0 = ua - c * chunk, ub - c * chunk
            quads.append([b, a, ia[i], ia[j]])
            quv.append([[ub0, 0], [ua0, 0], [ua0, v1], [ub0, v1]])
            isl.append(island + c)
            quads.append([ia[j], ia[i], ib[i], ib[j]])
            quv.append([[ub0, v1], [ua0, v1], [ua0, v2], [ub0, v2]])
            isl.append(island + c)
        island += int(s[-1] // chunk) + 1
        q = np.array(quads)
        hem.q = np.concatenate([hem.q, q])
        hem.qm = np.concatenate([hem.qm, np.full(len(q), MAT['paint'])])
        hem.quv = np.concatenate([hem.quv, np.array(quv)])
        hem.qi = np.concatenate([hem.qi, np.array(isl)])
    # hem.q indexes the combined vertex array: append without offset
    nv = len(m.v)
    m.v = np.concatenate([m.v, hem.v])
    if m.col is not None:
        m.col = np.concatenate([m.col, np.ones((len(hem.v), 4))])
    m.q = np.concatenate([m.q, hem.q])
    m.qm = np.concatenate([m.qm, hem.qm])
    m.quv = np.concatenate([m.quv, hem.quv])
    m.qi = np.concatenate([m.qi, hem.qi])
    return island


def smooth_normals(m, it=6):
    """Vertex normals averaged over the neighbourhood, so an offset copy does
    not fold behind tight creases."""
    n = m.vertex_normals()
    nb = neighbours(m)
    idx = [np.fromiter(s_, int) for s_ in nb]
    for _ in range(it):
        n2 = n.copy()
        for i, ix in enumerate(idx):
            if len(ix):
                n2[i] = n[i] + n[ix].mean(0)
        n = n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    return n


def inner_panel(skin, isb, depth=0.035, island=2000, rings=2, mat='paint'):
    """A simple inner panel: the skin's interior (two rings in from the
    border) pushed `depth` inward, facing the other way."""
    nb = neighbours(skin)
    ring = isb.copy()
    frontier = np.where(isb)[0]
    for _ in range(rings):
        nf = []
        for vi in frontier:
            for u in nb[vi]:
                if not ring[u]:
                    ring[u] = True
                    nf.append(u)
        frontier = nf
    keep = ~ring
    n = smooth_normals(skin)
    remap = -np.ones(len(skin.v), int)
    remap[keep] = np.arange(keep.sum())
    m = Mesh()
    m.v = skin.v[keep] - n[keep] * depth
    qk = keep[skin.q].all(1) if len(skin.q) else np.zeros(0, bool)
    tk = keep[skin.t].all(1) if len(skin.t) else np.zeros(0, bool)
    m.q = remap[skin.q[qk]]
    m.qm = np.full(len(m.q), MAT[mat])
    m.quv = skin.quv[qk].copy()
    m.qi = np.where(skin.qi[qk] >= 0, island + skin.qi[qk], -1)
    m.t = remap[skin.t[tk]]
    m.tm = np.full(len(m.t), MAT[mat])
    m.tuv = skin.tuv[tk].copy()
    m.ti = np.where(skin.ti[tk] >= 0, island + skin.ti[tk], -1)
    # drop faces the offset squeezed (behind tight creases): keep the panel
    # simple and well shaped rather than complete
    T = m.triangles()
    if len(m.q):
        P = m.v[m.q]
        d1 = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)
        d2 = np.linalg.norm(P[:, 3] - P[:, 1], axis=1)
        e = np.linalg.norm(np.roll(P, -1, 1) - P, axis=2)
        ok = (e.min(1) > 0.25 * e.max(1)) & (np.minimum(d1, d2) > 0.5 * np.maximum(d1, d2) * 0.5)
        src = skin.q[qk]
        S = skin.v[src]
        es = np.linalg.norm(np.roll(S, -1, 1) - S, axis=2)
        ok &= (e.min(1) > 0.5 * es.min(1))
        m.q, m.qm, m.quv, m.qi = m.q[ok], m.qm[ok], m.quv[ok], m.qi[ok]
    m.flip()
    return m


def relax(m, it=6, lam=0.35, crease_deg=12.0):
    """Tangential Laplacian smoothing of interior, non-crease vertices: evens
    out sheared cells where blocks meet without moving the surface (moves
    are a few mm within the tangent plane)."""
    loops = boundary_loops(m)
    fixed = np.zeros(len(m.v), bool)
    for lp in loops:
        fixed[lp] = True
    # creases: vertices whose faces' normals spread more than crease_deg
    F = [f for f in m.q] + [f for f in m.t]
    fn = []
    for f in F:
        P = m.v[f]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        if len(f) == 4:
            n = n + np.cross(P[2] - P[0], P[3] - P[0])
        fn.append(n / max(np.linalg.norm(n), 1e-12))
    fn = np.array(fn)
    vn = m.vertex_normals()
    mind = np.ones(len(m.v))
    for k, f in enumerate(F):
        d = vn[f] @ fn[k]
        mind[f] = np.minimum(mind[f], d)
    fixed |= mind < np.cos(np.radians(crease_deg))
    nb = neighbours(m)
    idx = [np.fromiter(s_, int) for s_ in nb]
    free = np.where(~fixed)[0]
    for _ in range(it):
        vn = m.vertex_normals()
        new = m.v.copy()
        for i in free:
            ix = idx[i]
            if len(ix) < 3:
                continue
            d = m.v[ix].mean(0) - m.v[i]
            d -= vn[i] * np.dot(d, vn[i])
            new[i] = m.v[i] + lam * d
        m.v = new


def round_corners(m, loops, min_turn=35.0):
    """Pull sharp border corners in along their bisector: a radiused panel
    corner, and hem rows that do not twist round it."""
    n = m.vertex_normals()
    for loop in loops:
        if len(loop) < 5:
            continue
        P = m.v[loop]
        e = np.roll(P, -1, 0) - P
        L = np.linalg.norm(e, axis=1)
        t0 = -np.roll(e, 1, 0) / np.maximum(np.roll(L, 1)[:, None], 1e-12)
        t1 = e / np.maximum(L[:, None], 1e-12)
        turn = 180 - np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', t0, t1), -1, 1)))
        w = loop_dirs(m, loop, n, smooth=0)
        wl = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        sharp = turn > min_turn
        amt = 0.35 * np.minimum(L, np.roll(L, 1)) * np.clip((turn - min_turn) / 55.0, 0, 1)
        m.v[np.array(loop)[sharp]] += wl[sharp] * amt[sharp, None]


def panel(block_meshes, gap=0.0046, hem=True, inner=None, island0=0, shut=None):
    """block_meshes: list of (Mesh, island_id). Returns the finished Mesh."""
    m = Mesh()
    for g, isl in block_meshes:
        g = g.copy()
        g.qi[:] = island0 + isl
        g.ti[:] = island0 + isl
        m.add(g)
    m.weld()
    relax(m)
    loops = boundary_loops(m)
    d = HEM_OUT + gap / 2 if hem else gap / 2
    if shut is not None:
        # openings (wheel arches, glass, lamps) keep their drawn edge: only
        # shut lines against another panel are pulled back for the gap
        key = [tuple(k) for k in np.round(m.v / 1e-6).astype(np.int64)]
        d = np.array([d if k in shut else HEM_OUT * 0.6 for k in key])
    loops, isb = inset_boundary(m, d, loops=loops)
    round_corners(m, loops)
    skin = m.copy()
    if hem:
        add_hem(m, loops, island_base=island0 + 100)
    if inner is not None:
        m.add(inner_panel(skin, isb, depth=inner, island=island0 + 1000, rings=3))
    return m


def glass_from(block_meshes, depth=0.012, grow=0.006, shared=None):
    """Glass from its opening: grown under the body flange, sunk `depth`
    below the skin; edges shared with another pane are pulled back 3 mm."""
    m = Mesh()
    for g, _ in block_meshes:
        m.add(g.copy())
    m.weld()
    d = np.full(len(m.v), -grow)
    if shared:
        key = [tuple(k) for k in np.round(m.v / 1e-6).astype(np.int64)]
        for i, k in enumerate(key):
            if k in shared:
                d[i] = 0.003
    inset_boundary(m, d, falloff=(0.5, 0.2))
    n = m.vertex_normals()
    m.v = m.v - n * depth
    m.set_mat('glass')
    m.qi[:] = -1
    m.ti[:] = -1
    return m


def offset_copy(m, depth, mat=None, flip=False):
    o = m.copy()
    n = smooth_normals(o)
    o.v = o.v - n * depth
    if mat:
        o.set_mat(mat)
    if flip:
        o.flip()
    o.qi[:] = -1
    o.ti[:] = -1
    return o
