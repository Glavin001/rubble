"""A small numpy mesh container (quads + triangles, per-face material and
per-corner UVs) and the primitive generators the parts are built from.

Coordinates are glTF's: metres, +Y up, the car's front +Z, +X the car's left.
"""
import numpy as np

MATERIALS = ['paint', 'chrome', 'trim_black', 'rubber', 'glass', 'lamp_lens',
             'lamp_emissive', 'interior_fabric', 'interior_plastic', 'metal_bare']
MAT = {n: i for i, n in enumerate(MATERIALS)}


_ISLAND = [100000]


def new_island():
    _ISLAND[0] += 1
    return _ISLAND[0]


class Mesh:
    def __init__(self):
        self.v = np.zeros((0, 3))
        self.q = np.zeros((0, 4), int)
        self.qm = np.zeros(0, int)
        self.quv = np.zeros((0, 4, 2))
        self.qi = np.zeros(0, int)        # uv island id per quad (-1: none)
        self.t = np.zeros((0, 3), int)
        self.tm = np.zeros(0, int)
        self.tuv = np.zeros((0, 3, 2))
        self.ti = np.zeros(0, int)
        self.col = None                   # (N,4) vertex colour or None

    # -- construction -----------------------------------------------------
    @staticmethod
    def from_faces(v, quads=None, tris=None, mat='paint', quv=None, tuv=None,
                   island=-1):
        m = Mesh()
        m.v = np.asarray(v, float).reshape(-1, 3)
        mi = MAT[mat] if isinstance(mat, str) else mat
        if quads is not None and len(quads):
            m.q = np.asarray(quads, int).reshape(-1, 4)
            m.qm = np.full(len(m.q), mi)
            m.quv = (np.zeros((len(m.q), 4, 2)) if quv is None
                     else np.asarray(quv, float).reshape(-1, 4, 2))
            m.qi = np.full(len(m.q), island)
        if tris is not None and len(tris):
            m.t = np.asarray(tris, int).reshape(-1, 3)
            m.tm = np.full(len(m.t), mi)
            m.tuv = (np.zeros((len(m.t), 3, 2)) if tuv is None
                     else np.asarray(tuv, float).reshape(-1, 3, 2))
            m.ti = np.full(len(m.t), island)
        return m

    def add(self, o):
        if o is None:
            return self
        n = len(self.v)
        if self.col is not None or o.col is not None:
            a = self.col if self.col is not None else np.ones((len(self.v), 4))
            b = o.col if o.col is not None else np.ones((len(o.v), 4))
            self.col = np.concatenate([a, b])
        self.v = np.concatenate([self.v, o.v])
        self.q = np.concatenate([self.q, o.q + n])
        self.qm = np.concatenate([self.qm, o.qm])
        self.quv = np.concatenate([self.quv, o.quv])
        self.qi = np.concatenate([self.qi, o.qi])
        self.t = np.concatenate([self.t, o.t + n])
        self.tm = np.concatenate([self.tm, o.tm])
        self.tuv = np.concatenate([self.tuv, o.tuv])
        self.ti = np.concatenate([self.ti, o.ti])
        return self

    def copy(self):
        return Mesh().add(self)

    def set_mat(self, mat):
        mi = MAT[mat]
        self.qm[:] = mi
        self.tm[:] = mi
        return self

    def tint(self, rgba):
        self.col = np.tile(np.asarray(rgba, float), (len(self.v), 1))
        return self

    def transform(self, R=None, t=(0, 0, 0)):
        if R is not None:
            self.v = self.v @ np.asarray(R).T
            if np.linalg.det(R) < 0:
                self.flip()
        self.v = self.v + np.asarray(t, float)
        return self

    def mirror_x(self):
        self.v = self.v * np.array([-1.0, 1, 1])
        self.flip()
        return self

    def flip(self):
        self.q = self.q[:, ::-1].copy()
        self.quv = self.quv[:, ::-1].copy()
        self.t = self.t[:, ::-1].copy()
        self.tuv = self.tuv[:, ::-1].copy()
        return self

    def ntris(self):
        return 2 * len(self.q) + len(self.t)

    def triangles(self):
        q = self.q
        return np.concatenate([self.t, q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    def weld(self, tol=1e-7):
        """Merge coincident vertices (exactly shared edge samples)."""
        key = np.round(self.v / tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True,
                                  return_inverse=True)
        inv = inv.reshape(-1)
        self.v = self.v[first]
        if self.col is not None:
            self.col = self.col[first]
        self.q = inv[self.q]
        self.t = inv[self.t]
        # drop collapsed faces; quads with one collapsed edge become triangles
        keepq = np.ones(len(self.q), bool)
        newt, newtm, newtuv, newti = [], [], [], []
        for i, f in enumerate(self.q):
            u = len(set(f.tolist()))
            if u == 4:
                continue
            keepq[i] = False
            if u == 3:
                for k in range(4):
                    if f[k] == f[(k + 1) % 4]:
                        idx = [j for j in range(4) if j != (k + 1) % 4]
                        newt.append(f[idx])
                        newtm.append(self.qm[i])
                        newtuv.append(self.quv[i][idx])
                        newti.append(self.qi[i])
                        break
        if newt:
            self.t = np.concatenate([self.t, np.array(newt)])
            self.tm = np.concatenate([self.tm, np.array(newtm)])
            self.tuv = np.concatenate([self.tuv, np.array(newtuv)])
            self.ti = np.concatenate([self.ti, np.array(newti)])
        self.q, self.qm, self.quv, self.qi = (self.q[keepq], self.qm[keepq],
                                              self.quv[keepq], self.qi[keepq])
        kt = np.array([len(set(f.tolist())) == 3 for f in self.t], bool) \
            if len(self.t) else np.zeros(0, bool)
        self.t, self.tm, self.tuv, self.ti = (self.t[kt], self.tm[kt],
                                              self.tuv[kt], self.ti[kt])
        return self

    def vertex_normals(self):
        tri = self.triangles()
        a, b, c = self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]]
        fn = np.cross(b - a, c - a)
        n = np.zeros_like(self.v)
        for k in range(3):
            np.add.at(n, tri[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(ln, 1e-12)

    def bbox(self):
        return self.v.min(0), self.v.max(0)


# -- primitives -----------------------------------------------------------------

def grid(P, uv=None, mat='paint', closed_u=False, closed_v=False, island=-1):
    """Quad grid from points P (nu, nv, 3); faces wound so (dP/du x dP/dv)
    is the front."""
    nu, nv = P.shape[:2]
    idx = np.arange(nu * nv).reshape(nu, nv)
    if closed_u:
        idx = np.concatenate([idx, idx[:1]], 0)
    if closed_v:
        idx = np.concatenate([idx, idx[:, :1]], 1)
    a = idx[:-1, :-1]
    b = idx[1:, :-1]
    c = idx[1:, 1:]
    d = idx[:-1, 1:]
    quads = np.stack([a, b, c, d], -1).reshape(-1, 4)
    quv = None
    if uv is not None:
        # uv may carry the seam column/row explicitly: (nu+closed_u, nv+closed_v)
        U = np.asarray(uv, float)
        if closed_u and U.shape[0] == nu:
            U = np.concatenate([U, U[:1]], 0)
        if closed_v and U.shape[1] == nv:
            U = np.concatenate([U, U[:, :1]], 1)
        ua = U[:-1, :-1]
        ub = U[1:, :-1]
        uc = U[1:, 1:]
        ud = U[:-1, 1:]
        quv = np.stack([ua, ub, uc, ud], 2).reshape(-1, 4, 2)
    return Mesh.from_faces(P.reshape(-1, 3), quads, mat=mat, quv=quv,
                           island=island)


def lathe(profile, segs, mat='rubber', cap_ends=False, a0=0.0, a1=2 * np.pi):
    """Revolve profile [(r, a), ...] about the +X axis (a = axial position).
    Points with r == 0 become poles. Front faces point away from the axis
    when the profile runs with the axis on its left (increasing a at r>0
    going outward is the usual tyre orientation: pass the profile in the
    order that sweeps the outside clockwise when seen from +Y)."""
    pr = np.asarray(profile, float)
    full = abs((a1 - a0) - 2 * np.pi) < 1e-9
    n = segs if full else segs + 1
    th = a0 + (a1 - a0) * np.arange(n) / (segs if full else segs)
    verts = []
    ids = np.zeros((len(pr), n), int)
    for i, (r, a) in enumerate(pr):
        if r < 1e-9:
            ids[i, :] = len(verts)
            verts.append([a, 0, 0])
        else:
            for j in range(n):
                ids[i, j] = len(verts)
                verts.append([a, r * np.cos(th[j]), r * np.sin(th[j])])
    quads, tris = [], []
    jr = range(n) if full else range(n - 1)
    for i in range(len(pr) - 1):
        for j in jr:
            j2 = (j + 1) % n
            a, b, c, d = ids[i, j], ids[i, j2], ids[i + 1, j2], ids[i + 1, j]
            f = [a, b, c, d]
            u = []
            for x in f:
                if x not in u:
                    u.append(x)
            if len(u) == 4:
                quads.append(f)
            elif len(u) == 3:
                tris.append(u)
    m = Mesh.from_faces(np.array(verts), quads, tris, mat=mat)
    return m


def rbox(center, half, radius, n=3, mat='trim_black'):
    """Closed rounded box: a cube grid pushed onto the Minkowski sum of a box
    and a sphere. half: (hx, hy, hz); radius < min(half)."""
    half = np.asarray(half, float)
    r = min(radius, half.min() * 0.999)
    inner = half - r
    m = Mesh()
    # six faces of a cube with (n_k) divisions proportional to size
    divs = np.full(3, 2 * (n + 1))
    for axis in range(3):
        for sgn in (-1, 1):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            if sgn < 0:
                u_ax, v_ax = v_ax, u_ax
            nu, nv = divs[u_ax], divs[v_ax]
            U = np.linspace(-1, 1, nu + 1)
            V = np.linspace(-1, 1, nv + 1)
            P = np.zeros((nu + 1, nv + 1, 3))
            # cluster samples towards the edges so the rounds get vertices
            Uc = np.sign(U) * np.abs(U) ** 0.6
            Vc = np.sign(V) * np.abs(V) ** 0.6
            P[:, :, axis] = sgn
            P[:, :, u_ax] = Uc[:, None]
            P[:, :, v_ax] = Vc[None, :]
            P = P * half
            cl = np.clip(P, -inner, inner)
            d = P - cl
            ln = np.linalg.norm(d, axis=-1, keepdims=True)
            P = cl + d / np.maximum(ln, 1e-12) * r
            m.add(grid(P, mat=mat))
    m.weld(1e-6)
    m.v += np.asarray(center, float)
    return m


def sweep(path, section, mat='chrome', closed=False, caps=True, up=(0, 1, 0),
          scale=None, twist_up=None):
    """Sweep a closed 2D section [(sx, sy), ...] (counter-clockwise) along a 3D
    path. sx follows the frame's side vector, sy its up vector. scale: per-path
    point (sx, sy) multipliers. Returns a closed tube (caps by fans)."""
    P = np.asarray(path, float)
    S = np.asarray(section, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    if closed:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ups = np.tile(np.asarray(up, float), (n, 1)) if twist_up is None else np.asarray(twist_up, float)
    side = np.cross(T, ups)
    bad = np.linalg.norm(side, axis=1) < 1e-6
    side[bad] = np.cross(T[bad], [1, 0, 0])
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    upv = np.cross(side, T)
    sc = np.ones((n, 2)) if scale is None else np.asarray(scale, float)
    G = (P[:, None, :] + side[:, None, :] * (S[None, :, 0:1] * sc[:, None, 0:1])
         + upv[:, None, :] * (S[None, :, 1:2] * sc[:, None, 1:2]))
    m = grid(G, uv=_tube_uv(G, closed), mat=mat, closed_u=closed, closed_v=True,
             island=new_island())
    if caps and not closed:
        for end, sgn in ((0, -1), (n - 1, 1)):
            ring = np.arange(len(S)) + end * len(S)
            tris = zigzag_cap(ring, m.v, T[end] * sgn)
            m.t = np.concatenate([m.t, np.array(tris)])
            m.tm = np.concatenate([m.tm, np.full(len(tris), MAT[mat])])
            m.tuv = np.concatenate([m.tuv, np.zeros((len(tris), 3, 2))])
            m.ti = np.concatenate([m.ti, np.full(len(tris), -1)])
    return m


def zigzag_cap(ring, P, outward):
    """Triangulate a ring (indices) by ear clipping, always cutting the
    best-shaped ear: well-shaped triangles for long thin sections and
    densified (collinear) edges."""
    ring = list(ring)
    pts3 = P[ring]
    c = pts3.mean(0)
    nrm = np.asarray(outward, float)
    nrm = nrm / max(np.linalg.norm(nrm), 1e-12)
    ax = np.cross(nrm, [1, 0, 0])
    if np.linalg.norm(ax) < 0.1:
        ax = np.cross(nrm, [0, 1, 0])
    ax /= np.linalg.norm(ax)
    ay = np.cross(nrm, ax)
    q = np.stack([(pts3 - c) @ ax, (pts3 - c) @ ay], 1)
    area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    idx = list(range(len(ring)))
    if area < 0:
        idx = idx[::-1]
    tris = []
    while len(idx) > 3:
        best, bi = -1.0, None
        k = len(idx)
        for i in range(k):
            a, b, cc = idx[i - 1], idx[i], idx[(i + 1) % k]
            A, B, C = q[a], q[b], q[cc]
            cr = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
            if cr <= 1e-10:
                continue
            ok = True
            for j in idx:
                if j in (a, b, cc):
                    continue
                Pq = q[j]
                d1 = (B[0] - A[0]) * (Pq[1] - A[1]) - (B[1] - A[1]) * (Pq[0] - A[0])
                d2 = (C[0] - B[0]) * (Pq[1] - B[1]) - (C[1] - B[1]) * (Pq[0] - B[0])
                d3 = (A[0] - C[0]) * (Pq[1] - C[1]) - (A[1] - C[1]) * (Pq[0] - C[0])
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    ok = False
                    break
            if not ok:
                continue
            L = max(np.linalg.norm(B - A), np.linalg.norm(C - B), np.linalg.norm(A - C))
            qual = cr / (L * L)
            if qual > best:
                best, bi = qual, i
        if bi is None:
            bi = 1
        a, b, cc = idx[bi - 1], idx[bi], idx[(bi + 1) % len(idx)]
        tris.append([ring[a], ring[b], ring[cc]])
        idx.pop(bi)
    tris.append([ring[idx[0]], ring[idx[1]], ring[idx[2]]])
    T = np.array(tris)
    n = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]).sum(0)
    if np.dot(n, nrm) < 0:
        T = T[:, ::-1]
    return T


def _tube_uv(G, closed_u):
    """Metric unwrap of a ring grid G (n, m, 3): u along the path, v round the
    ring, with the seam column (and row) carried past the end."""
    Gc = np.concatenate([G, G[:, :1]], 1)
    if closed_u:
        Gc = np.concatenate([Gc, Gc[:1]], 0)
    du = np.linalg.norm(np.diff(Gc, axis=0), axis=2).mean(1)
    dv = np.linalg.norm(np.diff(Gc, axis=1), axis=2).mean(0)
    u = np.concatenate([[0], np.cumsum(du)])
    v = np.concatenate([[0], np.cumsum(dv)])
    U, V = np.meshgrid(u, v, indexing='ij')
    return np.stack([U, V], -1)


def circle(r, n, cx=0.0, cy=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def rrect(w, h, r, n=4):
    """Counter-clockwise rounded rectangle (w, h full sizes), corner arcs n."""
    r = min(r, w / 2 * 0.999, h / 2 * 0.999)
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / n)
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.array(pts)


def loft(sections, mat='interior_fabric', caps=True):
    """Loft closed 3D rings sections (k, m, 3) into a closed solid."""
    S = np.asarray(sections, float)
    k, mcount = S.shape[:2]
    m = grid(S, uv=_tube_uv(S, False), mat=mat, closed_v=True, island=new_island())
    if caps:
        for end, sgn in ((0, 1), (k - 1, -1)):
            ring = np.arange(mcount) + end * mcount
            other = S[1] if end == 0 else S[-2]
            outward = S[end].mean(0) - other.mean(0)
            tris = zigzag_cap(ring, m.v, outward)
            m.t = np.concatenate([m.t, np.array(tris)])
            m.tm = np.concatenate([m.tm, np.full(len(tris), MAT[mat])])
            m.tuv = np.concatenate([m.tuv, np.zeros((len(tris), 3, 2))])
            m.ti = np.concatenate([m.ti, np.full(len(tris), -1)])
    return m


def orient_outward(m, center=None):
    """Flip a closed mesh so its signed volume is positive."""
    tri = m.triangles()
    a, b, c = m.v[tri[:, 0]], m.v[tri[:, 1]], m.v[tri[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum()
    if vol < 0:
        m.flip()
    return m


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def box(center, half, mat='trim_black', R=None):
    """Closed axis-aligned box (optionally rotated by R about its centre)."""
    h = np.asarray(half, float)
    s = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    v = s * h
    if R is not None:
        v = v @ np.asarray(R).T
    v = v + np.asarray(center, float)
    q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
    m = Mesh.from_faces(v, q, mat=mat)
    return orient_outward(m)


def closed_lathe(profile, segs, mat, a0=0.0, a1=2 * np.pi):
    """Revolve a closed profile loop [(r, a), ...] (not repeating the first
    point) into a closed solid, outward facing."""
    pr = np.asarray(profile, float)
    pr = np.concatenate([pr, pr[:1]])
    m = lathe(pr, segs, mat=mat, a0=a0, a1=a1)
    m.weld(1e-7)
    return orient_outward(m)


def cube_uv(m, island_base=0, scale=1.0):
    """Per-face box projection; island per dominant axis and sign."""
    tri_n = []
    for faces, uvs, isl in ((m.q, 'quv', 'qi'), (m.t, 'tuv', 'ti')):
        if not len(faces):
            continue
        P = m.v[faces]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        if faces.shape[1] == 4:
            n = n + np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])
        ax = np.argmax(np.abs(n), 1)
        sg = (np.take_along_axis(n, ax[:, None], 1)[:, 0] > 0).astype(int)
        U = np.zeros(faces.shape + (2,))
        for a in range(3):
            o = [k for k in range(3) if k != a]
            sel = ax == a
            U[sel] = P[sel][:, :, o] * scale
        setattr(m, uvs, U)
        setattr(m, isl, island_base + ax * 2 + sg)
    return m


def catmull(P, n=6, closed=False):
    """Catmull-Rom densify a polyline."""
    P = np.asarray(P, float)
    if closed:
        Q = np.concatenate([P[-1:], P, P[:2]])
    else:
        Q = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    out = []
    segs = len(P) if closed else len(P) - 1
    for i in range(segs):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        for k in range(n):
            t = k / n
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-1])
    return np.array(out)
