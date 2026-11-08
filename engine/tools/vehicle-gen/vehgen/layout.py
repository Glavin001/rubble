"""Block-structured quad meshing over charts.

A layout is a set of corner nodes, edges between them and four-sided blocks.
Each block lives in a chart (a 2D parameter space mapped to 3D). Opposite
edges of a block share a chord, so they get the same number of segments;
counts are solved over the whole layout (union-find), which makes every
shared edge sample identically and lets neighbouring blocks weld exactly.
Edges are sampled evenly by 3D arc length; block interiors come from
transfinite (Coons) interpolation in the block's chart.
"""
import numpy as np
from .mesh import grid


class TubeChart:
    def __init__(self, tube):
        self.tube = tube

    def map(self, p):
        p = np.asarray(p, float).reshape(-1, 2)
        return self.tube.eval(p[:, 0], p[:, 1])

    def inv(self, P):
        raise ValueError('tube chart has no inverse; define the edge in the tube chart')

    def normal(self, p):
        p = np.asarray(p, float).reshape(-1, 2)
        return self.tube.eval(p[:, 0], p[:, 1], with_normal=True)[1]

    def uv(self, p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.stack([p[:, 0], self.tube.girth(p[:, 0], p[:, 1])], 1)


class GraphChart:
    """z = f(x, y): front and rear end faces. facing = +1 (front) or -1."""

    def __init__(self, fz, facing):
        self.fz = fz
        self.facing = facing

    def map(self, p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.stack([p[:, 0], p[:, 1], self.fz(p[:, 0], p[:, 1])], 1)

    def inv(self, P):
        return np.asarray(P, float).reshape(-1, 3)[:, :2]

    def normal(self, p):
        p = np.asarray(p, float).reshape(-1, 2)
        n = np.zeros((len(p), 3))
        n[:, 2] = self.facing
        return n

    def uv(self, p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.stack([p[:, 0] * self.facing, p[:, 1]], 1)


class Block:
    def __init__(self, corners, chart, part, mat, tag):
        self.c = corners
        self.chart = chart
        self.part = part
        self.mat = mat
        self.tag = tag


class Layout:
    def __init__(self, h=0.02, h_max=0.029):
        self.h = h
        self.h_max = h_max
        self.charts = {}
        self.nodes = {}
        self.edges = {}
        self.blocks = []
        self.counts = None
        self._cache = {}

    def chart(self, name, ch):
        self.charts[name] = ch

    def node(self, nid, chart, p):
        self.nodes[nid] = (chart, np.asarray(p, float))

    def node_pos(self, nid):
        ch, p = self.nodes[nid]
        return self.charts[ch].map(p)[0]

    def node_param(self, nid, chart):
        ch, p = self.nodes[nid]
        if ch == chart:
            return p
        return self.charts[chart].inv(self.charts[ch].map(p))[0]

    def edge(self, a, b, chart=None, curve=None, min_count=1):
        """curve(t) -> (len(t), 2) params in `chart`, from node a to node b."""
        if (a, b) in self.edges or (b, a) in self.edges:
            k = (a, b) if (a, b) in self.edges else (b, a)
            del self.edges[k]
        self.edges[(a, b)] = dict(chart=chart, curve=curve, min=min_count)

    def block(self, corners, chart, part, mat='paint', tag=None):
        """corners (a, b, c, d): bottom a->b, right b->c, top d->c, left a->d."""
        self.blocks.append(Block(tuple(corners), chart, part, mat, tag))

    # -- edges ----------------------------------------------------------------
    def _edef(self, a, b, chart_hint):
        if (a, b) in self.edges:
            return (a, b), self.edges[(a, b)], False
        if (b, a) in self.edges:
            return (b, a), self.edges[(b, a)], True
        self.edges[(a, b)] = dict(chart=None, curve=None, min=1)
        return (a, b), self.edges[(a, b)], False

    def _echart(self, k, d, chart_hint):
        if d['chart'] is not None:
            return d['chart']
        ca = self.nodes[k[0]][0]
        cb = self.nodes[k[1]][0]
        if ca == cb:
            return ca
        return chart_hint

    def _dense(self, k, d, chart_hint, n=160):
        ch = self._echart(k, d, chart_hint)
        t = np.linspace(0, 1, n + 1)
        if d['curve'] is not None:
            p = d['curve'](t)
        else:
            pa = self.node_param(k[0], ch)
            pb = self.node_param(k[1], ch)
            p = pa[None] + (pb - pa)[None] * t[:, None]
        P = self.charts[ch].map(p)
        # snap ends to the nodes' exact positions
        P[0] = self.node_pos(k[0])
        P[-1] = self.node_pos(k[1])
        return ch, p, P

    def sample(self, a, b, n, chart_hint):
        k, d, rev = self._edef(a, b, chart_hint)
        key = (k, n)
        if key not in self._cache:
            ch, p, P = self._dense(k, d, chart_hint)
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            ts = np.linspace(0, s[-1], n + 1)
            idx = np.interp(ts, s, np.arange(len(P)))
            i0 = np.clip(np.floor(idx).astype(int), 0, len(P) - 2)
            f = (idx - i0)[:, None]
            pp = p[i0] * (1 - f) + p[i0 + 1] * f
            PP = self.charts[ch].map(pp)
            PP[0] = P[0]
            PP[-1] = P[-1]
            self._cache[key] = (ch, pp, PP)
        ch, pp, PP = self._cache[key]
        if rev:
            return ch, pp[::-1].copy(), PP[::-1].copy()
        return ch, pp.copy(), PP.copy()

    def length(self, a, b, chart_hint):
        k, d, _ = self._edef(a, b, chart_hint)
        _, _, P = self._dense(k, d, chart_hint, 96)
        return np.linalg.norm(np.diff(P, axis=0), axis=1).sum()

    # -- counts ---------------------------------------------------------------
    def solve(self, h_overrides=None):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        def ek(a, b):
            return (a, b) if (a, b) in self.edges else (b, a)

        for bl in self.blocks:
            a, b, c, d = bl.c
            for (x, y) in ((a, b), (b, c), (d, c), (a, d)):
                self._edef(x, y, bl.chart)
            union(ek(a, b), ek(d, c))
            union(ek(b, c), ek(a, d))
        # a chord's count: its median edge at the target size, unless that
        # would stretch its longest edge past h_max (30 mm). One long outlier
        # (say the strip over a wheel arch) then does not crowd every other
        # block on the chord with thin rows.
        lens = {}
        mins = {}
        seen = set()
        for bl in self.blocks:
            a, b, c, d = bl.c
            for (x, y) in ((a, b), (b, c), (d, c), (a, d)):
                k = ek(x, y)
                if k in seen:
                    continue
                seen.add(k)
                r = find(k)
                lens.setdefault(r, []).append(self.length(x, y, bl.chart))
                mins[r] = max(mins.get(r, 1), self.edges[k]['min'])
        need = {}
        for r, Ls in lens.items():
            Ls = np.array(Ls)
            n = max(int(np.ceil(np.median(Ls) / self.h - 0.15)),
                    int(np.ceil(Ls.max() / self.h_max - 0.05)), mins[r], 1)
            need[r] = n
        self.counts = {k: need[find(k)] for k in list(parent.keys())}

    def count(self, a, b):
        k = (a, b) if (a, b) in self.edges else (b, a)
        return self.counts[k]

    # -- meshing --------------------------------------------------------------
    def mesh_block(self, bl):
        a, b, c, d = bl.c
        ch = self.charts[bl.chart]
        n = self.count(a, b)
        m = self.count(a, d)
        sides = []
        for (x, y, cnt) in ((a, b, n), (d, c, n), (a, d, m), (b, c, m)):
            ech, pp, PP = self.sample(x, y, cnt, bl.chart)
            if ech != bl.chart:
                pp = ch.inv(PP)
            sides.append((pp, PP))
        (B, B3), (T, T3), (Lf, L3), (R, R3) = sides
        s = np.linspace(0, 1, n + 1)[:, None, None]
        t = np.linspace(0, 1, m + 1)[None, :, None]
        c00, c10, c01, c11 = B[0], B[-1], T[0], T[-1]
        Pp = ((1 - t) * B[:, None] + t * T[:, None] + (1 - s) * Lf[None, :] + s * R[None, :]
              - ((1 - s) * (1 - t) * c00 + s * (1 - t) * c10 + (1 - s) * t * c01 + s * t * c11))
        P3 = ch.map(Pp.reshape(-1, 2)).reshape(n + 1, m + 1, 3)
        P3[:, 0] = B3
        P3[:, -1] = T3
        P3[0, :] = L3
        P3[-1, :] = R3
        uv = ch.uv(Pp.reshape(-1, 2)).reshape(n + 1, m + 1, 2)
        g = grid(P3, uv=uv, mat=bl.mat)
        # orient: compare the grid normal with the chart's outward normal
        ci, cj = n // 2, m // 2
        du = P3[min(ci + 1, n), cj] - P3[max(ci - 1, 0), cj]
        dv = P3[ci, min(cj + 1, m)] - P3[ci, max(cj - 1, 0)]
        nn = np.cross(du, dv)
        ref = ch.normal(Pp[ci, cj][None])[0]
        if np.dot(nn, ref) < 0:
            g.flip()
        return g, Pp, P3

    def parts(self):
        out = {}
        for bl in self.blocks:
            out.setdefault(bl.part, []).append(bl)
        return out
