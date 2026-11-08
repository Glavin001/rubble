"""The body as a block layout: z-lines (stations) x v-lines (girth lines) on the
tube, wheel-arch openings, and end caps (front fender ends, tail panel).

Node ids: ('t', i, j) for z-line i and ring v-line j (j wraps), ('c', name, r, c)
for cap nodes.
"""
import numpy as np
from .layout import Layout, TubeChart, GraphChart


class CarBody:
    def __init__(self, tube, zlines, vlines, h=0.02):
        """zlines: list of (name, z_default). vlines: increasing left-half
        v values from 0 (floor centre) to tube.NK-1 (top centre)."""
        self.tube = tube
        self.L = Layout(h)
        self.L.chart('tube', TubeChart(tube))
        self.zn = [z[0] for z in zlines]
        self.zi = {n: i for i, n in enumerate(self.zn)}
        self.z0 = [z[1] for z in zlines]
        vl = list(vlines)
        self.vl = vl
        self.nl = len(vl)
        ring = vl + [tube.VM - v for v in reversed(vl[1:-1])]
        self.ring = ring
        self.J = len(ring)
        self.zover = {}
        self.vover = {}
        self.holes = set()
        self.arcs = []

    # -- indexing ---------------------------------------------------------------
    def j(self, jl, side='L'):
        """ring index of left-half v-line index jl on a side."""
        if side == 'L':
            return jl
        return (self.J - jl) % self.J

    def jl_of(self, j):
        """(left index, side) for ring index j."""
        if j < self.nl:
            return j, 'L'
        return self.J - j, 'R'

    def nid(self, i, j):
        """Ring line j in [0, J]; J is the floor centre line again (v = VM),
        a distinct node at the same place so the wrap edge runs 17 -> 18."""
        return ('t', i, j)

    def set_z(self, i, jl, z):
        """Override z of node (i, jl) on both sides."""
        self.zover[(i, jl)] = z

    def set_zv(self, i, jl, z, v):
        self.zover[(i, jl)] = z
        self.vover[(i, jl)] = v

    def param(self, i, j):
        if j == self.J:
            z, v = self.param(i, 0)
            return z, self.tube.VM - v
        jl, side = self.jl_of(j)
        z = self.zover.get((i, jl), self.z0[i])
        v = self.vover.get((i, jl), self.vl[jl])
        if side == 'R':
            v = self.tube.VM - v
        return z, v

    # -- features ---------------------------------------------------------------
    def end_fit(self, i, fz, iters=8):
        """Make z-line i lie on the end surface z = fz(x, y)."""
        for jl in range(self.nl):
            z = self.zover.get((i, jl), self.z0[i])
            v = self.vover.get((i, jl), self.vl[jl])
            for _ in range(iters):
                P = self.tube.eval(z, v)[0]
                z = float(fz(np.array([abs(P[0])]), np.array([P[1]]))[0])
            self.zover[(i, jl)] = z

    def arch(self, center, R, cols, rows, thetas, vrange):
        """Wheel opening. cols: (i_front, i_top, i_rear) z-line indices; rows:
        left v-line indices from the lowest (rocker) to the arch-top line;
        thetas: angles (deg, 0 = forward, 90 = up) of the boundary nodes in
        order front-bottom ... top ... rear-bottom."""
        zc, yc = center
        iF, iT, iR = cols
        nodes = [(iF, r) for r in rows] + [(iT, rows[-1])] + [(iR, r) for r in reversed(rows)]
        assert len(nodes) == len(thetas)
        tube = self.tube
        for (i, jl), th in zip(nodes, thetas):
            a = np.radians(th)
            z = zc + R * np.cos(a)
            y = yc + R * np.sin(a)
            v = tube.v_at_height(z, y, *vrange)[0]
            self.set_zv(i, jl, z, v)
        for k in range(len(nodes) - 1):
            (i0, j0), (i1, j1) = nodes[k], nodes[k + 1]
            t0, t1 = np.radians(thetas[k]), np.radians(thetas[k + 1])

            def curve(t, t0=t0, t1=t1, side='L'):
                a = t0 + (t1 - t0) * t
                z = zc + R * np.cos(a)
                y = yc + R * np.sin(a)
                v = tube.v_at_height(z, y, *vrange)
                if side == 'R':
                    v = tube.VM - v
                return np.stack([z, v], 1)

            for side in ('L', 'R'):
                a_ = self.nid(i0, self.j(j0, side))
                b_ = self.nid(i1, self.j(j1, side))
                self.arcs.append((a_, b_, (lambda t, t0=t0, t1=t1, s=side: curve(t, t0, t1, s))))
        # hole blocks
        assert iT == iF + 1 and iR == iT + 1
        for r0, r1 in zip(rows[:-1], rows[1:]):
            for i in (iF, iT):
                self.holes.add((i, r0))
                self.holes.add((i, (self.J - r1) % self.J))

    # -- tube blocks ------------------------------------------------------------
    def build_nodes(self):
        for i in range(len(self.zn)):
            for j in range(self.J + 1):
                self.L.node(self.nid(i, j), 'tube', self.param(i, j))
        for a, b, c in self.arcs:
            self.L.edge(a, b, chart='tube', curve=c)

    def tube_blocks(self, assign):
        """assign(zname_from, zname_to, jl_lo, jl_hi, side) -> (part, mat) or None.
        Blocks between ring lines j and j+1; on the right side jl decreases."""
        for i in range(len(self.zn) - 1):
            for j in range(self.J):
                j2 = j + 1
                if (i, j) in self.holes:
                    continue
                jl0, s0 = self.jl_of(j)
                jl1, s1 = self.jl_of(j2 % self.J)
                side = 'L' if j < self.nl - 1 else 'R'
                lo, hi = (jl0, jl1) if side == 'L' else (jl1, jl0)
                if j2 == self.J:
                    lo, hi = 0, 1
                r = assign(self.zn[i], self.zn[i + 1], lo, hi, side)
                if r is None:
                    continue
                part, mat = r if isinstance(r, tuple) else (r, 'paint')
                a = self.nid(i, j)
                b = self.nid(i + 1, j)
                c = self.nid(i + 1, j2)
                d = self.nid(i, j2)
                self.L.block((a, b, c, d), 'tube', part, mat)

    # -- caps -------------------------------------------------------------------
    def cap(self, name, fz, facing, left, right, bottom, top, part, holes=(),
            overrides=None, arcs=None, mat='paint', xy_interior=None):
        """A structured cap. left/right: lists bottom->top; bottom/top: lists
        left->right; entries are node ids or (x, y) for new cap nodes. Corner
        entries must agree. holes: set of (r, c) blocks left open. overrides:
        {(r, c): (x, y)} for interior/boundary cap nodes. arcs: {((r,c),(r,c)):
        curve(t)->(x,y)} cap-chart edge curves."""
        ch = name
        if ch not in self.L.charts:
            self.L.chart(ch, GraphChart(fz, facing))
        R = len(left) - 1
        C = len(bottom) - 1
        assert len(right) == R + 1 and len(top) == C + 1
        ids = {}
        xy = np.zeros((R + 1, C + 1, 2))

        def pos(e):
            if isinstance(e, tuple) and e and e[0] in ('t', 'c'):
                P = self.L.node_pos(e)
                return P[:2]
            return np.asarray(e, float)

        def setb(r, c, e):
            xy[r, c] = pos(e)
            if isinstance(e, tuple) and e and e[0] in ('t', 'c'):
                ids[(r, c)] = e

        for r in range(R + 1):
            setb(r, 0, left[r])
            setb(r, C, right[r])
        for c in range(C + 1):
            setb(0, c, bottom[c])
            setb(R, c, top[c])
        # interior by TFI over node indices (fractions by boundary chord)
        for r in range(1, R):
            for c in range(1, C):
                s = c / C
                t = r / R
                xy[r, c] = ((1 - t) * xy[0, c] + t * xy[R, c] + (1 - s) * xy[r, 0] + s * xy[r, C]
                            - ((1 - s) * (1 - t) * xy[0, 0] + s * (1 - t) * xy[0, C]
                               + (1 - s) * t * xy[R, 0] + s * t * xy[R, C]))
        if xy_interior:
            xy_interior(xy)
        for k, p in (overrides or {}).items():
            xy[k] = p
        for r in range(R + 1):
            for c in range(C + 1):
                if (r, c) not in ids:
                    nid = ('c', name, r, c)
                    self.L.node(nid, ch, xy[r, c])
                    ids[(r, c)] = nid
        for (ka, kb), cur in (arcs or {}).items():
            self.L.edge(ids[ka], ids[kb], chart=ch, curve=cur)
        for r in range(R):
            for c in range(C):
                if (r, c) in holes:
                    continue
                pt = part(r, c) if callable(part) else part
                if pt is None:
                    continue
                self.L.block((ids[(r, c)], ids[(r, c + 1)], ids[(r + 1, c + 1)], ids[(r + 1, c)]),
                             ch, pt, mat)
        return ids, xy


def circle_arc(xc, yc, R, a0, a1):
    a0, a1 = np.radians(a0), np.radians(a1)

    def f(t):
        a = a0 + (a1 - a0) * t
        return np.stack([xc + R * np.cos(a), yc + R * np.sin(a)], 1)
    return f
