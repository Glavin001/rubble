"""The body's outer skin as one smooth parametric surface S(z, v).

Every cross-section (constant z) is a G1 chain of cubic Béziers through key
points K0..K(n-1) on the car's left half: K0 on the centre line under the
floor, then out and up the side, over the shoulder and the greenhouse, to
K(n-1) on the centre line of the roof (or bonnet, or boot lid). Each key point's
(x, y) is a monotone spline of z, so the designer draws the car as a handful
of plan and elevation lines. v in [0, n-1] runs along the left half, v in
[n-1, 2(n-1)] mirrors it down the right side; the ring is periodic.

Param z is the 3D z, so (z, girth length) is a near-isometric UV chart.
"""
import numpy as np
from .curves import Pchip, const


class Tube:
    def __init__(self, keys, handles=None, tangents=None, displace=None):
        """keys: list of (x_of_z, y_of_z) callables.
        handles: list of (in, out) handle factors (constants) per key.
        tangents: dict key -> angle_of_z (radians, direction of increasing v
        in the left half's (x, y) plane) to override the automatic tangent.
        displace: list of f(P, n, z, v) -> (N,3) offsets added to the surface.
        """
        self.keys = keys
        self.NK = len(keys)
        self.VM = 2 * (self.NK - 1)
        self.h = handles or [(1 / 3, 1 / 3)] * self.NK
        self.tan = tangents or {}
        self.disp = displace or []

    def keypoints(self, z):
        z = np.atleast_1d(np.asarray(z, float))
        K = np.empty((len(z), self.NK, 2))
        for k, (fx, fy) in enumerate(self.keys):
            K[:, k, 0] = fx(z)
            K[:, k, 1] = fy(z)
        return K

    def _section(self, z, v):
        """Left-frame section point, derivative wrt v, and mirror flag."""
        z = np.atleast_1d(np.asarray(z, float))
        v = np.mod(np.atleast_1d(np.asarray(v, float)), self.VM)
        NK = self.NK
        mir = v > NK - 1
        vm = np.where(mir, self.VM - v, v)
        k = np.clip(np.floor(vm).astype(int), 0, NK - 2)
        t = vm - k
        K = self.keypoints(z)
        flipx = np.array([-1.0, 1.0])
        Kp = np.concatenate([K[:, 1:2] * flipx, K[:, :-1]], 1)
        Kn = np.concatenate([K[:, 1:], K[:, -2:-1] * flipx], 1)
        T = Kn - Kp
        for kk, ang in self.tan.items():
            a = ang(z)
            T[:, kk, 0] = np.cos(a)
            T[:, kk, 1] = np.sin(a)
        T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-12)
        i = np.arange(len(z))
        A, B = K[i, k], K[i, k + 1]
        TA, TB = T[i, k], T[i, k + 1]
        L = np.linalg.norm(B - A, axis=1, keepdims=True)
        # handles may be constants or functions of z
        Hin = np.stack([h[0](z) if callable(h[0]) else np.full(len(z), h[0]) for h in self.h], 1)
        Hout = np.stack([h[1](z) if callable(h[1]) else np.full(len(z), h[1]) for h in self.h], 1)
        hout = Hout[i, k][:, None]
        hin = Hin[i, k + 1][:, None]
        P1 = A + TA * hout * L
        P2 = B - TB * hin * L
        # arc-length parametrisation within each segment, so girth lines
        # (and mesh rows) are evenly spaced even where short handles make
        # the Bezier slow near a crease
        ks = np.linspace(0, 1, 17)
        U = 1 - ks
        C = (U[None, :, None] ** 3 * A[:, None] + 3 * (U ** 2 * ks)[None, :, None] * P1[:, None]
             + 3 * (U * ks ** 2)[None, :, None] * P2[:, None] + ks[None, :, None] ** 3 * B[:, None])
        seg = np.linalg.norm(np.diff(C, axis=1), axis=2)
        cum = np.concatenate([np.zeros((len(z), 1)), np.cumsum(seg, 1)], 1)
        cum /= np.maximum(cum[:, -1:], 1e-12)
        j = np.clip((cum < t[:, None]).sum(1) - 1, 0, 15)
        c0 = cum[np.arange(len(z)), j]
        c1 = cum[np.arange(len(z)), j + 1]
        t = (j + np.clip((t - c0) / np.maximum(c1 - c0, 1e-12), 0, 1)) / 16.0
        tt = t[:, None]
        u = 1 - tt
        pt = u ** 3 * A + 3 * u * u * tt * P1 + 3 * u * tt * tt * P2 + tt ** 3 * B
        d = 3 * u * u * (P1 - A) + 6 * u * tt * (P2 - P1) + 3 * tt * tt * (B - P2)
        return pt, d, mir

    def eval(self, z, v, with_normal=False):
        z = np.atleast_1d(np.asarray(z, float))
        v = np.atleast_1d(np.asarray(v, float))
        z, v = np.broadcast_arrays(z, v)
        shape = z.shape
        z = z.reshape(-1)
        v = v.reshape(-1)
        pt, d, mir = self._section(z, v)
        # outward normal in the left frame: tangent rotated -90 degrees
        n2 = np.stack([d[:, 1], -d[:, 0]], 1)
        n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
        x = np.where(mir, -pt[:, 0], pt[:, 0])
        nx = np.where(mir, -n2[:, 0], n2[:, 0])
        P = np.stack([x, pt[:, 1], z], 1)
        N = np.stack([nx, n2[:, 1], np.zeros_like(nx)], 1)
        for f in self.disp:
            P = P + f(P, N, z, v)
        if with_normal:
            return P.reshape(shape + (3,)), N.reshape(shape + (3,))
        return P.reshape(shape + (3,))

    # -- helpers --------------------------------------------------------------
    def v_at_height(self, z, y, v0, v1, iters=48):
        """Solve y(z, v) = y for v in [v0, v1] (y must be monotone there)."""
        z = np.atleast_1d(np.asarray(z, float))
        y = np.atleast_1d(np.asarray(y, float))
        lo = np.full(len(z), float(v0))
        hi = np.full(len(z), float(v1))
        ylo = self.eval(z, lo)[:, 1]
        inc = self.eval(z, hi)[:, 1] > ylo
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ym = self.eval(z, mid)[:, 1]
            go_up = (ym < y) == inc
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        return 0.5 * (lo + hi)

    def v_at_x(self, z, x, v0, v1, iters=48):
        z = np.atleast_1d(np.asarray(z, float))
        x = np.atleast_1d(np.asarray(x, float))
        lo = np.full(len(z), float(v0))
        hi = np.full(len(z), float(v1))
        inc = self.eval(z, hi)[:, 0] > self.eval(z, lo)[:, 0]
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            xm = self.eval(z, mid)[:, 0]
            go_up = (xm < x) == inc
            lo = np.where(go_up, mid, lo)
            hi = np.where(go_up, hi, mid)
        return 0.5 * (lo + hi)

    def girth_table(self, z0, z1, dz=0.01, dv=0.01):
        """Cumulative girth length s(z, v) table for the UV chart."""
        zs = np.arange(z0 - dz, z1 + 2 * dz, dz)
        vs = np.arange(0, self.VM + dv / 2, dv)
        Z, V = np.meshgrid(zs, vs, indexing='ij')
        P = self.eval(Z.reshape(-1), V.reshape(-1)).reshape(len(zs), len(vs), 3)
        seg = np.linalg.norm(np.diff(P[:, :, :2], axis=1), axis=2)
        S = np.concatenate([np.zeros((len(zs), 1)), np.cumsum(seg, 1)], 1)
        self._gt = (zs, vs, S)

    def girth(self, z, v):
        zs, vs, S = self._gt
        z = np.clip(z, zs[0], zs[-1])
        v = np.mod(v, self.VM + 1e-9)
        fi = (z - zs[0]) / (zs[1] - zs[0])
        fj = v / (vs[1] - vs[0])
        i = np.clip(np.floor(fi).astype(int), 0, len(zs) - 2)
        j = np.clip(np.floor(fj).astype(int), 0, len(vs) - 2)
        a, b = fi - i, fj - j
        return ((1 - a) * (1 - b) * S[i, j] + a * (1 - b) * S[i + 1, j]
                + (1 - a) * b * S[i, j + 1] + a * b * S[i + 1, j + 1])


def key(xs, ys):
    """Key point from two [(z, value)] tables (or constants)."""
    fx = const(xs) if np.isscalar(xs) else Pchip(*zip(*xs))
    fy = const(ys) if np.isscalar(ys) else Pchip(*zip(*ys))
    return (fx, fy)
