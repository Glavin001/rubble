"""1D interpolation used by every design table: monotone cubic (PCHIP) so a
profile never overshoots between stations, plus small smoothing helpers."""
import numpy as np


class Pchip:
    """Fritsch–Carlson monotone cubic through (x, y); clamps outside the range."""

    def __init__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        if len(x) == 1:
            x = np.array([x[0] - 1, x[0] + 1])
            y = np.array([y[0], y[0]])
        h = np.diff(x)
        d = np.diff(y) / h
        n = len(x)
        m = np.zeros(n)
        if n == 2:
            m[:] = d[0]
        else:
            for k in range(1, n - 1):
                if d[k - 1] * d[k] <= 0:
                    m[k] = 0.0
                else:
                    w1 = 2 * h[k] + h[k - 1]
                    w2 = h[k] + 2 * h[k - 1]
                    m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
            m[0] = _end(h[0], h[1], d[0], d[1])
            m[-1] = _end(h[-1], h[-2], d[-1], d[-2])
        self.x, self.y, self.m = x, y, m

    def __call__(self, q):
        q = np.asarray(q, float)
        x, y, m = self.x, self.y, self.m
        qc = np.clip(q, x[0], x[-1])
        k = np.clip(np.searchsorted(x, qc) - 1, 0, len(x) - 2)
        h = x[k + 1] - x[k]
        t = (qc - x[k]) / h
        t2, t3 = t * t, t * t * t
        r = ((2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * h * m[k]
             + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * h * m[k + 1])
        return r


def _end(h0, h1, d0, d1):
    m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(m) > abs(3 * d0):
        return 3 * d0
    return m


def table(pairs):
    """Pchip from [(z, value), ...]."""
    z, v = zip(*pairs)
    return Pchip(z, v)


def const(v):
    return Pchip([-10, 10], [v, v])


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bump(x, c, w):
    """Smooth compact bump: 1 at c, 0 beyond |x-c| >= w."""
    t = np.clip(1 - ((np.asarray(x, float) - c) / w) ** 2, 0, 1)
    return t * t


def resample_polyline(P, n):
    """n+1 points evenly spaced by arc length along polyline P (k,d)."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n + 1)
    out = np.empty((n + 1, P.shape[1]))
    for d in range(P.shape[1]):
        out[:, d] = np.interp(t, s, P[:, d])
    return out


def arc_params(P, n):
    """Parameter fractions (into the index space of P) giving n+1 evenly spaced
    points by arc length."""
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.linspace(0, len(P) - 1, n + 1)
    t = np.linspace(0, s[-1], n + 1)
    return np.interp(t, s, np.arange(len(P)))
