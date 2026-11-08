"""Photo pixels <-> glTF 3D points through a camera solved by compare.py
(<out>.png.cam.json). Plain python3 with numpy (Blender's bundled python
works).

    backproject.py <cam.json> y H  u v [u v ...]   pixel rays onto the plane y = H
    backproject.py <cam.json> x X  u v [u v ...]   ... onto the plane x = X
    backproject.py <cam.json> z Z  u v [u v ...]   ... onto the plane z = Z
    backproject.py <cam.json> project x y z [x y z ...]   3D points -> pixels
    backproject.py <cam.json> mirror uL vL uR vR [z0 z1]
        the depth z at which a left/right pair of features (two lamps, two
        exhaust tips) is mirror-symmetric about x = 0, and their x, y there

One photo cannot separate "further back" from "further out": put a feature
on a plane you know (a measured height, the centre line) or use symmetry
(`mirror`). A ray nearly parallel to its plane (a camera at about the plane's
height) is ill-conditioned; pick another plane or another photo.
"""
import sys
import json
import numpy as np


def rot(a):
    """camera-to-world rotation for compare.py's (yaw, pitch, roll)."""
    yaw, pitch, roll = a
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


class Cam:
    def __init__(self, path):
        c = json.load(open(path))
        self.p = np.array(c['p'])
        self.W, self.H = c['W'], c['H']
        self.C = self.p[:3]
        self.R = rot(self.p[3:6])
        self.f = self.p[6]

    def ray(self, u, v):
        d = np.array([(u - self.W / 2) / self.f, -(v - self.H / 2) / self.f, -1.0])
        return self.R @ d

    def on_plane(self, u, v, axis, val):
        d = self.ray(u, v)
        t = (val - self.C[axis]) / d[axis]
        return self.C + t * d

    def project(self, X):
        Xc = self.R.T @ (np.asarray(X, float) - self.C)
        return np.array([self.W / 2 + self.f * Xc[0] / -Xc[2], self.H / 2 - self.f * Xc[1] / -Xc[2]])


def main(argv):
    cam = Cam(argv[0])
    mode = argv[1]
    if mode in ('x', 'y', 'z'):
        axis = 'xyz'.index(mode)
        val = float(argv[2])
        a = [float(x) for x in argv[3:]]
        for u, v in zip(a[0::2], a[1::2]):
            X = cam.on_plane(u, v, axis, val)
            print('pixel (%6.1f, %6.1f) -> x %.3f  y %.3f  z %.3f' % (u, v, *X))
    elif mode == 'project':
        a = [float(x) for x in argv[2:]]
        for X in zip(a[0::3], a[1::3], a[2::3]):
            print('(%.3f, %.3f, %.3f) -> pixel (%.1f, %.1f)' % (*X, *cam.project(X)))
    elif mode == 'mirror':
        uL, vL, uR, vR = (float(x) for x in argv[2:6])
        z0, z1 = (float(argv[6]), float(argv[7])) if len(argv) > 7 else (-4.0, 4.0)
        zs = np.linspace(z0, z1, 4001)
        err = [cam.on_plane(uL, vL, 2, z)[0] + cam.on_plane(uR, vR, 2, z)[0] for z in zs]
        k = int(np.argmin(np.abs(err)))
        z = zs[k]
        L, Rr = cam.on_plane(uL, vL, 2, z), cam.on_plane(uR, vR, 2, z)
        print('mirror-symmetric at z %.3f (residual %.4f m): left x %.3f y %.3f, right x %.3f y %.3f'
              % (z, err[k], L[0], L[1], Rr[0], Rr[1]))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
