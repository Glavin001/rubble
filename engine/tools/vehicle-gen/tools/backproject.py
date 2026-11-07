"""Back-project photo pixels to 3D on a horizontal plane y = h, using a
camera solved by compare.py.

    python3 tools/backproject.py <cam.json> u v h [u v h ...]
"""
import sys
import json
import numpy as np


def rot(a):
    yaw, pitch, roll = a
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


c = json.load(open(sys.argv[1]))
p = np.array(c['p'])
W, H = c['W'], c['H']
vals = [float(x) for x in sys.argv[2:]]
for u, v, h in zip(vals[0::3], vals[1::3], vals[2::3]):
    d = np.array([(u - W / 2) / p[6], -(v - H / 2) / p[6], -1.0])
    d = rot(p[3:6]) @ d
    t = (h - p[1]) / d[1]
    X = p[:3] + t * d
    print('pixel (%5.1f, %5.1f) at y=%.3f -> x %.3f, z %.3f' % (u, v, h, X[0], X[2]))
