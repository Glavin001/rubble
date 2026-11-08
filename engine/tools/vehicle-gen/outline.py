"""Draw the real car's traced feature lines (yellow) and the model's same
lines projected through the photo's solved camera (cyan) on the photo.

    Blender -b <car.blend> --python outline.py -- <photo> <cam.json> <trace.json> <out.png> <design> [side]

trace.json: {"lines": {"name": [[u, v], ...]}} in photo pixels.
side: L or R, which side of the car faces the camera (for the tube lines).
"""
import sys
import os
import json
import numpy as np
import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'designs'))
argv = sys.argv[sys.argv.index('--') + 1:]
photo, camf, tracef, out, design = argv[:5]
side = argv[5] if len(argv) > 5 else 'R'
cam = json.load(open(camf))
p = np.array(cam['p'])
W, H = cam['W'], cam['H']


def rot(a):
    yaw, pitch, roll = a
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (Ry @ Rx @ Rz).T


def project(X):
    Xc = (np.asarray(X) - p[:3]) @ rot(p[3:6]).T
    return np.stack([W / 2 + p[6] * Xc[:, 0] / -Xc[:, 2], H / 2 - p[6] * Xc[:, 1] / -Xc[:, 2]], 1)


img = bpy.data.images.load(photo)
im = np.array(img.pixels[:]).reshape(H, W, 4)[::-1].copy()


def draw(P, col, th=1):
    P = np.asarray(P, float)
    for a, b in zip(P[:-1], P[1:]):
        n = int(np.ceil(np.linalg.norm(b - a))) + 1
        for t in np.linspace(0, 1, n):
            x, y = a + (b - a) * t
            xi, yi = int(round(x)), int(round(y))
            im[max(yi - th, 0):yi + th + 1, max(xi - th, 0):xi + th + 1, :3] = col


# model: glass outlines from the actual parts (border edges)
for ob in bpy.data.objects:
    if ob.type != 'MESH' or not ob.name.startswith('glass'):
        continue
    me = ob.data
    cnt = {}
    for poly in me.polygons:
        vs = list(poly.vertices)
        for i in range(len(vs)):
            e = tuple(sorted((vs[i], vs[(i + 1) % len(vs)])))
            cnt[e] = cnt.get(e, 0) + 1
    V = np.array([ob.matrix_world @ v.co for v in me.vertices])
    V = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)       # Blender -> glTF
    for (a, b), c in cnt.items():
        if c == 1:
            draw(project(V[[a, b]]), (0.0, 1.0, 1.0))
# model: tube lines (shoulder, belt, roof rail, centre)
import importlib  # noqa: E402
D = importlib.import_module(design)
t = D.tube()
zs = np.linspace(-2.42, 2.30, 300)
for v in (5.0, 6.0, 9.0):
    vv = v if side == 'L' else t.VM - v
    draw(project(t.eval(zs, np.full(len(zs), vv))), (0.0, 0.8, 1.0))
# real: traced lines
for name, L in json.load(open(tracef))['lines'].items():
    draw(L, (1.0, 0.9, 0.0), 1)
o = bpy.data.images.new('o', W, H)
o.pixels[:] = im[::-1].reshape(-1)
o.filepath_raw = out
o.file_format = 'PNG'
o.save()
print('wrote', out)
