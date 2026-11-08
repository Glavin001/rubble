"""List the worst-aspect triangles of one part near a point (glTF coords),
to find where check.py's aspect failures come from.

    Blender -b --python tools/aspect_at.py -- <car.glb> <part> x,y,z radius
"""
import sys
import numpy as np
import bpy

a = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
ob = bpy.data.objects[a[1]]
M = ob.matrix_world
P = np.array([(M @ v.co)[:] for v in ob.data.vertices])
P = np.c_[P[:, 0], P[:, 2], -P[:, 1]]                     # Blender -> glTF
T = np.array([p.vertices[:] for p in ob.data.polygons])
c = np.array([float(x) for x in a[2].split(',')])
r = float(a[3])
A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
L = np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)], 1)
area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
asp = L.max(1) ** 2 / np.maximum(2 * area, 1e-12) * (np.sqrt(3) / 2)
cen = (A + B + C) / 3
sel = np.where((np.linalg.norm(cen - c, axis=1) < r) & (asp > 6))[0]
print('bad', len(sel))
for i in sel[np.argsort(-asp[sel])][:12]:
    print(round(float(asp[i]), 1), np.round(P[T[i]], 3).tolist())
