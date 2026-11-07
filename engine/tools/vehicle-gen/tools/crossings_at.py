"""Where two parts of a built vehicle cross: the centroids (glTF coords) of
the crossing triangles on each side, to find what check.py's "no part pairs
cross" failures are.

    Blender -b --python tools/crossings_at.py -- <car.glb> partA/partB [partC/partD ...]
"""
import sys
import numpy as np
import bpy
from mathutils.bvhtree import BVHTree

a = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])


def get(n):
    ob = bpy.data.objects[n]
    M = ob.matrix_world
    P = np.array([(M @ v.co)[:] for v in ob.data.vertices])
    return P, [tuple(p.vertices) for p in ob.data.polygons]


def gltf(c):
    return np.c_[c[:, 0], c[:, 2], -c[:, 1]]              # Blender -> glTF


for pair in a[1:]:
    A, B = pair.split('/')
    PA, TA = get(A)
    PB, TB = get(B)
    ov = BVHTree.FromPolygons(PA.tolist(), TA).overlap(BVHTree.FromPolygons(PB.tolist(), TB))
    print(pair, len(ov), 'crossing triangle pairs')
    if not ov:
        continue
    ca = gltf(np.array([PA[list(TA[i])].mean(0) for i, _ in ov]))
    cb = gltf(np.array([PB[list(TB[j])].mean(0) for _, j in ov]))
    print('  %s:' % A, np.round(ca[:8], 3).tolist())
    print('  %s:' % B, np.round(cb[:8], 3).tolist())
