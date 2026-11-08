"""Crop and enlarge an image region with a pixel grid (10 px yellow, 50 px red)
for reading coordinates off a photo.

    Blender -b --python tools/zoom.py -- <src> <dst.png> x0 y0 x1 y1 scale
"""
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]
src, dst = a[0], a[1]
x0, y0, x1, y1, sc = map(int, a[2:7])
im = bpy.data.images.load(src)
w, h = im.size
p = np.array(im.pixels[:]).reshape(h, w, 4)[::-1][y0:y1, x0:x1].copy()
p = np.repeat(np.repeat(p, sc, 0), sc, 1)
for x in range(x0, x1):
    if x % 10 == 0:
        p[:, (x - x0) * sc, :] = (1, 0, 0, 1) if x % 50 == 0 else (1, 1, 0, 1)
for y in range(y0, y1):
    if y % 10 == 0:
        p[(y - y0) * sc, :, :] = (1, 0, 0, 1) if y % 50 == 0 else (1, 1, 0, 1)
o = bpy.data.images.new('z', p.shape[1], p.shape[0])
o.pixels[:] = p[::-1].reshape(-1)
o.filepath_raw = dst
o.file_format = 'PNG'
o.save()
