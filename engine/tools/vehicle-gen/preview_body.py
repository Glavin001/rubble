"""Quick look at a design's body panels only.

Blender -b --python preview_body.py -- <design> <out_prefix> [views...]
"""
import sys
import os
import importlib
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'designs'))

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
design = argv[0] if argv else 'mach1_1969'
out = argv[1] if len(argv) > 1 else '/tmp/body'
views = [a for a in argv[2:] if not a.startswith('--')] or ['front34', 'side', 'rear34']
QUICK = '--quick' in argv
WIRE = '--wire' in argv

from vehgen import assemble, blender, render  # noqa: E402

D = importlib.import_module(design)
t0 = time.time()
parts, cb = assemble.body_parts(D)
print('body built in %.1fs' % (time.time() - t0))
blender.reset_scene()
mats = blender.make_materials(D.PAINT, D.PAINT_METALLIC, D.PAINT_ROUGHNESS)
root = blender.make_root()
for name, m in parts.items():
    blender.make_object(name, m, mats, root)
cam = (render.quick(wire=WIRE) if QUICK or WIRE else render.studio(samples=48, res=(1280, 720)))
render.render_views(cam, out, views)
