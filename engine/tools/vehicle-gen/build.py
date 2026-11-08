"""Build a vehicle: .glb + .blend + renders + acceptance report.

    Blender -b --python build.py -- <design> [--out DIR] [--no-render] [--samples N]

Designs live in designs/<name>.py. Output defaults to
engine/scenes/vehicles/<name>/.
"""
import sys
import os
import importlib
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'designs'))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
design = argv[0] if argv else 'mach1_1969'


def opt(name, default=None):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


out_dir = opt('--out', os.path.join(REPO, 'engine', 'scenes', 'vehicles', design))
samples = int(opt('--samples', '128'))
do_render = '--no-render' not in argv
_v = opt('--views', 'front34,side,rear34,top')
# 'at:' views carry commas, so a list holding one is separated by ';'
views = _v.split(';') if 'at:' in _v else _v.split(',')
os.makedirs(out_dir, exist_ok=True)

from vehgen import assemble, blender, render, uvpack  # noqa: E402
from vehgen.mesh import Mesh  # noqa: E402

D = importlib.import_module(design)
t0 = time.time()
parts, cb = assemble.body_parts(D)
new, extras, markers = D.parts(cb)
for k, m in extras.items():
    parts[k].add(m)
parts.update(new)
if hasattr(D, 'post'):
    D.post(parts)
S = uvpack.pack(parts)
uvpack.normalise_other(parts)
print('built %d parts in %.1fs; paint atlas %.2f m square' % (len(parts), time.time() - t0, S))

blender.reset_scene()
mats = blender.make_materials(D.PAINT, D.PAINT_METALLIC, D.PAINT_ROUGHNESS, getattr(D, 'PAINT_COAT', 1.0))
root = blender.make_root()
for name in sorted(parts):
    blender.make_object(name, parts[name], mats, root)
for name in sorted(markers):
    pos, xa = markers[name]
    blender.make_marker(name, pos, root, xa)

glb = os.path.join(out_dir, design + '.glb')
blend = os.path.join(out_dir, design + '.blend')
blender.export(glb, blend)
print('wrote', glb)

if do_render:
    rdir = os.path.join(out_dir, 'renders')
    os.makedirs(rdir, exist_ok=True)
    cam = render.studio(samples=samples)
    render.render_views(cam, os.path.join(rdir, design), views)

if opt('--out') is None:
    # keep the model viewer's list (engine/scenes/vehicles/index.json) current
    # (after the renders: it lists their thumbnails)
    sys.path.insert(0, os.path.join(HERE, 'tools'))
    import vehicles_manifest  # noqa: E402
    print('wrote %s (%d vehicles)' % vehicles_manifest.write())
