"""Studio renders of a .glb (e.g. one written by the Rust vehiclegen CLI).

    Blender -b --python tools/render_glb.py -- <car.glb> <out_prefix> [views ...] [--samples N]
"""
import os
import sys
import bpy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from vehgen import render  # noqa: E402

a = sys.argv[sys.argv.index('--') + 1:]
glb, out = a[0], a[1]
views = [x for x in a[2:] if not x.startswith('--') and not x.isdigit()] or ['front34', 'side', 'rear34']
samples = int(a[a.index('--samples') + 1]) if '--samples' in a else 64
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
cam = render.studio(samples=samples)
render.render_views(cam, out, views)
