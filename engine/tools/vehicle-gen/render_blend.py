"""Render views of a built .blend.

    Blender -b <file.blend> --python render_blend.py -- <out_prefix> [views] [--quick] [--samples N]
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vehgen import render  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
out = argv[0]
views = [a for a in argv[1:] if not a.startswith('--') and not a.isdigit()] or ['front34', 'side', 'rear34', 'top']
samples = int(argv[argv.index('--samples') + 1]) if '--samples' in argv else 96
cam = render.quick() if '--quick' in argv else render.studio(samples=samples)
render.render_views(cam, out, views)
