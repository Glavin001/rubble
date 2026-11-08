"""Turntable showcase video of a built vehicle: the camera circles the car once
in the studio set, rising and falling a little, and the frames are encoded to
an H.264 mp4 with ffmpeg.

    Blender -b <car.blend> --python showcase.py -- <out.mp4> [--seconds 10] [--fps 30]
        [--res 1280x720] [--samples 32] [--eevee] [--frames a-b]

--eevee renders with EEVEE (fast, raster) instead of Cycles. Frames are kept
next to the video (<out>_frames/), and frames already there are not rendered
again, so an interrupted run resumes. --frames renders only that range (no
encode), to test timing.
"""
import os
import sys
import subprocess
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vehgen import render  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
out = os.path.abspath(argv[0])


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


seconds = float(opt('--seconds', 10))
fps = int(opt('--fps', 30))
W, H = (int(x) for x in opt('--res', '1280x720').split('x'))
samples = int(opt('--samples', 32))
n = int(round(seconds * fps))
fr = opt('--frames', None)
first, last = (int(x) for x in fr.split('-')) if fr else (0, n - 1)

cam = render.studio(samples=samples, res=(W, H))
sc = bpy.context.scene
if '--eevee' in argv:
    sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.eevee.taa_render_samples = samples
    sc.eevee.use_raytracing = True
    sc.eevee.use_shadows = True
cam.data.lens = 45
sc.render.image_settings.file_format = 'PNG'
fdir = os.path.splitext(out)[0] + '_frames'
os.makedirs(fdir, exist_ok=True)

# orbit in Blender axes (x, -z, y of glTF): start at the front 3/4, go round
# once; the camera rises over the rear and comes back down at the front
target = Vector((0, 0, 0.58))
for f in range(first, last + 1):
    path = os.path.join(fdir, 'f%04d.png' % f)
    if os.path.exists(path):
        continue
    t = f / n
    a = np.radians(-49) + 2 * np.pi * t
    # ease the speed a little so the 3/4 views linger
    a += 0.10 * np.sin(4 * np.pi * t)
    r = 7.4 + 0.5 * np.cos(2 * np.pi * t)
    h = 1.35 + 0.55 * (1 - np.cos(2 * np.pi * t)) / 2
    cam.location = Vector((-r * np.sin(a), -r * np.cos(a), h))
    d = target - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print('frame %d/%d' % (f + 1, n), flush=True)

if fr is None:
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps),
                    '-i', os.path.join(fdir, 'f%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '18', '-movflags', '+faststart', out], check=True)
    print('wrote', out)
