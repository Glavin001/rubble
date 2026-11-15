"""Render a film written by engine-softbody's `vehicle_film` example (the CPU
reference's skinned cars, frame by frame) and encode it with ffmpeg.

    Blender -b --python render_film.py -- <film dir> <out.mp4> [--samples 16]
        [--size 1280x720] [--orbit SECONDS] [--stills DIR] [--every N]

--orbit renders the film's last frame from a camera circling the car once
(SECONDS long, 30 fps) instead of the film. --stills writes every Nth frame's
PNG (default: the first, middle and last) to DIR. Frames are kept beside the
film (<film dir>/png*), so a rerun resumes.
"""
import sys
import os
import json
import math
import struct
import subprocess

import bpy
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
film, out = argv[0], argv[1]


def opt(k, d=None):
    return argv[argv.index(k) + 1] if k in argv else d


samples = int(opt('--samples', '16'))
W, H = (int(x) for x in opt('--size', '1280x720').split('x'))
orbit = float(opt('--orbit', '0'))
stills = opt('--stills')
scene = json.load(open(os.path.join(film, 'scene.json')))

with open(os.path.join(film, 'mesh.bin'), 'rb') as f:
    nv, nt = struct.unpack('<II', f.read(8))
    col = np.frombuffer(f.read(nv * 12), dtype='<f4').reshape(nv, 3)
    idx = np.frombuffer(f.read(nt * 12), dtype='<u4').reshape(nt, 3)


def frame_pos(k):
    p = np.fromfile(os.path.join(film, 'frame_%04d.bin' % k), dtype='<f4').reshape(-1, 3)
    # Rubble (+X left, +Y up, +Z forward) to Blender (+Z up): a rotation.
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
try:
    sc.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError:
    sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = samples
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.image_settings.file_format = 'PNG'
sc.view_settings.view_transform = 'AgX'
world = bpy.data.worlds.new('w')
sc.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.55, 0.62, 0.72, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.6

# The car(s).
P0 = frame_pos(0)
me = bpy.data.meshes.new('car')
me.vertices.add(nv)
me.vertices.foreach_set('co', P0.reshape(-1))
me.loops.add(nt * 3)
me.loops.foreach_set('vertex_index', idx.reshape(-1).astype(np.int32))
me.polygons.add(nt)
me.polygons.foreach_set('loop_start', np.arange(0, nt * 3, 3, dtype=np.int32))
me.polygons.foreach_set('loop_total', np.full(nt, 3, dtype=np.int32))
me.update()
ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
rgba = np.concatenate([col, np.ones((nv, 1), 'f4')], 1)
ca.data.foreach_set('color', rgba.reshape(-1))
me.polygons.foreach_set('use_smooth', np.ones(nt, dtype=bool))
car = bpy.data.objects.new('car', me)
sc.collection.objects.link(car)
mat = bpy.data.materials.new('car')
mat.use_nodes = True
nt_ = mat.node_tree
bsdf = nt_.nodes['Principled BSDF']
attr = nt_.nodes.new('ShaderNodeAttribute')
attr.attribute_name = 'Col'
nt_.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
bsdf.inputs['Roughness'].default_value = 0.35
bsdf.inputs['Metallic'].default_value = 0.3
me.materials.append(mat)


def solid(name, rgb, rough=0.8):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1)
    b.inputs['Roughness'].default_value = rough
    return m


# Ground, obstacles, light.
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0))
bpy.context.object.data.materials.append(solid('ground', (0.32, 0.32, 0.30)))
for o in scene.get('obstacles', []):
    if 'box' in o:
        c, h = o['box']['centre'], o['box']['half']
        bpy.ops.mesh.primitive_cube_add(location=(c[0], -c[2], c[1]))
        ob = bpy.context.object
        ob.scale = (h[0], h[2], h[1])
        ob.data.materials.append(solid('wall', (0.55, 0.55, 0.52)))
    if 'cylinder' in o:
        c, r, hh = o['cylinder']['centre'], o['cylinder']['radius'], o['cylinder']['half_height']
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=2 * hh, location=(c[0], -c[2], c[1]))
        bpy.context.object.data.materials.append(solid('pole', (0.75, 0.62, 0.20), 0.5))
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3.5
sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(-35))
sc.collection.objects.link(sun)

cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 35
sc.collection.objects.link(cam)
sc.camera = cam


def look(at, frm):
    d = np.asarray(at) - np.asarray(frm)
    cam.location = frm
    cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)


n = scene['frames']
if '--first' in argv:
    n = int(opt('--first'))
pngdir = os.path.join(film, 'png_orbit' if orbit else 'png')
os.makedirs(pngdir, exist_ok=True)
keep = set()
if stills:
    os.makedirs(stills, exist_ok=True)
    every = int(opt('--every', '0'))
    total = int(orbit * 30) if orbit else n
    keep = set(range(0, total, every)) if every else {0, total // 2, total - 1}

if orbit:
    P = frame_pos(n - 1)
    me.vertices.foreach_set('co', P.reshape(-1))
    me.update()
    c = P.mean(0)
    span = np.ptp(P, 0).max()
    frames = int(orbit * 30)
    for k in range(frames):
        path = os.path.join(pngdir, 'f%04d.png' % k)
        if not os.path.exists(path):
            a = 2 * math.pi * k / frames + math.radians(30)
            r = 1.5 * span
            look((c[0], c[1], 0.6), (c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0.45 * span))
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
        if k in keep:
            subprocess.run(['cp', path, os.path.join(stills, os.path.basename(out).replace('.mp4', '_%04d.png' % k))])
else:
    prev = None
    for k in range(n):
        path = os.path.join(pngdir, 'f%04d.png' % k)
        P = frame_pos(k)
        c = P.mean(0)
        if not os.path.exists(path):
            me.vertices.foreach_set('co', P.reshape(-1))
            me.update()
            if scene['camera'] == 'chase':
                # Behind and above, along the car's travel (smoothed).
                hd = np.array([0.0, 1.0, 0.0]) if prev is None else c - prev
                h2 = np.hypot(hd[0], hd[1])
                hd = np.array([0.0, 1.0]) if h2 < 1e-4 else hd[:2] / h2
                look((c[0], c[1], 0.7), (c[0] - 7.5 * hd[0] + 2.5 * hd[1], c[1] - 7.5 * hd[1] - 2.5 * hd[0], 2.8))
            else:
                c0 = frame_pos(0).mean(0)
                # Toward the impact (the car travels along Blender -Y; two
                # cars meet at their centroid).
                ahead = 0.0 if scene['scene'] == 'c2c' else 1.4
                if scene['scene'] == 'pole':
                    look((c0[0], c0[1] - 0.4, 0.7), (c0[0] + 6.0, c0[1] - 6.0, 2.6))
                else:
                    look((c0[0], c0[1] - ahead, 0.6), (c0[0] + 7.0, c0[1] - ahead - 4.0, 2.4))
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
        prev = c if prev is None else 0.8 * prev + 0.2 * c
        if k in keep:
            subprocess.run(['cp', path, os.path.join(stills, os.path.basename(out).replace('.mp4', '_%04d.png' % k))])

os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', os.path.join(pngdir, 'f%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', out], check=True)
print('wrote', out)
