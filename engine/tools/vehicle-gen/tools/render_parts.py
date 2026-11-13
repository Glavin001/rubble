"""Isolated and exploded views of a .glb's parts (do solid parts look whole on their own?).

    Blender -b --python tools/render_parts.py -- <car.glb> <out_dir> [shot ...] [--samples N]

A shot is `name=part[,part...]:view[:zoom[:x,y,z]]`: only the named parts (prefixes ending in `*`
match), framed from `view` (the whole of them in shot at zoom 1; `x,y,z`, in glTF axes, aims the
camera at that point instead of their centre): a direction the camera looks from, in glTF axes (+x left, +y up, +z front), as
`x,y,z` or one of out, in, front, rear, top, under, and their mixes `out-front`, `in-rear-top` ...
(out/in are relative to the first part's side of the car). `explode` as the part list moves every
panel away from the body along its direction from the car's centre. Without shots: the standard
set for the sedan's solid parts.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

a = sys.argv[sys.argv.index('--') + 1:]
glb, out = a[0], a[1]
samples = int(a[a.index('--samples') + 1]) if '--samples' in a else 48
shots = [x for x in a[2:] if '=' in x]
os.makedirs(out, exist_ok=True)

DEFAULT = [
    'door_FL_out=door_FL,glass_door_FL:out-front-top',
    'door_FL_in=door_FL:in-rear-top',
    'door_FL_in_trim=door_FL,glass_door_FL,door_trim_FL:in-rear-top',
    'door_FL_edge=door_FL,glass_door_FL:rear-in:1.0',
    'door_FL_frame=door_FL,glass_door_FL:in-front-top:2.4',
    'door_RL_in=door_RL,glass_door_RL:in-front-top',
    'hood_under=hood:under-front-out',
    'hood_edge=hood:under-rear-out:2.2',
    'trunk_under=trunk_lid:under-rear-out',
    'fender_FL_in=fender_FL:in-rear-top',
    'fender_FL_out=fender_FL:out-front-top',
    'bumper_front_in=bumper_front:rear-top-out',
    'bumper_rear_in=bumper_rear:front-top-out',
    'body_shell_door=body_shell:out-front-top',
    'body_shell_jamb=body_shell:out-rear:2.2',
    'body_shell_inside=body_shell:under-out-rear',
    'glass=glass_*:out-front-top',
    'explode=explode:out-front-top',
    'explode_rear=explode:in-rear-top',
]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = {o.name: o for o in bpy.data.objects if o.type == 'MESH'}
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.samples = samples
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = 1600, 1000
sc.render.image_settings.file_format = 'JPEG'
sc.render.image_settings.quality = 92
sc.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.75, 0.77, 0.80, 1)
world.node_tree.nodes['Background'].inputs[1].default_value = 0.9
sc.world = world
# back faces show (a hole in a solid would show the inside): flag them red
for m in bpy.data.materials:
    m.use_backface_culling = False
    nt = m.node_tree
    if not nt:
        continue
    outn = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    surf = outn.inputs['Surface'].links[0].from_node if outn and outn.inputs['Surface'].links else None
    if surf is None or m.name.startswith(('glass', 'lamp_lens')):
        continue
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    red = nt.nodes.new('ShaderNodeEmission')
    red.inputs[0].default_value = (1, 0, 0, 1)
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(geo.outputs['Backfacing'], mix.inputs[0])
    nt.links.new(surf.outputs[0], mix.inputs[1])
    nt.links.new(red.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], outn.inputs['Surface'])
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3.0
sun.data.angle = math.radians(12)
sc.collection.objects.link(sun)
fill = bpy.data.objects.new('fill', bpy.data.lights.new('fill', 'SUN'))
fill.data.energy = 1.2
fill.data.angle = math.radians(40)
sc.collection.objects.link(fill)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
cam.data.lens = 70
sc.collection.objects.link(cam)
sc.camera = cam
home = {n: o.location.copy() for n, o in objs.items()}


def gl(v):
    """glTF (x, y, z) as Blender's (x, -z, y)."""
    return Vector((v[0], -v[2], v[1]))


def bounds(sel):
    pts = [o.matrix_world @ Vector(c) for o in sel for c in o.bound_box]
    lo = Vector((min(p[i] for p in pts) for i in range(3)))
    hi = Vector((max(p[i] for p in pts) for i in range(3)))
    return lo, hi


def direction(view, side):
    if view[0].isdigit() or view[0] == '-' and view[1].isdigit():
        return Vector([float(x) for x in view.split(',')]).normalized()
    d = Vector((0, 0, 0))
    for w in view.split('-'):
        d += Vector({'out': (side, 0, 0), 'in': (-side, 0, 0), 'front': (0, 0, 1), 'rear': (0, 0, -1),
                     'top': (0, 0.6, 0), 'under': (0, -1, 0)}[w])
    return d.normalized()


PANELS = ('door_', 'hood', 'trunk_lid', 'fender_', 'bumper_', 'glass_', 'door_trim_')
for shot in shots or DEFAULT:
    name, spec = shot.split('=')
    f = spec.split(':')
    names, view, zoom = f[0].split(','), f[1], float(f[2]) if len(f) > 2 else 1.0
    aim = gl([float(x) for x in f[3].split(',')]) if len(f) > 3 else None
    for n, o in objs.items():
        o.location = home[n]
    if names == ['explode']:
        sel = list(objs.values())
        blo, bhi = bounds([objs['body_shell']])
        c = (blo + bhi) / 2
        for n, o in objs.items():
            if n.startswith(PANELS):
                lo, hi = bounds([o])
                d = (lo + hi) / 2 - c
                d.z = max(d.z, 0.0) * 1.5
                o.location = home[n] + d.normalized() * 0.55
    else:
        sel = [o for n, o in objs.items() if any(n == k or (k.endswith('*') and n.startswith(k[:-1])) for k in names)]
    for o in objs.values():
        o.hide_render = o not in sel
    bpy.context.view_layer.update()
    lo, hi = bounds(sel)
    c = (lo + hi) / 2
    side = 1.0 if c.x >= -1e-6 else -1.0
    d = gl(direction(view, side))
    # the camera backs off until every vertex of the parts is in the frame (the frame is wider
    # than tall), with a margin
    q = (-d).to_track_quat('-Z', 'Y')
    right, up = q @ Vector((1, 0, 0)), q @ Vector((0, 1, 0))
    r = (hi - lo).length / 2
    tx = math.tan(cam.data.angle / 2)
    ty = tx * sc.render.resolution_y / sc.render.resolution_x
    dist = 0.0
    for o in sel:
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get('co', co)
        p = co.reshape(-1, 3) @ np.array(o.matrix_world.to_3x3()).T + np.array(o.matrix_world.translation) - np.array(c)
        need = np.maximum(np.abs(p @ np.array(right)) / tx, np.abs(p @ np.array(up)) / ty) + p @ np.array(d)
        dist = max(dist, float(need.max()))
    dist = dist * 1.06 / zoom
    if aim is not None:
        c = aim
    cam.location = c + d * dist
    cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y').to_euler()
    cam.data.clip_start, cam.data.clip_end = 0.01, dist + 4 * r + 10
    sun.rotation_euler = (d + Vector((0.35, 0.2, 0.6))).to_track_quat('Z', 'Y').to_euler()
    fill.rotation_euler = (d + Vector((-0.5, -0.3, -0.2))).to_track_quat('Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(out, name + '.jpg')
    bpy.ops.render.render(write_still=True)
    print('wrote', sc.render.filepath)
