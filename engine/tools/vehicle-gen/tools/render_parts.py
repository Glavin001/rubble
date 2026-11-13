"""Isolated and exploded views of a .glb's parts (do solid parts look whole on their own?).

    Blender -b --python tools/render_parts.py -- <car.glb> <out_dir> [shot ...] [--samples N]

A shot is `name=part[,part...]:view[:zoom[:x,y,z]]`: only the named parts (prefixes ending in `*`
match), framed from `view` (the whole of them in shot at zoom 1; `x,y,z`, in glTF axes, aims the
camera at that point instead of their centre): a direction the camera looks from, in glTF axes (+x left, +y up, +z front), as
`x,y,z` or one of out, in, front, rear, top, under, and their mixes `out-front`, `in-rear-top` ...
(out/in are relative to the first part's side of the car). `explode` as the part list moves every
panel away from the body along its direction from the car's centre. Without shots: the standard
set for the sedan's solid parts.

More part lists: `all` is every part, and `-part` (or `-prefix*`) takes parts away from the list
(`hood_off=all,-hood:front-top-out`: what is under the hood). `systems` as the part list is the
exploded view of everything: every part moved with its system (body panels outward, glass and
shell up, interior above the structure, powertrain and running gear below it, wheels outward) and
spread within it. A list entry `cut@y<1.05` (an axis, `<` or `>`, metres, glTF axes) shows only
what is on that side of the plane (a cutaway; cut solids show their red insides).
`--res WxH` sets the frame.
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
shots = [x for x in a[2:] if '=' in x and not x.startswith('--')]
res = [int(v) for v in a[a.index('--res') + 1].split('x')] if '--res' in a else [1600, 1000]
os.makedirs(out, exist_ok=True)

DEFAULT = [
    'door_FL_out=door_FL,glass_door_FL:out-front-top',
    'door_FL_in=door_FL:in-rear-top',
    'door_FL_in_trim=door_FL,glass_door_FL,door_trim_FL:in-rear-top',
    'door_FL_edge=door_FL,glass_door_FL:rear-in',
    'door_FL_frame=door_FL,glass_door_FL:in-front-top:2.4:0.76,1.0,0.45',
    'door_FL_sail=door_FL,glass_door_FL:in-rear-top:5:0.72,0.99,0.78',
    'door_RL_in=door_RL,glass_door_RL:in-front-top',
    'door_RL_in_trim=door_RL,glass_door_RL,door_trim_RL:in-front-top',
    'door_RL_edge=door_RL,glass_door_RL:front-in',
    'hood_out=hood:out-front-top',
    'hood_under=hood:under-front-out',
    'hood_edge=hood:under-rear-out:2.2:0.55,0.95,1.0',
    'trunk_out=trunk_lid:out-rear-top',
    'trunk_under=trunk_lid:under-rear-out',
    'trunk_edge=trunk_lid:under-front-out:2.2:0.6,1.05,-1.85',
    'fender_FL_in=fender_FL:in-rear-top',
    'fender_FL_out=fender_FL:out-front-top',
    'bumper_front_in=bumper_front:rear-top-out',
    'bumper_rear_in=bumper_rear:front-top-out',
    'body_shell_door=body_shell:out-front-top',
    'body_shell_jamb=body_shell:out-front:2.6:0.9,0.75,0.3',
    'body_shell_b_pillar=body_shell:out-rear-top:3.0:0.85,0.8,-0.22',
    'body_shell_inside=body_shell:under-out-rear',
    'glass=glass_*:out-front-top',
    'taillight_L=taillight_L_lens,taillight_L_housing:out-rear-top',
    'taillight_L_back=taillight_L_lens,taillight_L_housing:in-front-top',
    'headlight_L=headlight_L_lens,headlight_L_housing:out-front-top',
    'headlight_L_back=headlight_L_lens,headlight_L_housing:in-rear-top',
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
sc.render.resolution_x, sc.render.resolution_y = res
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
    if m.name.startswith('lamp_emissive') and 'Emission Strength' in surf.inputs:
        # a lit reflector burns out the still: show its shape
        surf.inputs['Emission Strength'].default_value = 0.0
        surf.inputs['Base Color'].default_value = (0.3, 0.3, 0.32, 1)
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    red = nt.nodes.new('ShaderNodeEmission')
    red.inputs[0].default_value = (1, 0, 0, 1)
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(geo.outputs['Backfacing'], mix.inputs[0])
    nt.links.new(surf.outputs[0], mix.inputs[1])
    nt.links.new(red.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], outn.inputs['Surface'])
    surf = mix
for m in bpy.data.materials:
    # the cut plane: transparent where dot(position, CUT[:3]) > CUT[3] (off: a plane far away)
    nt = m.node_tree
    outn = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None) if nt else None
    if outn is None or not outn.inputs['Surface'].links:
        continue
    surf = outn.inputs['Surface'].links[0].from_node
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    dot = nt.nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    dot.name = 'cut_dot'
    gt = nt.nodes.new('ShaderNodeMath')
    gt.operation = 'GREATER_THAN'
    gt.name = 'cut_gt'
    gt.inputs[1].default_value = 1e9
    clear = nt.nodes.new('ShaderNodeBsdfTransparent')
    cmix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(geo.outputs['Position'], dot.inputs[0])
    nt.links.new(dot.outputs['Value'], gt.inputs[0])
    nt.links.new(gt.outputs[0], cmix.inputs[0])
    nt.links.new(surf.outputs[0], cmix.inputs[1])
    nt.links.new(clear.outputs[0], cmix.inputs[2])
    nt.links.new(cmix.outputs[0], outn.inputs['Surface'])
sc.cycles.transparent_max_bounces = 64


def set_cut(spec):
    """`y<1.05` keeps what has y below 1.05 (glTF axes); None: no cut."""
    n, lim = Vector((0, 0, 0)), 1e9
    if spec:
        ax, op, val = spec[0], spec[1], float(spec[2:])
        n = gl([float(ax == 'x'), float(ax == 'y'), float(ax == 'z')])
        if op == '<':
            lim = val
        else:
            n, lim = -n, -val
    for m in bpy.data.materials:
        if m.node_tree and 'cut_dot' in m.node_tree.nodes:
            m.node_tree.nodes['cut_dot'].inputs[1].default_value = n
            m.node_tree.nodes['cut_gt'].inputs[1].default_value = lim


# the exploded view by system: (prefixes, the system's move in glTF axes, spread within it,
# move along the part's direction from the car's centre line)
SYSTEMS = [
    (('glass_',), (0, 2.1, 0), 0.5, 0.5),
    (('body_shell',), (0, 1.9, 0), 0.0, 0.0),
    (('headliner', 'sun_visor', 'interior_mirror', 'pillar_trim'), (0, 1.25, 0), 0.3, 0.0),
    (('door_trim_', 'hinge_door', 'latch_door', 'door_beam', 'window_regulator'), (0, 0, 0), 0.0, 0.75),
    (('door_', 'fender_', 'handle_', 'mirror_', 'fuel_flap'), (0, 0, 0), 0.0, 1.25),
    (('hood', 'hinge_hood', 'latch_hood', 'hood_strut'), (0, 1.0, 0.7), 0.0, 0.0),
    (('trunk_lid', 'hinge_trunk', 'latch_trunk'), (0, 1.0, -0.7), 0.0, 0.0),
    (('bumper_front', 'grille', 'headlight', 'plate_front', 'absorber_front', 'impact_beam_front'), (0, 0, 1.0), 0.6, 0.0),
    (('bumper_rear', 'taillight', 'plate_rear', 'absorber_rear', 'impact_beam_rear'), (0, 0, -1.0), 0.6, 0.0),
    (('wiper',), (0, 0.5, 0.3), 0.0, 0.0),
    (('seat', 'dash', 'console', 'steering_wheel', 'steering_column', 'pedal', 'carpet', 'parcel', 'boot_', 'spare', 'belt_', 'instrument', 'glovebox'), (0, 0.75, 0), 0.45, 0.0),
    (('structure_', 'wheel_well'), (0, 0, 0), 0.12, 0.0),
    (('engine', 'transmission', 'driveshaft', 'differential', 'halfshaft', 'radiator', 'fan_', 'condenser', 'battery', 'airbox', 'coolant', 'washer', 'brake_booster', 'fuse_box', 'tower_brace', 'undertray'), (0, -1.0, 0), 0.55, 0.0),
    (('exhaust', 'fuel_'), (0, -1.55, 0), 0.3, 0.0),
    (('rim_', 'tyre_', 'hub_cap', 'wheel'), (0, -2.1, 0), 0.25, 0.9),
    (('',), (0, -2.1, 0), 0.25, 0.0),
]


def system_moves():
    blo, bhi = bounds([objs['body_shell']])
    c = (blo + bhi) / 2
    groups = {}
    for n, o in objs.items():
        k = next(i for i, sy in enumerate(SYSTEMS) if n.startswith(sy[0]))
        groups.setdefault(k, []).append((n, o))
    for k, members in groups.items():
        _, move, spread, radial = SYSTEMS[k]
        glo, ghi = bounds([o for _, o in members])
        gc = (glo + ghi) / 2
        for n, o in members:
            lo, hi = bounds([o])
            pc = (lo + hi) / 2
            d = gl(move) + (pc - gc) * spread
            if radial:
                side = pc - c
                side.y = side.z = 0.0
                if side.length > 0.2:
                    d += side.normalized() * radial
            o.location = home[n] + d
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
    name, spec = shot.split('=', 1)
    f = spec.split(':')
    names, view, zoom = f[0].split(','), f[1], float(f[2]) if len(f) > 2 else 1.0
    cut = next((k[4:] for k in names if k.startswith('cut@')), None)
    names = [k for k in names if not k.startswith('cut@')]
    set_cut(cut)
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
    elif names == ['systems']:
        sel = list(objs.values())
        system_moves()
    else:
        def match(n, keys):
            return any(n == k or (k.endswith('*') and n.startswith(k[:-1])) for k in keys)
        keep = [k for k in names if not k.startswith('-')]
        drop = [k[1:] for k in names if k.startswith('-')]
        sel = [o for n, o in objs.items() if ('all' in keep or match(n, keep)) and not match(n, drop)]
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
