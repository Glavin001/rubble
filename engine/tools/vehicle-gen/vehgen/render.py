"""Neutral studio renders (Cycles): front 3/4, side, rear 3/4, top."""
import bpy
import numpy as np
from mathutils import Vector


def _look(cam, target):
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def studio(samples=96, res=(1600, 900)):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = 'GPU'
    except Exception:
        pass
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    world = bpy.data.worlds.new('studio')
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    # soft gradient sky: brighter overhead
    nt = world.node_tree
    tex = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.035, 0.035, 0.038, 1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.55, 0.56, 0.58, 1)
    mapr = nt.nodes.new('ShaderNodeMapRange')
    mapr.inputs['From Min'].default_value = -1
    nt.links.new(tex.outputs['Generated'], sep.inputs[0])
    norm = nt.nodes.new('ShaderNodeVectorMath')
    norm.operation = 'NORMALIZE'
    nt.links.new(tex.outputs['Generated'], norm.inputs[0])
    sep2 = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(norm.outputs[0], sep2.inputs[0])
    nt.links.new(sep2.outputs['Z'], mapr.inputs['Value'])
    nt.links.new(mapr.outputs['Result'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = 0.7
    sc.world = world
    # floor: large shadow-catching grey cyclorama
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = '_floor'
    fm = bpy.data.materials.new('_floor')
    fm.use_nodes = True
    fb = fm.node_tree.nodes['Principled BSDF']
    fb.inputs['Base Color'].default_value = (0.11, 0.11, 0.115, 1)
    fb.inputs['Roughness'].default_value = 0.45
    floor.data.materials.append(fm)
    # lights: big soft key above, two strip lights for reflections
    def area(name, loc, rot, size, power, shape='RECTANGLE', sy=None):
        bpy.ops.object.light_add(type='AREA', location=loc, rotation=rot)
        l = bpy.context.object
        l.name = name
        l.data.shape = shape
        l.data.size = size
        if sy:
            l.data.size_y = sy
        l.data.energy = power
        return l
    area('_key', (0, 0, 6), (0, 0, 0), 7, 1100, sy=3.0)
    area('_stripL', (5.0, 0, 2.6), (0, np.radians(68), 0), 7, 380, sy=0.5)
    area('_stripR', (-5.0, 0, 2.6), (0, np.radians(-68), 0), 7, 380, sy=0.5)
    area('_back', (0, 6.5, 2.5), (np.radians(-65), 0, 0), 4, 300, sy=1.5)
    area('_front', (0, -7.0, 1.6), (np.radians(80), 0, 0), 4, 220, sy=1.2)
    cam_data = bpy.data.cameras.new('_cam')
    cam = bpy.data.objects.new('_cam', cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


VIEWS = {
    # name: (camera location in Blender coords, target, lens mm)
    'front34': ((5.2, -6.0, 1.55), (0, -0.25, 0.55), 50),
    'side': ((9.5, 0.0, 0.85), (0, 0, 0.62), 50),
    'rear34': ((-5.2, 6.0, 1.75), (0, 0.25, 0.60), 50),
    'top': ((0.0, 0.0, 11.5), (0, 0, 0), 50),
    'front': ((0.0, -9.0, 0.9), (0, 0, 0.62), 60),
    'rear': ((0.0, 9.0, 1.0), (0, 0, 0.62), 60),
    'closeF': ((2.2, -3.8, 1.2), (0.2, -1.9, 0.6), 50),
    'closeR': ((-2.2, 4.0, 1.4), (-0.2, 2.0, 0.7), 50),
    'door': ((4.0, -0.2, 0.75), (0.9, -0.2, 0.62), 50),
    'roof': ((3.0, 1.5, 3.2), (0, 0.2, 1.0), 45),
}


def render_views(cam, out_prefix, views=('front34', 'side', 'rear34', 'top'), fmt='JPEG'):
    sc = bpy.context.scene
    sc.render.image_settings.file_format = fmt
    if fmt == 'JPEG':
        sc.render.image_settings.quality = 90
    outs = []
    for v in views:
        if v.startswith('at'):
            # at:x,y,z:dx,dy,dz[:lens] in glTF coordinates (target, camera offset)
            f = v.split(':')
            t = [float(x) for x in f[1].split(',')]
            o = [float(x) for x in f[2].split(',')]
            lens = float(f[3]) if len(f) > 3 else 50
            tgt = (t[0], -t[2], t[1])
            loc = (t[0] + o[0], -(t[2] + o[2]), t[1] + o[1])
            v = 'at_%s' % '_'.join('%.2f' % x for x in t)
        else:
            loc, tgt, lens = VIEWS[v]
        cam.location = loc
        cam.data.lens = lens
        if v == 'top':
            cam.rotation_euler = (0, 0, np.radians(90))
        else:
            _look(cam, tgt)
        path = f'{out_prefix}_{v}.' + ('jpg' if fmt == 'JPEG' else 'png')
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        outs.append(path)
    return outs


def quick(res=(1280, 720), wire=False):
    wire = wire or '--wire' in __import__('sys').argv
    """Workbench look-dev: fast, shows shading and topology defects."""
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'STUDIO'
    sh.color_type = 'MATERIAL'
    sh.show_cavity = False
    sh.show_specular_highlight = True
    sc.render.resolution_x, sc.render.resolution_y = res
    if wire:
        for ob in sc.objects:
            if ob.type == 'MESH':
                mod = ob.modifiers.new('wf', 'WIREFRAME')
                mod.thickness = 0.0012
                mod.use_replace = False
    cam_data = bpy.data.cameras.new('_cam')
    cam = bpy.data.objects.new('_cam', cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam
