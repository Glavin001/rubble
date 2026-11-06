"""Blender side: objects, materials, markers, glTF export, .blend save."""
import bpy
import numpy as np
from mathutils import Matrix
from .mesh import MATERIALS, MAT


def to_bl(P):
    """glTF (x, y up, z fwd) -> Blender (x, -z, y)."""
    P = np.asarray(P, float)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_materials(paint_rgb=(0.0, 0.1, 0.7), metallic=0.45, rough=0.26):
    spec = {
        'paint': dict(c=(*paint_rgb, 1), m=metallic, r=rough, coat=1.0),
        'chrome': dict(c=(0.92, 0.92, 0.94, 1), m=1.0, r=0.06),
        'trim_black': dict(c=(0.018, 0.018, 0.02, 1), m=0.0, r=0.55),
        'rubber': dict(c=(0.016, 0.016, 0.017, 1), m=0.0, r=0.78),
        'glass': dict(c=(0.06, 0.075, 0.08, 0.22), m=0.0, r=0.02, alpha=True),
        'lamp_lens': dict(c=(1, 1, 1, 0.45), m=0.0, r=0.03, alpha=True, vcol=True),
        'lamp_emissive': dict(c=(0.9, 0.9, 0.88, 1), m=0.6, r=0.2),
        'interior_fabric': dict(c=(0.035, 0.035, 0.037, 1), m=0.0, r=0.9),
        'interior_plastic': dict(c=(0.03, 0.03, 0.032, 1), m=0.0, r=0.55),
        'metal_bare': dict(c=(0.30, 0.30, 0.31, 1), m=1.0, r=0.55),
    }
    mats = {}
    for name in MATERIALS:
        s = spec[name]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = s['c']
        b.inputs['Metallic'].default_value = s['m']
        b.inputs['Roughness'].default_value = s['r']
        if s.get('coat'):
            b.inputs['Coat Weight'].default_value = s['coat']
            b.inputs['Coat Roughness'].default_value = 0.03
        if s.get('alpha'):
            b.inputs['Alpha'].default_value = s['c'][3]
            m.blend_method = 'BLEND'
            m.surface_render_method = 'BLENDED'
        if s.get('vcol'):
            ca = nt.nodes.new('ShaderNodeVertexColor')
            ca.layer_name = 'Color'
            nt.links.new(ca.outputs['Color'], b.inputs['Base Color'])
        if name == 'lamp_emissive':
            b.inputs['Emission Color'].default_value = (1, 0.95, 0.85, 1)
            b.inputs['Emission Strength'].default_value = 0.0
        mats[name] = m
    return mats


def make_object(name, mesh, mats, parent, sharp_deg=42):
    me = bpy.data.meshes.new(name)
    V = to_bl(mesh.v)
    nq, nt = len(mesh.q), len(mesh.t)
    me.vertices.add(len(V))
    me.vertices.foreach_set('co', V.reshape(-1).astype(np.float32))
    loops = np.concatenate([mesh.q.reshape(-1), mesh.t.reshape(-1)]).astype(np.int32)
    me.loops.add(len(loops))
    me.loops.foreach_set('vertex_index', loops)
    me.polygons.add(nq + nt)
    starts = np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3]).astype(np.int32)
    totals = np.concatenate([np.full(nq, 4), np.full(nt, 3)]).astype(np.int32)
    me.polygons.foreach_set('loop_start', starts)
    try:
        me.polygons.foreach_set('loop_total', totals)
    except (AttributeError, TypeError, RuntimeError):
        pass  # Blender 4.x derives loop totals from the starts
    fm = np.concatenate([mesh.qm, mesh.tm]).astype(np.int32)
    used = sorted(set(fm.tolist()))
    slot = {mi: k for k, mi in enumerate(used)}
    for mi in used:
        me.materials.append(mats[MATERIALS[mi]])
    me.polygons.foreach_set('material_index', np.array([slot[x] for x in fm], np.int32))
    uv = np.concatenate([mesh.quv.reshape(-1, 2), mesh.tuv.reshape(-1, 2)]).astype(np.float32)
    lay = me.uv_layers.new(name='UVMap')
    lay.data.foreach_set('uv', uv.reshape(-1))
    if mesh.col is not None:
        ca = me.color_attributes.new('Color', 'FLOAT_COLOR', 'POINT')
        ca.data.foreach_set('color', mesh.col.astype(np.float32).reshape(-1))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=np.radians(sharp_deg))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    return ob


def make_root():
    root = bpy.data.objects.new('vehicle', None)
    bpy.context.scene.collection.objects.link(root)
    return root


def make_marker(name, pos, parent, xaxis=(1, 0, 0), size=0.05):
    """Empty at glTF position pos; its local X axis along glTF direction xaxis
    (rotation about the vertical only)."""
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = 'ARROWS'
    e.empty_display_size = size
    p = to_bl(np.asarray(pos, float))
    xa = to_bl(np.asarray(xaxis, float))
    ang = np.arctan2(xa[1], xa[0])
    e.matrix_world = Matrix.Translation(p.tolist()) @ Matrix.Rotation(ang, 4, 'Z')
    bpy.context.scene.collection.objects.link(e)
    e.parent = parent
    return e


def triangulate_beauty():
    """Triangulate every mesh with the 'beauty' rule (best triangles from
    each quad, mirror-independent) for export."""
    import bmesh
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(ob.data)
        bm.free()


def export(path_glb, path_blend=None):
    if path_blend:
        bpy.ops.wm.save_as_mainfile(filepath=path_blend, compress=True)
    triangulate_beauty()
    bpy.ops.export_scene.gltf(
        filepath=path_glb, export_format='GLB', export_yup=True, export_apply=True,
        export_normals=True, export_tangents=False, export_materials='EXPORT',
        export_vertex_color='MATERIAL', export_extras=False, export_cameras=False,
        export_lights=False, use_selection=False)
