"""Blender (background) helper for the FE-import spike: build, decimate, export and render.

    Blender -b --factory-startup --python-exit-code 1 -P blender_fe_view.py -- build MESH_DIR OUT_DIR
    Blender -b --factory-startup --python-exit-code 1 -P blender_fe_view.py -- decimate OUT_DIR TARGET_TRIS
    Blender -b --factory-startup --python-exit-code 1 -P blender_fe_view.py -- render OUT_DIR BLEND_NAME

build:    reads MESH_DIR/parts.bin + parts.json (fe_mesh.py), one object per part named
          "<pid> <name>", one material per crash section, saves OUT_DIR/camry_full.blend and
          OUT_DIR/camry_full.glb (full resolution).
decimate: opens camry_full.blend, adds a collapse Decimate modifier to every object with
          ratio = TARGET_TRIS / total (at least 40 triangles per part kept), applies it,
          saves camry_decimated.blend and camry_decimated.glb.
render:   Workbench orthographic stills of BLEND_NAME: side, top, front three-quarter, and a
          structure-only view (doors, glass, hood, fenders, trunk lid, bumper covers, wheels
          and the "other" section hidden).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

COLORS = {
    "front_lower": (0.85, 0.15, 0.12, 1), "front_upper": (0.95, 0.55, 0.10, 1),
    "cabin": (0.15, 0.40, 0.85, 1), "door": (0.20, 0.70, 0.30, 1), "rear": (0.55, 0.30, 0.75, 1),
    "wheel_suspension": (0.25, 0.25, 0.25, 1), "other": (0.75, 0.75, 0.72, 1),
    "unclassified": (1.0, 0.0, 1.0, 1),
}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def materials():
    mats = {}
    for k, c in COLORS.items():
        m = bpy.data.materials.new(k)
        m.diffuse_color = c
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = c
            bsdf.inputs["Roughness"].default_value = 0.6
        mats[k] = m
    return mats


def build(mesh_dir, out):
    clear()
    meta = json.load(open(os.path.join(mesh_dir, "parts.json")))
    mats = materials()
    raw = open(os.path.join(mesh_dir, "parts.bin"), "rb").read()
    cols = {}
    for k in COLORS:
        c = bpy.data.collections.new(k)
        bpy.context.scene.collection.children.link(c)
        cols[k] = c
    for p in meta:
        nv, nt, off = p["n_vert"], p["n_tri"], p["offset"]
        v = np.frombuffer(raw, dtype=np.float32, count=3 * nv, offset=off).reshape(-1, 3)
        idx = np.frombuffer(raw, dtype=np.uint32, count=3 * nt, offset=off + 12 * nv)
        # file is glTF axes (X left, Y up, Z forward); Blender is Z up: (x, y, z) = (Z, X, Y)
        vb = np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=1).astype(np.float32)
        me = bpy.data.meshes.new(str(p["pid"]))
        me.vertices.add(nv)
        me.vertices.foreach_set("co", vb.ravel())
        me.loops.add(3 * nt)
        me.loops.foreach_set("vertex_index", idx.astype(np.int32))
        me.polygons.add(nt)
        me.polygons.foreach_set("loop_start", np.arange(0, 3 * nt, 3, dtype=np.int32))
        me.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
        me.update(calc_edges=True)
        me.validate(verbose=False)
        me.materials.append(mats[p["section"]])
        ob = bpy.data.objects.new("%d %s" % (p["pid"], p["name"]), me)
        ob["section"] = p["section"]
        cols[p["section"]].objects.link(ob)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "camry_full.blend"))
    export(os.path.join(out, "camry_full.glb"))


def export(path):
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_apply=True,
                              export_yup=True, export_normals=False, export_texcoords=False,
                              export_materials="EXPORT")


def decimate(out, target):
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out, "camry_full.blend"))
    obs = [o for o in bpy.data.objects if o.type == "MESH"]
    total = sum(len(o.data.polygons) for o in obs)
    ratio = target / total
    for o in obs:
        n = len(o.data.polygons)
        r = min(1.0, max(ratio, 40.0 / max(n, 1)))
        if r < 1.0:
            m = o.modifiers.new("dec", "DECIMATE")
            m.decimate_type = "COLLAPSE"
            m.ratio = r
            m.use_collapse_triangulate = True
    deps = bpy.context.evaluated_depsgraph_get()
    after = 0
    for o in obs:
        if o.modifiers:
            me = bpy.data.meshes.new_from_object(o.evaluated_get(deps))
            o.modifiers.clear()
            old = o.data
            o.data = me
            bpy.data.meshes.remove(old)
        after += len(o.data.polygons)
    print("DECIMATE total %d -> %d (ratio %.4f)" % (total, after, ratio))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "camry_decimated.blend"))
    export(os.path.join(out, "camry_decimated.glb"))


HIDE_STRUCT = ("hood", "fender", "trunkouter", "trunkinner", "bumperfrontcover", "bumperrearcover",
               "bumper_foam_front", "bumperfoamrear", "rear bumper null", "bumperfront_null")


def render(out, blend):
    bpy.ops.wm.open_mainfile(filepath=os.path.join(out, blend))
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"; sh.color_type = "MATERIAL"
    sh.show_cavity = True; sh.show_object_outline = False
    sh.show_backface_culling = False
    sc.render.resolution_x, sc.render.resolution_y = 1800, 1000
    sc.render.film_transparent = False
    sc.world = sc.world or bpy.data.worlds.new("w")
    obs = [o for o in bpy.data.objects if o.type == "MESH"]
    lo = Vector((1e9, 1e9, 1e9)); hi = Vector((-1e9, -1e9, -1e9))
    for o in obs:
        for c in o.bound_box:
            v = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, v)); hi = Vector(map(max, hi, v))
    ctr = (lo + hi) / 2
    size = hi - lo
    cam_data = bpy.data.cameras.new("cam"); cam_data.type = "ORTHO"
    cam = bpy.data.objects.new("cam", cam_data); sc.collection.objects.link(cam); sc.camera = cam

    def shot(name, direction, up, scale, hide=lambda o: False):
        for o in obs:
            o.hide_render = hide(o)
        d = Vector(direction).normalized()
        cam.location = ctr + d * 20.0
        rot = (-d).to_track_quat("-Z", "Y")
        cam.rotation_euler = rot.to_euler()
        cam_data.ortho_scale = scale
        cam_data.clip_end = 100
        sc.render.filepath = os.path.join(out, name)
        bpy.ops.render.render(write_still=True)
        print("RENDERED", sc.render.filepath)

    L = size.x * 1.08
    shot("camry_side_left.png", (0, 1, 0), None, L)
    shot("camry_top.png", (0, 0, 1), None, L)
    shot("camry_front_three_quarter.png", (1, 0.8, 0.55), None, L * 0.95)

    def structural_hidden(o):
        s = o.get("section", "other")
        n = o.name.split(" ", 1)[1].lower() if " " in o.name else o.name.lower()
        if s in ("other", "door", "wheel_suspension"):
            return True
        return any(k in n for k in HIDE_STRUCT)
    shot("camry_structure_three_quarter.png", (1, 0.8, 0.55), None, L * 0.95, structural_hidden)
    shot("camry_structure_side.png", (0, 1, 0), None, L, structural_hidden)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    mode = argv[0]
    if mode == "build":
        build(argv[1], argv[2])
    elif mode == "decimate":
        decimate(argv[1], int(argv[2]))
    elif mode == "render":
        render(argv[1], argv[2])


main()
