"""Render a built vehicle from the camera of a reference photograph and put
the two side by side (and overlaid), so the model can be judged against the
real car.

    Blender -b <car.blend> --python compare.py -- <photo.jpg> <points.json> <out.png> [--samples N]

points.json: {"pairs": [[u, v, x, y, z], ...]} -- pixel (u right, v down) of
a feature in the photo and its glTF position on the model (wheel hubs, tyre
contact points, lamp centres). The camera (position, orientation, focal
length) is solved from them by least squares, so the render uses the photo's
own viewpoint and lens.
"""
import sys
import os
import json
import numpy as np
import bpy
from mathutils import Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vehgen import render  # noqa: E402

argv = sys.argv[sys.argv.index('--') + 1:]
photo, pts_path, out = argv[0], argv[1], argv[2]
samples = int(argv[argv.index('--samples') + 1]) if '--samples' in argv else 64
spec = json.load(open(pts_path))
P = np.array(spec['pairs'], float)
prior = spec.get('prior', {})   # {"cam_y": [value, px_weight], "f": [px, weight]}
uv, X = P[:, :2], P[:, 2:]

img = bpy.data.images.load(photo)
W, H = img.size
ref = np.array(img.pixels[:]).reshape(H, W, 4)[::-1, :, :3].copy()   # top-down


def rot(a):
    """yaw (about +Y), pitch (about camera x), roll -> world-to-camera R.
    Camera looks along -Z_cam, +Y_cam up, +X_cam right (glTF axes)."""
    yaw, pitch, roll = a
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (Ry @ Rx @ Rz).T          # camera-to-world is Ry Rx Rz


def project(p, X):
    C = p[0:3]
    R = rot(p[3:6])
    f = p[6]
    Xc = (X - C) @ R.T
    u = W / 2 + f * Xc[:, 0] / -Xc[:, 2]
    v = H / 2 - f * Xc[:, 1] / -Xc[:, 2]
    return np.stack([u, v], 1)


def resid(p):
    r = [(project(p, X) - uv).reshape(-1)]
    if 'cam_y' in prior:
        r.append([(p[1] - prior['cam_y'][0]) * prior['cam_y'][1]])
    if 'f' in prior:
        r.append([(p[6] - prior['f'][0]) / W * prior['f'][1]])
    return np.concatenate([np.asarray(x, float) for x in r])


# initial guess: camera 9 m from the car's centre, looking at it, in the
# direction the photo's points suggest (try a ring of azimuths, keep best)
best = None
for az in np.radians(np.arange(0, 360, 20)):
    for el in (0.05, 0.15, 0.3):
        d = 9.0
        C = np.array([d * np.sin(az), 1.0 + d * np.sin(el), d * np.cos(az)])
        fwd = np.array([0, 0.6, 0]) - C
        yaw = np.arctan2(-fwd[0], -fwd[2])
        pitch = np.arctan2(fwd[1], np.hypot(fwd[0], fwd[2]))
        p = np.array([*C, yaw, pitch, 0.0, 1.2 * W])
        lam = 1e-2
        for _ in range(200):
            r = resid(p)
            J = np.zeros((len(r), 7))
            for k in range(7):
                dp = np.zeros(7)
                dp[k] = 1e-5 * (1 + abs(p[k]))
                J[:, k] = (resid(p + dp) - r) / dp[k]
            A = J.T @ J
            g = J.T @ r
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            pn = p + step
            if np.sum(resid(pn) ** 2) < np.sum(r ** 2):
                p, lam = pn, lam * 0.5
            else:
                lam *= 4
        # camera must see the points in front of it
        Xc = (X - p[:3]) @ rot(p[3:6]).T
        err = np.sqrt(np.mean(np.sum((project(p, X) - uv) ** 2, 1)))
        if np.all(Xc[:, 2] < 0) and p[6] > 0 and (best is None or err < best[0]):
            best = (err, p)
err, p = best
print('camera solve: rms %.1f px, f %.0f px (%.0f mm on 36 mm), at %s' %
      (err, p[6], p[6] / W * 36, np.round(p[:3], 2)))
for (u, v), q in zip(uv, project(p, X)):
    print('   photo (%4.0f,%4.0f)  model (%6.1f,%6.1f)' % (u, v, q[0], q[1]))

json.dump({'p': [float(x) for x in p], 'W': W, 'H': H}, open(out + '.cam.json', 'w'))

# Blender camera from the solve (glTF -> Blender axes: x, -z, y)
if '--outdoor' in argv:
    cam = render.outdoor(samples=samples, res=(W * 2, H * 2))
else:
    cam = render.studio(samples=samples, res=(W * 2, H * 2))
if '--paint' in argv:
    rgb = [float(x) for x in argv[argv.index('--paint') + 1].split(',')]
    b = bpy.data.materials['paint'].node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1)
G2B = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
Rcw = rot(p[3:6]).T                      # camera-to-world, glTF
M = np.eye(4)
M[:3, :3] = G2B @ Rcw
M[:3, 3] = G2B @ p[:3]
cam.matrix_world = Matrix(M.tolist())
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36
cam.data.lens = p[6] / W * 36
sc = bpy.context.scene
sc.render.image_settings.file_format = 'PNG'
tmp = out + '.render.png'
sc.render.filepath = tmp
bpy.ops.render.render(write_still=True)

ri = bpy.data.images.load(tmp)
rr = np.array(ri.pixels[:]).reshape(H * 2, W * 2, 4)[::-1, :, :3]
rr = rr.reshape(H, 2, W, 2, 3).mean((1, 3))          # back to photo size
gap = np.ones((H, 12, 3))


def save(arr, path):
    o = bpy.data.images.new(os.path.basename(path), arr.shape[1], arr.shape[0])
    px = np.concatenate([arr, np.ones(arr.shape[:2] + (1,))], 2)[::-1]
    o.pixels[:] = px.reshape(-1)
    o.filepath_raw = path
    o.file_format = 'PNG'
    o.save()


save(np.concatenate([ref, gap, rr], 1), out)
save(0.5 * ref + 0.5 * rr, out.replace('.png', '_overlay.png'))
print('wrote', out)

# --- line drawing: the model's visible edges (Freestyle) in cyan and the
# photo's traced lines in yellow, both over the photo ------------------------
if '--lines' in argv:
    sc.render.use_freestyle = True
    sc.render.line_thickness_mode = 'ABSOLUTE'
    sc.render.line_thickness = 1.6
    sc.render.film_transparent = True
    vl = sc.view_layers[0]
    fs = vl.freestyle_settings
    fs.crease_angle = np.radians(140)
    ls = fs.linesets[0] if fs.linesets else fs.linesets.new('lines')
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_crease = True
    ls.select_external_contour = True
    if ls.linestyle is None:
        ls.linestyle = bpy.data.linestyles.new('lines')
    ls.linestyle.color = (0.0, 1.0, 1.0)
    ls.linestyle.thickness = 1.6
    # surfaces hold out (transparent) but still hide the lines behind them
    hold = bpy.data.materials.new('_holdout')
    hold.use_nodes = True
    nt = hold.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    ho = nt.nodes.new('ShaderNodeHoldout')
    mo = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(ho.outputs[0], mo.inputs['Surface'])
    for ob in sc.objects:
        if ob.type == 'MESH':
            if ob.name.startswith('_floor'):
                ob.hide_render = True
                continue
            for i in range(len(ob.material_slots)):
                ob.material_slots[i].material = hold
    sc.cycles.samples = 1
    tmp2 = out + '.lines.png'
    sc.render.filepath = tmp2
    bpy.ops.render.render(write_still=True)
    li = bpy.data.images.load(tmp2)
    la = np.array(li.pixels[:]).reshape(H * 2, W * 2, 4)[::-1]
    la = la.reshape(H, 2, W, 2, 4).mean((1, 3))
    a = np.clip(la[:, :, 3:4] * 1.6, 0, 1)
    comp = ref * (1 - a) + np.array([0.0, 1.0, 1.0]) * a
    for L in spec.get('lines', {}).values():
        L = np.asarray(L, float)
        for p0, p1 in zip(L[:-1], L[1:]):
            n = int(np.ceil(np.linalg.norm(p1 - p0))) + 1
            for t in np.linspace(0, 1, n):
                x, y = np.round(p0 + (p1 - p0) * t).astype(int)
                comp[max(y - 1, 0):y + 1, max(x - 1, 0):x + 1] = (1.0, 0.85, 0.0)
    save(comp, out.replace('.png', '_lines.png'))
