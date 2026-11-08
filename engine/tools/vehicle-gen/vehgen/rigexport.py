"""The design's rig data: what the physics rigger (engine-softbody
`authoring::gen2::vg_input`) needs besides the `.glb`.

    <design>.rig.json

* `dims`: wheelbase, axles, track, tyre and rim, the ends, the doors and the
  rear closure (boot lid or tailgate);
* `arches`: each wheel opening's centre (z, y) and radius;
* `keys`: the design's key lines (the tube's key points, left half) sampled
  along z, named by `KEYLINES` (sill, door bottom, widest, waist, shoulder,
  belt, roof rail, glass edge, centre);
* `zlines`, `vlines`: the layout's stations and girth lines;
* `sections`: the body tube's left half sampled on a (z, v) grid (the surface
  the rigger casts its section rays against; hole free, so a ray through a
  wheel opening finds the side, as on the Rust surface);
* `ends`: `front_z(x, y)` and `rear_z(x, y)` on an (x, y) grid (the nose and
  tail faces);
* `skins`: each part's welded skin boundary loops (`cb.skins`, before the
  shut-line inset), glTF metres, 0.1 mm;
* `markers`: the art spec's markers (position, axis);
* `physics`: the design's `PHYSICS` block (engine/docs/generated-vehicles.md);
  empty means the census defaults.

Frames are glTF: +Y up, +Z forward, +X left, metres, origin between the axles.
"""
import json
import numpy as np

# The ten key points' meanings (every design so far uses this order).
KEYLINES = {'floor': 0, 'sill': 1, 'door_bottom': 2, 'widest': 3, 'waist': 4,
            'shoulder': 5, 'belt': 6, 'roof_rail': 7, 'glass_edge': 8, 'centre': 9}

DZ = 0.02         # section and key-line spacing along z (m)
DV = 1.0 / 16     # section spacing in v (a sixteenth of a key segment)
DXY = 0.02        # end-face grid (m)


def _r(a, q=1e-4):
    """Round to q (0.1 mm) for a compact file."""
    return (np.round(np.asarray(a, float) / q) * q).round(6).tolist()


def export(D, cb, markers, path):
    tube = cb.tube
    zf, zr = float(D.ZFRONT), float(D.ZREAR)
    zs = np.arange(zr - 0.06, zf + 0.06 + DZ / 2, DZ)
    # key lines: (x, y) of every key point at every station
    kp = tube.keypoints(zs)                       # (nz, NK, 2)
    keys = {name: _r(kp[:, k, :]) for name, k in getattr(D, 'KEYLINES', KEYLINES).items()
            if k < tube.NK}
    # the tube's left half (v from the floor centre to the top centre)
    vs = np.arange(0.0, tube.NK - 1 + DV / 2, DV)
    Z, V = np.meshgrid(zs, vs, indexing='ij')
    P = tube.eval(Z.reshape(-1), V.reshape(-1)).reshape(len(zs), len(vs), 3)
    sections = _r(P[:, :, :2])
    # the nose and tail faces
    xs = np.arange(0.0, 1.2 + DXY / 2, DXY)
    ys = np.arange(0.0, 2.2 + DXY / 2, DXY)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    fz = np.asarray(D.front_z(X.reshape(-1), Y.reshape(-1)), float).reshape(X.shape)
    rz = np.asarray(D.rear_z(X.reshape(-1), Y.reshape(-1)), float).reshape(X.shape)
    # welded skins' boundary loops
    from .panels import boundary_loops
    skins = {}
    for name, sk in sorted(cb.skins.items()):
        loops = [_r(sk.v[lp]) for lp in boundary_loops(sk) if len(lp) >= 3]
        skins[name] = loops
    arches = []
    for a in (getattr(D, 'ARCH_F', None), getattr(D, 'ARCH_R', None)):
        if a:
            arches.append({'center': [float(a['center'][0]), float(a['center'][1])],
                           'R': float(a['R'])})
    rear = 'tailgate' if any(k.startswith('hinge_tailgate') for k in markers) else 'trunk_lid'
    dims = {
        'wheelbase': float(D.WHEELBASE), 'track': float(D.TRACK),
        'axle_f': float(D.ZF_AX), 'axle_r': float(D.ZR_AX),
        'tyre_d': float(D.TYRE_D), 'tyre_w': float(D.TYRE_W),
        'rim_d': float(D.RIM_D), 'wheel_y': float(D.WHEEL_Y),
        'z_front': zf, 'z_rear': zr,
        'doors': list(getattr(D, 'DOORS', ['FL', 'FR', 'RL', 'RR'])),
        'rear_closure': rear,
    }
    mk = {}
    for name, (pos, axis) in sorted(markers.items()):
        mk[name] = {'pos': _r(pos), 'axis': _r(axis)}
    phys = getattr(D, 'PHYSICS', {}) or {}
    out = {
        'format': 1,
        'design': D.NAME,
        'base': getattr(D, 'BASE', None),
        'title': getattr(D, 'TITLE', D.NAME),
        'dims': dims,
        'arches': arches,
        'keys': {'z0': float(zs[0]), 'dz': DZ, 'n': len(zs), 'lines': keys},
        'zlines': [[n, float(z)] for n, z in D.ZLINES],
        'vlines': [float(v) for v in D.VLINES],
        'sections': {'z0': float(zs[0]), 'dz': DZ, 'nz': len(zs), 'dv': DV, 'nv': len(vs),
                     'xy': sections},
        'ends': {'x0': 0.0, 'y0': 0.0, 'd': DXY, 'nx': len(xs), 'ny': len(ys),
                 'front_z': _r(fz), 'rear_z': _r(rz)},
        'skins': skins,
        'markers': mk,
        'physics': phys,
    }
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    return path
