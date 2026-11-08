"""Drag-strip gasser built from the 1969 fastback: the body raked nose-up on
skinny front tyres, fat rear slicks on wide rims, a supercharger standing
through the bonnet, no wing, flat primer paint. Derived from
designs/mach1_1969.py; only the differences are here.
"""
import numpy as np
from vehgen.design import derive

derive('mach1_1969', globals())

NAME = 'mach1_gasser'
TITLE = 'Rubble fastback gasser'
PAINT = (0.055, 0.058, 0.062)          # flat primer grey
PAINT_METALLIC = 0.0
PAINT_ROUGHNESS = 0.75
PAINT_COAT = 0.0

# the body and every part on it pitch nose-up about the rear axle and sit on
# the bigger rear tyres; the wheels themselves are rebuilt below
RAKE_DEG = 2.6
FRONT = dict(TYRE_D=0.620, TYRE_W=0.165, RIM_D=15 * 0.0254, RIM_W=4.5 * 0.0254)
REAR = dict(TYRE_D=0.720, TYRE_W=0.300, RIM_D=15 * 0.0254, RIM_W=10 * 0.0254)
_Y0 = WHEEL_Y                          # noqa: F821 (the body was built for these)
_LIFT = REAR['TYRE_D'] / 2 - _Y0


def _pose():
    from vehgen.mesh import rot_x
    R = rot_x(-np.radians(RAKE_DEG))
    p = np.array([0.0, _Y0, ZR_AX])     # noqa: F821
    t = p - R @ p + np.array([0, _LIFT, 0])
    return R, t


def _is_wheel(name):
    return name.split('_')[0] in ('rim', 'tyre') or name.startswith('brake_disc')


_base_parts = parts  # noqa: F821
_base_post = post  # noqa: F821


def parts(cb):
    from vehgen import parts as P
    from vehgen.mesh import Mesh, rbox, closed_lathe
    out, extras, mk = _base_parts(cb)
    tube = cb.tube
    g = globals()
    # wheels: skinny fronts, slicks behind, each on its own axle height
    for tag, z, spec in (('F', ZF_AX, FRONT), ('R', ZR_AX, REAR)):  # noqa: F821
        saved = {k: g.get(k, 'magnum') for k in list(spec) + ['WHEEL_Y', 'RIM_STYLE']}
        g.update(spec)
        g['WHEEL_Y'] = spec['TYRE_D'] / 2
        g['RIM_STYLE'] = 'steel' if tag == 'F' else 'magnum'
        D = __import__(__name__)
        for s, sg in (('L', 1), ('R', -1)):
            rim, ty, di, c = P.wheel(D, sg, z)
            out['rim_' + tag + s] = rim
            out['tyre_' + tag + s] = ty
            out['brake_disc_' + tag + s] = di
            mk['wheel_center_' + tag + s] = (c, (sg, 0, 0))
        g.update(saved)
    # no bonnet scoop, pins or wing
    extras['hood'] = Mesh()
    extras['trunk_lid'] = Mesh()
    # the blower: a ribbed case on a black manifold, standing through the
    # bonnet, a chrome bug-catcher scoop on top, drive pulleys and belt
    z0, z1 = 1.18, 1.62
    ys = []
    for z in np.linspace(z0, z1, 7):
        for x in (-0.15, 0.0, 0.15):
            v = tube.v_at_x(z, abs(x), 7.0, 9.0)[0]
            ys.append(float(tube.eval(z, v)[0][1]))
    yb = max(ys) + 0.006
    zc = (z0 + z1) / 2
    bl = Mesh()
    bl.add(rbox((0, yb + 0.012, zc), (0.15, 0.012, 0.23), 0.006, 1, 'trim_black'))
    bl.add(rbox((0, yb + 0.100, zc), (0.13, 0.076, 0.21), 0.035, 3, 'metal_bare'))
    for z in np.linspace(zc - 0.17, zc + 0.17, 8):
        bl.add(rbox((0, yb + 0.178, z), (0.10, 0.004, 0.006), 0.002, 1, 'metal_bare'))
    sc = Mesh()
    sc.add(rbox((0, yb + 0.240, zc + 0.02), (0.11, 0.058, 0.13), 0.018, 2, 'chrome'))
    sc.add(rbox((0, yb + 0.252, zc + 0.152), (0.096, 0.044, 0.002), 0.002, 1, 'trim_black'))
    bl.add(sc)
    for y, r in ((yb + 0.160, 0.045), (yb + 0.078, 0.056)):
        pul = closed_lathe([(0.004, 0.0), (r, 0.0), (r, 0.030), (0.004, 0.030)], 28, 'chrome')
        pul.transform(P.rot_to(np.array([0.0, 0.0, 1.0])), (0, y, z1 + 0.012))
        bl.add(pul)
    bl.add(rbox((0, yb + 0.118, z1 + 0.027), (0.060, 0.072, 0.016), 0.03, 2, 'rubber'))
    out['blower'] = bl
    # pose the markers with the body (the wheel centres are already placed)
    R, t = _pose()
    for k, (pos, xa) in list(mk.items()):
        if not k.startswith('wheel_center_'):
            mk[k] = (R @ np.asarray(pos, float) + t, xa)
    return out, extras, mk


def post(parts):
    _base_post(parts)
    R, t = _pose()
    for name, m in parts.items():
        if not _is_wheel(name):
            m.transform(R, t)
