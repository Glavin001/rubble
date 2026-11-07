"""1969 fastback ("SportsRoof") Mach 1, electric blue.

Profiles were read off side, front-3/4 and rear-3/4 photographs of real cars
and scaled to the published dimensions: wheelbase 108 in (2,743 mm), length
187.4 in (4,760 mm), width 71.3 in (1,811 mm), height ~51 in (1,300 mm),
track 58.5 in (1,486 mm). No badges or logos are modelled.

z is forward from the midpoint between the axles; y up from the ground; x to
the car's left. All tables are (z, value) stations.
"""
import numpy as np
from vehgen.curves import table, smoothstep, bump
from vehgen.tube import Tube, key

NAME = 'mach1_1969'
TITLE = '1969 Fastback Mach 1'
PAINT = (0.004, 0.045, 0.62)          # electric blue, linear RGB
PAINT_METALLIC = 0.45
PAINT_ROUGHNESS = 0.26

WHEELBASE = 2.743
TRACK = 1.486
ZF_AX = WHEELBASE / 2
ZR_AX = -WHEELBASE / 2
TYRE_D = 0.662                         # F70-14 class (about 26 in), measured off photos
TYRE_W = 0.235
RIM_D = 15 * 0.0254
RIM_W = 7 * 0.0254
WHEEL_Y = TYRE_D / 2

# --- plan width -----------------------------------------------------------------
_W = table([(2.32, 0.866), (2.24, 0.884), (2.05, 0.896), (1.37, 0.902),
            (0.60, 0.902), (-0.70, 0.904), (-1.37, 0.906), (-1.90, 0.901),
            (-2.20, 0.892), (-2.44, 0.874)])
_ZS = np.linspace(-2.5, 2.4, 50)


def _off(d):
    return [(z, float(_W(z)) - d) for z in _ZS]


def _shoulder_x():
    # the hip's top rolls inboard over the rear wheel
    hip = table([(-2.6, 0.040), (-1.55, 0.040), (-0.95, 0.007), (2.5, 0.007)])
    return [(z, float(_W(z)) - float(hip(z))) for z in _ZS]


# --- key points (x, y) of the left half section --------------------------------
K = [
    # 0 floor centre
    key(0.0, 0.215),
    # 1 rocker / sill bottom
    key(_off(0.050), [(2.4, 0.40), (1.80, 0.40), (1.0, 0.235), (-0.98, 0.235),
                      (-1.80, 0.33), (-2.5, 0.30)]),
    # 2 door bottom line / rocker top
    key(_off(0.016), [(2.4, 0.47), (1.80, 0.46), (1.0, 0.30), (-0.98, 0.30),
                      (-1.80, 0.48), (-2.5, 0.560)]),
    # 3 widest line
    key(_off(0.0), [(2.4, 0.56), (1.0, 0.52), (-0.98, 0.52), (-1.80, 0.62), (-2.5, 0.690)]),
    # 4 arch-top line (upper body side)
    key(_off(0.002), [(2.4, 0.70), (2.1, 0.71), (1.37, 0.74), (0.6, 0.72), (-0.7, 0.73),
                      (-1.37, 0.77), (-2.5, 0.830)]),
    # 5 shoulder crease
    key(_shoulder_x(), [(2.32, 0.79), (1.75, 0.815), (1.37, 0.83), (1.0, 0.845),
                      (0.63, 0.860), (-0.24, 0.872), (-0.715, 0.885), (-0.95, 0.908),
                      (-1.37, 0.922), (-1.77, 0.918), (-2.10, 0.910), (-2.45, 0.900)]),
    # 6 belt (window sill) / bonnet shut line / quarter top
    key([(2.32, 0.600), (2.15, 0.615), (1.95, 0.690), (1.75, 0.762), (1.37, 0.786),
         (0.63, 0.780), (0.55, 0.785),
         (-0.715, 0.800), (-1.37, 0.795), (-2.0, 0.780), (-2.45, 0.770)],
        [(2.32, 0.800), (1.75, 0.832), (1.37, 0.848), (0.63, 0.887), (0.55, 0.905),
         (0.12, 0.915), (-0.62, 0.930), (-0.715, 0.935), (-0.95, 0.955), (-1.37, 0.963),
         (-1.77, 0.952), (-2.10, 0.942), (-2.45, 0.935)]),
    # 7 daylight-opening top (roof rail outer) / bonnet line / sail
    key([(2.32, 0.500), (1.95, 0.580), (1.37, 0.620), (0.80, 0.650), (0.55, 0.715), (0.12, 0.640),
         (-0.24, 0.635), (-0.62, 0.640), (-0.90, 0.660), (-1.37, 0.700), (-1.77, 0.730),
         (-2.00, 0.660), (-2.20, 0.600), (-2.45, 0.570)],
        [(2.32, 0.803), (1.37, 0.859), (0.63, 0.894), (0.55, 0.918), (0.12, 1.215),
         (-0.24, 1.250), (-0.62, 1.228), (-0.90, 1.175), (-1.37, 1.040), (-1.77, 0.957),
         (-2.10, 0.947), (-2.45, 0.943)]),
    # 8 glass edge (windscreen / back light) / bonnet line / boot lid
    key([(2.32, 0.380), (1.95, 0.440), (1.37, 0.460), (0.80, 0.490), (0.63, 0.520), (0.55, 0.645), (0.12, 0.585),
         (-0.24, 0.580), (-0.62, 0.580), (-0.90, 0.585), (-1.37, 0.600), (-1.77, 0.630),
         (-2.00, 0.540), (-2.20, 0.490), (-2.45, 0.470)],
        [(2.32, 0.808), (1.37, 0.862), (0.63, 0.900), (0.55, 0.921), (0.12, 1.238),
         (-0.24, 1.272), (-0.62, 1.243), (-0.90, 1.192), (-1.37, 1.065), (-1.77, 0.958),
         (-2.10, 0.949), (-2.45, 0.944)]),
    # 9 top centre line
    key(0.0, [(2.32, 0.812), (2.10, 0.838), (1.75, 0.853), (1.37, 0.868), (1.00, 0.886),
              (0.63, 0.903), (0.55, 0.925), (0.12, 1.262), (-0.24, 1.300), (-0.62, 1.268),
              (-0.90, 1.215), (-1.37, 1.090), (-1.77, 0.975), (-2.10, 0.950), (-2.45, 0.945)]),
]

# the shoulder is a crisp crease along the front fender and door and rolls
# into a round hip over the rear wheel
_HIP = table([(-2.6, 1.0), (-1.55, 1.0), (-0.95, 0.0), (2.5, 0.0)])
HANDLES = [(1 / 3, 1 / 3), (0.30, 0.30), (0.33, 0.33), (0.36, 0.36), (0.33, 0.33),
           (lambda z: 0.14 + 0.20 * _HIP(z), lambda z: 0.12 + 0.24 * _HIP(z)),
           (lambda z: 0.22 + 0.10 * _HIP(z), 0.25), (0.33, 0.33), (0.33, 0.33), (1 / 3, 1 / 3)]

ARCH_F = dict(center=(ZF_AX, WHEEL_Y), R=0.380)
ARCH_R = dict(center=(ZR_AX, WHEEL_Y), R=0.392)


def _arch_lip(P, N, z, v):
    """A rolled lip round each wheel opening."""
    out = np.zeros_like(P)
    side = np.abs(N[:, 0]) > 0.5
    for a in (ARCH_F, ARCH_R):
        zc, yc = a['center']
        r = np.hypot(z - zc, P[:, 1] - yc)
        lip = 0.007 * bump(r, a['R'] + 0.012, 0.045) * (P[:, 1] > yc - 0.1)
        out[:, 0] += np.sign(P[:, 0]) * lip * side
    return out


def _bonnet_bulge(P, N, z, v):
    """The raised centre of the bonnet, fading out at the cowl and the nose."""
    out = np.zeros_like(P)
    ax = np.abs(P[:, 0])
    w = smoothstep(0.42, 0.34, ax) * smoothstep(0.62, 0.75, z) * smoothstep(2.32, 2.20, z)
    out[:, 1] = 0.012 * w * (np.abs(N[:, 1]) > 0.6)
    return out


def _quarter_hip(P, N, z, v):
    """The '69 rear quarter swells out over the rear wheel."""
    out = np.zeros_like(P)
    side = np.abs(N[:, 0]) > 0.3
    w = bump(z, -1.45, 0.75) * bump(P[:, 1], 0.70, 0.30)
    out[:, 0] = np.sign(P[:, 0]) * 0.008 * w * side
    return out


def _side_sculpt(P, N, z, v):
    """The '69 body side: a crisp light-catching ledge under the shoulder
    from the headlamp bucket back to the quarter scoop, the panel below it
    drawn in slightly."""
    out = np.zeros_like(P)
    side = np.clip((np.abs(N[:, 0]) - 0.6) / 0.3, 0, 1)
    zw = np.clip((z + 1.05) / 0.25, 0, 1) * np.clip((2.30 - z) / 0.15, 0, 1)
    y = P[:, 1]
    prof = 0.004 * bump(y, 0.775, 0.045) - 0.006 * bump(y, 0.66, 0.11)
    out[:, 0] = np.sign(P[:, 0]) * prof * side * zw
    return out


def tube():
    t = Tube(K, HANDLES, displace=[_arch_lip, _bonnet_bulge, _quarter_hip, _side_sculpt])
    return t


# --- layout ----------------------------------------------------------------------
ZLINES = [('F', 2.31), ('AF1', 1.76), ('AF2', ZF_AX), ('AF3', 0.99), ('DF', 0.63),
          ('WB', 0.55), ('WT', 0.12), ('GR', -0.62), ('DR', -0.715), ('AR1', -0.97),
          ('AR2', ZR_AX), ('AR3', -1.77), ('T', -2.43)]
VLINES = [0, 1, 2, 3, 4, 5, 5.33, 5.67, 6, 7, 8, 9]


def front_z(x, y):
    x = np.abs(x)
    lean = 0.095 * np.clip((0.80 - y) / 0.40, 0, 1) ** 1.3
    corner = 0.050 * (x / 0.87) ** 6
    vee = 0.012 * (1 - np.clip(x / 0.85, 0, 1) ** 2)
    return 2.31 - lean - corner + vee


def rear_z(x, y):
    x = np.abs(x)
    dish = 0.022 * (1 - np.clip(x / 0.80, 0, 1) ** 2)      # concave tail panel
    tuck = 0.035 * np.clip((0.52 - y) / 0.18, 0, 1)
    corner = 0.110 * np.clip(x / 0.88, 0, 1.05) ** 4          # corners roll round
    return -2.43 + dish + tuck + corner


HEADLIGHT = dict(x=0.748, y=0.645, R=0.088)
HEADLIGHT_INNER = dict(x=0.500, y=0.640, R=0.058)
TAILLIGHT = dict(x0=0.40, x1=0.64, y0=0.690, y1=0.835)
GRILLE_X = 0.62


def assign(zf, zt, lo, hi, side):
    """Part of the tube block between z-line zf..zt and left v-lines lo..hi.
    v-line indices: 0 floor, 1 rocker, 2 door bottom, 3 widest, 4 arch line,
    5 shoulder, 6-7 shoulder strip, 8 belt, 9 roof rail, 10 glass edge,
    11 centre."""
    s = side
    I = zf
    front = I in ('F', 'AF1', 'AF2', 'AF3')
    cabin = I in ('DF', 'WB', 'WT', 'GR')
    if lo == 0:
        return 'body_shell' if I in ('DF', 'WB', 'WT', 'GR', 'DR') else None
    if lo <= 7:
        if front:
            return 'fender_F' + s
        if cabin:
            return 'body_shell' if lo == 1 else 'door_F' + s
        return 'body_shell'
    if front:
        return 'hood'
    if I == 'AR3':
        return 'trunk_lid'
    if I == 'DF':
        return 'body_shell'
    if lo == 8:
        if I in ('WB', 'WT'):
            return 'glass_door_F' + s
        if I == 'GR':
            return 'glass_quarter_' + s
        return 'body_shell'
    if lo == 10:
        if I == 'WB':
            return 'glass_windshield'
        if I in ('DR', 'AR1', 'AR2'):
            return 'glass_rear'
    return 'body_shell'


def build_layout(cb):
    """Overrides, arches and end fits on a CarBody (before nodes are built)."""
    zi = cb.zi
    tube = cb.tube
    # back light top and quarter glass rear edge
    i = zi['DR']
    for jl, z in ((9, -0.88), (10, -0.89), (11, -0.90)):
        cb.set_z(i, jl, z)
    i0 = zi['F']
    cb.end_fit(i0, front_z)
    cb.end_fit(zi['T'], rear_z)
    cb.arch(ARCH_F['center'], ARCH_F['R'], (zi['AF1'], zi['AF2'], zi['AF3']), [1, 2, 3, 4],
            [12, 21, 40, 62, 90, 118, 145, 172, 193.5], (0.6, 5.2))
    cb.arch(ARCH_R['center'], ARCH_R['R'], (zi['AR1'], zi['AR2'], zi['AR3']), [1, 2, 3, 4],
            [-13.4, 8, 35, 62, 90, 118, 145, 160, 175], (0.6, 5.2))


def build_caps(cb):
    from vehgen.carbody import circle_arc
    zi = cb.zi
    i0, iT = zi['F'], zi['T']
    J = cb.J
    hl = HEADLIGHT
    d = hl['R'] * np.sqrt(0.5)
    R = hl['R']
    for s in ('L', 'R'):
        sg = 1 if s == 'L' else -1
        n = (lambda jl, s=s: cb.nid(i0, cb.j(jl, s)))
        P = {k: cb.L.node_pos(n(k)) for k in range(1, 9)}
        xr = P[8][0]
        left = [n(k) for k in range(1, 6)]
        right = [(xr, P[k][1]) for k in range(1, 5)] + [n(8)]
        bottom = [n(1), (P[6][0], P[1][1]), (P[7][0], P[1][1]), (xr, P[1][1])]
        top = [n(5), n(6), n(7), n(8)]
        xc, yc = sg * hl['x'], hl['y']
        ov = {(2, 1): (xc + sg * d, yc - d), (2, 2): (xc - sg * d, yc - d),
              (3, 1): (xc + sg * d, yc + d), (3, 2): (xc - sg * d, yc + d)}
        if s == 'L':
            arcs = {((2, 1), (2, 2)): circle_arc(xc, yc, R, -45, -135),
                    ((2, 2), (3, 2)): circle_arc(xc, yc, R, -135, -225),
                    ((3, 1), (3, 2)): circle_arc(xc, yc, R, 45, 135),
                    ((2, 1), (3, 1)): circle_arc(xc, yc, R, -45, 45)}
        else:
            arcs = {((2, 1), (2, 2)): circle_arc(xc, yc, R, -135, -45),
                    ((2, 2), (3, 2)): circle_arc(xc, yc, R, -45, 45),
                    ((3, 1), (3, 2)): circle_arc(xc, yc, R, 135, 45),
                    ((2, 1), (3, 1)): circle_arc(xc, yc, R, -135, -225)}
        cb.cap('capF' + s, front_z, 1, left, right, bottom, top, 'fender_F' + s,
               holes={(2, 1)}, overrides=ov, arcs=arcs)
    # tail panel: one cap across the back, taillight openings left open
    tl = TAILLIGHT
    nT = (lambda j: cb.nid(iT, j % J))
    left = [nT(k) for k in range(1, 6)]
    right = [nT(J - k) for k in range(1, 6)]
    y0 = cb.L.node_pos(nT(1))[1]
    xk1 = cb.L.node_pos(nT(1))[0]
    xs = [tl['x1'], tl['x0'], 0.0, -tl['x0'], -tl['x1']]
    lid = []
    for x in xs:
        v = cb.tube.v_at_x(-2.43, abs(x), 6.0, 9.0)[0]
        lid.append(cb.tube.eval(-2.43, v)[0][1])
    DROP = 0.0125
    lid = [y - DROP for y in lid]
    # the tail panel tucks 7 mm under the lid's rear edge all the way to the
    # corners (the lid overhangs it), so its top-row corners are its own
    pk = cb.L.node_pos(nT(8))
    top = ([nT(5), nT(6), nT(7), (pk[0], pk[1] - DROP)] + [(x, y) for x, y in zip(xs, lid)]
           + [(-pk[0], pk[1] - DROP), nT(J - 7), nT(J - 6), nT(J - 5)])
    fan = [xk1 - (xk1 - tl['x1']) * k / 4 for k in (1, 2, 3)]
    bottom = ([nT(1)] + [(x, y0) for x in fan] + [(x, y0) for x in xs]
              + [(-x, y0) for x in reversed(fan)] + [nT(J - 1)])
    ov = {(2, 4): (tl['x1'], tl['y0']), (2, 5): (tl['x0'], tl['y0']),
          (3, 4): (tl['x1'], tl['y1']), (3, 5): (tl['x0'], tl['y1']),
          (2, 7): (-tl['x0'], tl['y0']), (2, 8): (-tl['x1'], tl['y0']),
          (3, 7): (-tl['x0'], tl['y1']), (3, 8): (-tl['x1'], tl['y1'])}
    # the tail panel's top edges follow the boot lid's rear edge
    lidp = [cb.L.nodes[nT(j)][1] for j in range(8, J - 7)]
    lidp = np.array(lidp)
    dense = []
    for a_, b_ in zip(lidp[:-1], lidp[1:]):
        for t in np.linspace(0, 1, 16, endpoint=False):
            dense.append(a_ + (b_ - a_) * t)
    dense.append(lidp[-1])
    LP = cb.tube.eval(np.array(dense)[:, 0], np.array(dense)[:, 1])
    o = np.argsort(LP[:, 0])
    lx, ly = LP[o, 0], LP[o, 1]

    def lid_curve(xa, xb):
        def f(t):
            x = xa + (xb - xa) * t
            return np.stack([x, np.interp(x, lx, ly) - DROP], 1)
        return f
    xk = cb.L.node_pos(nT(8))[0]
    tx = [xk] + xs + [-xk]
    arcs = {((4, 3 + k), (4, 4 + k)): lid_curve(tx[k], tx[k + 1]) for k in range(len(tx) - 1)}
    cb.cap('capR', rear_z, -1, left, right, bottom, top, 'body_shell',
           holes={(2, 4), (2, 7)}, overrides=ov, arcs=arcs)


# --- everything that is not a stamped panel --------------------------------------
DOORS = ['FL', 'FR']
RED = (0.80, 0.03, 0.02, 1.0)
CLEAR = (1.0, 1.0, 1.0, 1.0)
AMBER = (1.0, 0.45, 0.05, 1.0)


from vehgen.parts import molding as _molding, surface_bump as _surface_bump  # noqa: E402


def parts(cb):
    """Returns (new_parts, extras, markers). extras: {existing part: Mesh} to
    merge into a panel (scoops, trim). markers: {name: (pos, xaxis)}."""
    from vehgen import parts as P
    from vehgen.mesh import (Mesh, rbox, box, sweep, rrect, circle, closed_lathe,
                             orient_outward, catmull, grid, rot_x, rot_y)
    from vehgen.panels import offset_copy, inset_boundary
    tube = cb.tube
    out, extras, mk = {}, {}, {}

    # wheels
    for tag, z in (('F', ZF_AX), ('R', ZR_AX)):
        for s, sg in (('L', 1), ('R', -1)):
            rim, ty, di, c = P.wheel(__import__(__name__), sg, z)
            out['rim_' + tag + s] = rim
            out['tyre_' + tag + s] = ty
            out['brake_disc_' + tag + s] = di
            mk['wheel_center_' + tag + s] = (c, (sg, 0, 0))

    # headlamps: outer in the fender ends, inner in the grille
    zg = float(front_z(np.array([0.60]), np.array([0.65]))[0]) - 0.040
    for s, sg in (('L', 1), ('R', -1)):
        hl = HEADLIGHT
        zs = float(front_z(np.array([hl['x']]), np.array([hl['y']]))[0])
        nrm = P.graph_normal(front_z, hl['x'], hl['y'], 1) * [sg, 1, 1]
        # the lamp sits at the bottom of a scooped bucket under the fender brow
        DEEP = 0.042
        cen = np.array([sg * hl['x'], hl['y'], zs]) - nrm * DEEP
        l1, h1 = P.round_lamp(cen, hl['R'] - 0.020, tint=CLEAR, axis=nrm)
        Rt = P.rot_to(nrm)
        rings = []
        for t, rr, dd in ((0.0, hl['R'] - 0.003, 0.007), (0.5, hl['R'] - 0.007, 0.024),
                          (1.0, hl['R'] - 0.012, DEEP - 0.010)):
            ring = []
            for a_ in np.linspace(0, 2 * np.pi, 49)[:-1]:
                off = Rt @ np.array([0.0, rr * np.cos(a_), rr * np.sin(a_)])
                q = np.array([sg * hl['x'], hl['y'], 0.0]) + off
                zq = float(front_z(np.array([abs(q[0])]), np.array([q[1]]))[0])
                q[2] = zq
                ring.append(q - nrm * dd)
            rings.append(ring)
        bucket = grid(np.array(rings), mat='paint', closed_v=True)
        cn = bucket.vertex_normals().mean(0)
        if np.dot(cn, nrm) < 0:
            bucket.flip()
        extras.setdefault('fender_F' + s, Mesh()).add(bucket)
        hi = HEADLIGHT_INNER
        l2, h2 = P.round_lamp((sg * hi['x'], hi['y'], zg + 0.020), hi['R'], tint=CLEAR, depth=0.014)
        # side markers: amber at the front of the fender, red at the tail
        fm, fn = P.surface_point(tube, 2.06, 0.560, sg)
        mk_ = rbox((0, 0, 0), (0.004, 0.016, 0.034), 0.003, 1, 'lamp_lens')
        mk_.transform(P.rot_to(fn) @ rot_y(0), fm + fn * 0.005)
        mk_.tint(AMBER)
        out['headlight_%s_lens' % s] = l1.add(l2).add(mk_)
        out['headlight_%s_housing' % s] = h1.add(h2)

    # grille: black egg-crate in a thin bright surround, recessed between the
    # fender ends, closed behind by a back plate
    g = Mesh()
    gx, gy0, gy1 = 0.584, 0.505, 0.770
    loop = np.array([[gx, gy0], [gx, gy1], [-gx, gy1], [-gx, gy0]])
    from vehgen.mesh import rrect as _rr
    outline = _rr(2 * gx, gy1 - gy0, 0.035, 5)
    path = np.c_[outline[:, 0], outline[:, 1] + (gy0 + gy1) / 2, np.full(len(outline), zg + 0.004)]
    g.add(orient_outward(sweep(path, rrect(0.010, 0.022, 0.004, 2), mat='chrome', closed=True,
                               up=(0, 0, 1))))
    g.add(box((0, (gy0 + gy1) / 2, zg - 0.045), (gx - 0.004, (gy1 - gy0) / 2 - 0.004, 0.003), 'trim_black'))
    xs = np.arange(-0.420, 0.4201, 0.035)
    for x in xs:
        g.add(box((x, (gy0 + gy1) / 2, zg - 0.018), (0.0022, (gy1 - gy0) / 2 - 0.012, 0.022), 'trim_black'))
    for y in np.linspace(gy0 + 0.04, gy1 - 0.04, 6):
        g.add(box((0, y, zg - 0.018), (0.43, 0.0022, 0.022), 'trim_black'))
    # solid black end panels round the inner lamps
    for sg in (1, -1):
        g.add(box((sg * 0.505, (gy0 + gy1) / 2, zg - 0.006), (0.070, (gy1 - gy0) / 2 - 0.010, 0.004), 'trim_black'))
    out['grille'] = g

    # front bumper: chrome blade wrapping round to the wheel openings, painted
    # valance and a black chin spoiler below it; paths follow the body outline
    def full(Pxz):
        R_ = (Pxz * [-1, 1])[::-1]
        return np.concatenate([R_[:-1], Pxz])

    yb = 0.452
    ols = [full(P.plan_outline(tube, front_z, y, 1, 2.3, side_len=0.14)) for y in (0.40, 0.452, 0.50)]
    pp = P.fit_path(ols, 0.022, 1, step=0.018)
    path = np.c_[pp[:, 0], np.full(len(pp), yb), pp[:, 1]]
    bf = orient_outward(sweep(path, rrect(0.032, 0.084, 0.016, 4), mat='chrome', up=(0, 1, 0)))
    ols = [full(P.plan_outline(tube, front_z, 0.40, 1, 2.3, side_len=0.06))]
    pv = P.fit_path(ols, -0.022, 1, step=0.018)
    pv = pv[4:-4]
    vpath = np.c_[pv[:, 0], np.full(len(pv), 0.0), pv[:, 1]]
    vsec = np.array([[0.004, 0.392], [-0.026, 0.392], [-0.046, 0.310], [-0.058, 0.282],
                     [-0.040, 0.280], [-0.014, 0.310]])
    bf.add(orient_outward(sweep(vpath, P.densify(vsec, 0.02), mat='paint', up=(0, 1, 0))))
    csec = np.array([[0.070, 0.272], [-0.050, 0.272], [-0.050, 0.260], [0.070, 0.262]])
    k = len(vpath) // 8
    bf.add(orient_outward(sweep(vpath[k:-k] * [0.97, 1, 1], csec, mat='trim_black', up=(0, 1, 0))))
    out['bumper_front'] = bf
    for k_, x in enumerate((-0.62, -0.30, 0.30, 0.62)):
        zz = float(front_z(np.array([abs(x)]), np.array([yb]))[0])
        mk['bolt_bumper_front_%d' % (k_ + 1)] = ((x, yb, zz - 0.004), (1, 0, 0))

    # rear bumper
    yr = 0.600
    ols = [full(P.plan_outline(tube, rear_z, y, -1, -2.4, side_len=0.18)) for y in (0.56, 0.60, 0.64)]
    pr = P.fit_path(ols, 0.023, -1, step=0.018)[::-1]
    path = np.c_[pr[:, 0], np.full(len(pr), yr), pr[:, 1]]
    out['bumper_rear'] = orient_outward(sweep(path, rrect(0.040, 0.088, 0.018, 4), mat='chrome', up=(0, 1, 0)))
    for k_, x in enumerate((-0.62, -0.30, 0.30, 0.62)):
        zz = float(rear_z(np.array([abs(x)]), np.array([yr]))[0])
        mk['bolt_bumper_rear_%d' % (k_ + 1)] = ((x, yr, zz + 0.004), (1, 0, 0))

    # taillamps: three lenses each side in a bright bezel, a reflector box behind
    tl = TAILLIGHT
    for s, sg in (('L', 1), ('R', -1)):
        xm = (tl['x0'] + tl['x1']) / 2
        ym = (tl['y0'] + tl['y1']) / 2
        zt = float(rear_z(np.array([xm]), np.array([ym]))[0])
        w = tl['x1'] - tl['x0']
        hgt = tl['y1'] - tl['y0']
        # three tall rounded lamps, each in its own bright bezel, on a black
        # backing that fills the opening
        lens = Mesh()
        hous = Mesh()
        gapx = 0.012
        lw = (w - 0.020 - 2 * gapx) / 3
        hous.add(box((sg * xm, ym, zt + 0.046), (w / 2 - 0.005, hgt / 2 - 0.005, 0.006), 'trim_black'))
        for k in range(3):
            x = tl['x0'] + 0.010 + lw / 2 + k * (lw + gapx)
            zk = float(rear_z(np.array([x]), np.array([ym]))[0])
            lh = hgt - 0.016
            lens.add(rbox((sg * x, ym, zk + 0.016), (lw / 2 - 0.007, lh / 2 - 0.008, 0.006),
                          min(lw / 2 - 0.008, 0.022), 3, 'lamp_lens'))
            rr = rrect(lw - 0.002, lh + 0.002, min(lw / 2 - 0.002, 0.024), 4)
            path = np.c_[sg * x + rr[:, 0], ym + rr[:, 1], np.full(len(rr), zk + 0.008)]
            hous.add(orient_outward(sweep(path, rrect(0.003, 0.010, 0.0013, 1), mat='chrome', closed=True,
                                          up=(0, 0, 1))))
            hous.add(box((sg * x, ym, zk + 0.032), (lw / 2 - 0.006, lh / 2 - 0.006, 0.003), 'lamp_emissive'))
        lens.tint(RED)
        rm, rn = P.surface_point(tube, -2.20, 0.690, sg)
        mr = rbox((0, 0, 0), (0.004, 0.014, 0.030), 0.003, 1, 'lamp_lens')
        mr.transform(P.rot_to(rn), rm + rn * 0.005)
        lens.add(mr.tint(RED))
        out['taillight_%s_lens' % s] = lens
        out['taillight_%s_housing' % s] = hous

    # mirrors, handles, wipers
    for s, sg in (('L', 1), ('R', -1)):
        v = 5.75 if sg > 0 else tube.VM - 5.75
        base = tube.eval(0.40, v)[0]
        out['mirror_' + s] = P.sport_mirror(base, sg)
        mk['mount_mirror_' + s] = (base, (sg, 0, 0))
        out['handle_F' + s] = P.door_handle(tube, -0.53, 0.826, sg)
    for s, x0, x1 in (('L', 0.56, 0.10), ('R', 0.02, -0.44)):
        zs = 0.495
        pts = []
        for x in np.linspace(x0, x1, 8):
            v = tube.v_at_x(zs, abs(x), 7.9, 9.0)[0]
            if x < 0:
                v = tube.VM - v
            Pp, Nn = tube.eval(zs, v, with_normal=True)
            pts.append(Pp[0] + Nn[0] * 0.004)
        out['wiper_' + s] = P.wiper(np.array(pts))

    # body extras: quarter scoops, rocker mouldings, window reveals, fuel cap
    bx = Mesh()
    for sg in (1, -1):
        bx.add(_surface_bump(tube, -0.735, -1.00, 0.775, 0.925, 0.030, sg))
        # the scoop's dark mouth facing forward
        from vehgen.parts import surface_point as _sp
        Q, Nq = _sp(tube, -0.742, 0.850, sg)
        Nq = Nq / np.linalg.norm(Nq)
        mouth = rbox((0, 0, 0), (0.011, 0.058, 0.0015), 0.0014, 1, 'trim_black')
        mouth.transform(None, Q + Nq * 0.016)
        bx.add(mouth)
        sv = np.linspace(0.52, -0.66, 30)
        Pp, Nn = P.path_on_surface(tube, sv, 0.285, sg, off=0.004)
        bx.add(orient_outward(sweep(Pp, rrect(0.005, 0.016, 0.002, 1), mat='chrome', twist_up=None)))
    VM = tube.VM
    for side in (1, -1):
        f = (lambda v: v) if side > 0 else (lambda v: VM - v)
        drip = [(0.53, f(7.25)), (0.12, f(7.25)), (-0.30, f(7.25)), (-0.62, f(7.25)), (-0.86, f(7.25))]
        bx.add(_molding(tube, drip, off=0.003, sec=(0.008, 0.005)))
    ws = ([(0.55, 8.0), (0.335, 8.0), (0.12, 8.0), (0.12, 9.0), (0.12, 10.0), (0.335, 10.0),
           (0.55, 10.0), (0.55, 9.0)])
    bx.add(_molding(tube, ws, off=0.002, sec=(0.010, 0.005), closed=True, n=6))
    zDR = [-0.89, -0.90, -0.89]
    rw = ([(-0.89, 8.0), (-1.33, 8.0), (-1.742, 8.0), (-1.742, 9.0), (-1.742, 10.0), (-1.33, 10.0),
           (-0.89, 10.0), (-0.90, 9.0)])
    bx.add(_molding(tube, rw, off=0.002, sec=(0.010, 0.005), closed=True, n=6))
    zt = float(rear_z(np.array([0.0]), np.array([0.735]))[0])
    cap = closed_lathe([(0.0, 0.012), (0.045, 0.010), (0.058, 0.0), (0.056, -0.004), (0.0, -0.004)], 40, 'chrome')
    cap.transform(rot_y(np.pi / 2), (0, 0.735, zt - 0.004))
    bx.add(cap)
    extras.setdefault('body_shell', Mesh()).add(bx)

    # bright lip mouldings round the wheel openings (fender front, body rear)
    for (arch, owner) in ((ARCH_F, 'fender_F'), (ARCH_R, 'body_shell')):
        zc, yc = arch['center']
        Ra = arch['R'] + 0.004
        for s, sg in (('L', 1), ('R', -1)):
            lo_y = 0.30 if owner == 'fender_F' else 0.26
            a0 = np.arcsin(np.clip((lo_y - yc) / Ra, -1, 1))
            angs = np.linspace(a0, np.pi - a0, 60)
            zs_ = zc + Ra * np.cos(angs)
            ys_ = yc + Ra * np.sin(angs)
            Pp, Nn = P.path_on_surface(tube, zs_, ys_, sg, off=0.004)
            mold = orient_outward(sweep(Pp, rrect(0.004, 0.012, 0.0018, 1), mat='chrome', twist_up=Nn))
            key_ = owner + (s if owner == 'fender_F' else '')
            extras.setdefault(key_, Mesh()).add(mold)

    # bonnet: scoop and pins
    hx = Mesh()
    rings = []
    for t in np.linspace(0, 1, 12):
        z = 1.10 + 0.32 * t
        hgt = 0.010 + 0.042 * t ** 1.2
        wdt = 0.15 + 0.025 * t
        yb_ = float(tube.eval(z, 9.0)[0][1]) + 0.010
        ring = []
        for a in np.linspace(0, np.pi, 11):
            ring.append([wdt * np.cos(a), yb_ + hgt * np.sin(a) ** 0.6, z])
        ring += [[-wdt * 0.98, yb_ - 0.006, z], [wdt * 0.98, yb_ - 0.006, z]]
        rings.append(ring)
    from vehgen.mesh import loft
    hx.add(orient_outward(loft(np.array(rings), mat='trim_black')))
    mouth = box((0, float(tube.eval(1.42, 9.0)[0][1]) + 0.034, 1.424), (0.15, 0.016, 0.002), 'interior_plastic')
    hx.add(mouth)
    for sg in (1, -1):
        z = 2.12
        v = tube.v_at_x(z, 0.47, 6.0, 9.0)[0]
        if sg < 0:
            v = VM - v
        Pp = tube.eval(z, v)[0]
        pin = closed_lathe([(0.0, 0.016), (0.010, 0.016), (0.012, 0.010), (0.012, -0.004), (0.0, -0.004)], 16, 'chrome')
        pin.transform(rot_x(0), (0, 0, 0))
        pin.v = pin.v[:, [1, 0, 2]]
        pin.flip()
        pin.transform(None, Pp + [0, 0.001, 0])
        hx.add(orient_outward(pin))
    extras['hood'] = hx

    # the optional rear wing: two black uprights on the lid, a cambered blade
    wx = Mesh()
    zw = -2.27
    for sg in (1, -1):
        v = tube.v_at_x(zw, 0.72, 6.0, 9.0)[0]
        if sg < 0:
            v = VM - v
        Q = tube.eval(zw, v)[0]
        up = rbox((0, 0, 0), (0.010, 0.030, 0.045), 0.007, 2, 'trim_black')
        up.transform(rot_x(np.radians(8)), Q + [0, 0.032, -0.005])
        wx.add(up)
    yb = float(tube.eval(zw, 9.0)[0][1]) + 0.068
    blade = []
    for x in np.linspace(-0.80, 0.80, 33):
        droop = 0.030 * max(0.0, (abs(x) - 0.62) / 0.18) ** 2
        blade.append([x, yb + 0.006 * (1 - (x / 0.8) ** 2) - droop, zw - 0.005])
    sec = np.array([[0.065, 0.004], [0.030, 0.012], [-0.040, 0.010], [-0.065, -0.004],
                    [-0.040, -0.006], [0.030, -0.004]])
    wb = orient_outward(sweep(np.array(blade), sec, mat='trim_black', up=(0, 1, 0)))
    wb.v = wb.v  # chord along z: sweep side = T x up = x x y = +z
    wx.add(wb)
    extras['trunk_lid'] = wx

    # interior
    hf = 0.255
    for s, sg in (('L', 1), ('R', -1)):
        out['seat_F' + s] = P.bucket_seat(sg * 0.37, -0.34, hf)
        mk['seat_H_point_F' + s] = ((sg * 0.37, hf + 0.155, -0.47), (1, 0, 0))
    rs = Mesh()
    rs.add(rbox((0, 0.40, -1.00), (0.50, 0.065, 0.20), 0.05, 3, 'interior_fabric'))
    back = rbox((0, 0, 0), (0.50, 0.22, 0.05), 0.05, 3, 'interior_fabric')
    back.transform(rot_x(np.radians(-38)), (0, 0.67, -1.28))
    rs.add(back)
    out['seat_rear'] = rs
    dash = Mesh()
    dash.add(rbox((0, 0.735, 0.42), (0.74, 0.115, 0.115), 0.05, 3, 'interior_plastic'))
    for sg in (1, -1):
        dash.add(rbox((sg * 0.37, 0.86, 0.36), (0.20, 0.035, 0.07), 0.03, 2, 'interior_plastic'))
    for k in range(2):
        g_ = closed_lathe([(0.0, 0.002), (0.040, 0.002), (0.040, -0.004), (0.0, -0.004)], 24, 'chrome')
        g_.transform(rot_y(-np.pi / 2), (0.37 + (k - 0.5) * 0.10, 0.80, 0.302))
        dash.add(g_)
    out['dash'] = dash
    out['steering_wheel'] = P.steering_wheel((0.37, 0.80, 0.13), R=0.19, tilt_deg=24, column=0.17)
    out['console'] = rbox((0, 0.405, -0.10), (0.09, 0.10, 0.40), 0.03, 2, 'interior_plastic')
    from vehgen.panels import inner_panel, boundary_loops
    for s in ('L', 'R'):
        sk = cb.skins['door_F' + s].copy()
        isb = np.zeros(len(sk.v), bool)
        for lp in boundary_loops(sk):
            isb[lp] = True
        out['door_trim_F' + s] = inner_panel(sk, isb, depth=0.085, rings=2, mat='interior_plastic',
                                             island=-10**6)
    zz = np.linspace(0.06, -0.84, 46)
    vv = np.linspace(7.35, VM - 7.35, 60)
    Z, V = np.meshgrid(zz, vv, indexing='ij')
    Pp, Nn = tube.eval(Z.reshape(-1), V.reshape(-1), with_normal=True)
    Pp = (Pp - Nn * 0.040).reshape(len(zz), len(vv), 3)
    hl_ = grid(Pp, mat='interior_fabric')
    hl_.flip()
    if np.mean(hl_.vertex_normals()[:, 1]) > 0:
        hl_.flip()
    out['headliner'] = hl_

    # engine bay and underbody
    eb = Mesh()
    eb.add(rbox((0, 0.43, 1.30), (0.21, 0.15, 0.30), 0.03, 2, 'metal_bare'))
    for sg in (1, -1):
        hd = rbox((0, 0, 0), (0.07, 0.06, 0.29), 0.02, 2, 'metal_bare')
        hd.transform(rot_y(0) @ np.array([[np.cos(sg * 0.6), -np.sin(sg * 0.6), 0], [np.sin(sg * 0.6), np.cos(sg * 0.6), 0], [0, 0, 1]]),
                     (sg * 0.22, 0.60, 1.30))
        eb.add(hd)
        vc = rbox((0, 0, 0), (0.05, 0.025, 0.27), 0.015, 2, 'chrome')
        vc.transform(np.array([[np.cos(sg * 0.6), -np.sin(sg * 0.6), 0], [np.sin(sg * 0.6), np.cos(sg * 0.6), 0], [0, 0, 1]]),
                     (sg * 0.26, 0.665, 1.30))
        eb.add(vc)
    eb.add(rbox((0, 0.61, 1.30), (0.14, 0.05, 0.24), 0.02, 2, 'metal_bare'))
    ac = closed_lathe([(0.0, 0.05), (0.18, 0.05), (0.19, 0.03), (0.19, -0.03), (0.18, -0.04), (0.0, -0.04)], 48, 'chrome')
    ac.v = ac.v[:, [1, 0, 2]]
    ac.flip()
    ac.transform(None, (0, 0.715, 1.30))
    eb.add(orient_outward(ac))
    eb.add(rbox((0, 0.25, 1.30), (0.17, 0.05, 0.27), 0.02, 2, 'metal_bare'))
    fan = closed_lathe([(0.0, 0.01), (0.17, 0.01), (0.17, -0.01), (0.0, -0.01)], 32, 'trim_black')
    fan.transform(rot_y(-np.pi / 2), (0, 0.48, 1.66))
    eb.add(fan)
    out['engine_block'] = eb
    rad = Mesh()
    rad.add(box((0, 0.56, 2.06), (0.37, 0.17, 0.028), 'trim_black'))
    rad.add(box((0, 0.745, 2.06), (0.385, 0.016, 0.034), 'metal_bare'))
    rad.add(box((0, 0.375, 2.06), (0.385, 0.016, 0.034), 'metal_bare'))
    out['radiator'] = rad
    out['firewall'] = box((0, 0.57, 0.625), (0.60, 0.27, 0.003), 'trim_black')
    ub = Mesh()
    ub.add(box((0, 0.165, 1.56), (0.44, 0.003, 0.41), 'trim_black'))
    ub.add(box((0, 0.375, -2.06), (0.48, 0.004, 0.27), 'trim_black'))
    ub.add(rbox((0, 0.43, -1.98), (0.42, 0.05, 0.20), 0.02, 2, 'metal_bare'))
    ax_ = closed_lathe([(0.0, 0.64), (0.045, 0.64), (0.045, -0.64), (0.0, -0.64)], 20, 'metal_bare')
    ax_.v = ax_.v[:, [0, 1, 2]]
    ax_.transform(None, (0, WHEEL_Y, ZR_AX))
    ub.add(ax_)
    ub.add(closed_lathe([(0.0, 0.0), (0.13, 0.0), (0.13, 0.0)], 20, 'metal_bare') if False else
           rbox((0, WHEEL_Y, ZR_AX + 0.02), (0.11, 0.10, 0.08), 0.05, 3, 'metal_bare'))
    # driveshaft
    ub.add(P.tube_pipe([(0, 0.27, 0.90), (0, 0.255, -0.40), (0, 0.27, -1.24)], 0.033, n=2))
    out['underbody'] = ub
    ex = Mesh()
    for sg in (1, -1):
        pts = [(sg * 0.30, 0.34, 1.32), (sg * 0.33, 0.25, 1.13), (sg * 0.31, 0.181, 0.98), (sg * 0.30, 0.178, 0.30),
               (sg * 0.30, 0.175, -0.60), (sg * 0.36, 0.185, -1.15), (sg * 0.40, 0.20, -1.45),
               (sg * 0.44, 0.22, -1.85), (sg * 0.45, 0.24, -2.20), (sg * 0.45, 0.245, -2.36)]
        ex.add(P.tube_pipe(pts, 0.026, n=5))
        ex.add(rbox((sg * 0.30, 0.177, -0.85), (0.10, 0.026, 0.22), 0.02, 2, 'metal_bare'))
        tip = closed_lathe([(0.033, 0.0), (0.036, 0.0), (0.036, -0.10), (0.033, -0.10)], 20, 'chrome')
        tip.transform(rot_y(np.pi / 2), (sg * 0.45, 0.245, -2.33))
        ex.add(tip)
    out['exhaust'] = ex
    for tag, zc, R in (('F', ZF_AX, 0.360), ('R', ZR_AX, 0.372)):
        for s, sg in (('L', 1), ('R', -1)):
            out['wheel_well_%s%s' % (tag, s)] = P.wheel_well((0, WHEEL_Y, zc), R, 0.535, 0.835, sg,
                                                             a_front=92, a_rear=92)

    # markers: closures
    for s, sg in (('L', 1), ('R', -1)):
        for nm, y in (('upper', 0.76), ('lower', 0.42)):
            Pp, Nn = P.surface_point(tube, 0.612, y, sg)
            mk['hinge_door_F%s_%s' % (s, nm)] = (Pp - Nn * 0.035, (sg, 0, 0))
        Pp, Nn = P.surface_point(tube, -0.700, 0.62, sg)
        mk['latch_door_F' + s] = (Pp - Nn * 0.035, (sg, 0, 0))
        mk['hinge_hood_' + s] = ((sg * 0.70, 0.860, 0.665), (1, 0, 0))
        mk['hinge_trunk_' + s] = ((sg * 0.55, 0.925, -1.72), (1, 0, 0))
        for k, (z, y) in enumerate(((2.10, 0.79), (1.70, 0.82), (1.10, 0.85), (0.70, 0.87))):
            v = tube.v_at_height(z, y, 5.0, 6.0)[0] if False else None
            x = 0.78 if z < 2.0 else 0.70
            mk['bolt_fender_F%s_%d' % (s, k + 1)] = ((sg * x, y - 0.03, z), (1, 0, 0))
    mk['latch_hood'] = ((0.0, 0.775, 2.25), (1, 0, 0))
    mk['latch_trunk'] = ((0.0, 0.905, -2.39), (1, 0, 0))
    return out, extras, mk


def post(parts):
    """Mach 1 bonnet: the matte-black centre between the glass-edge lines
    (a mesh row, so the colour break is a clean line, not a staircase)."""
    from vehgen.mesh import MAT
    t = tube()
    m = parts['hood']
    for F, M, I in ((m.q, m.qm, m.qi), (m.t, m.tm, m.ti)):
        if not len(F):
            continue
        c = m.v[F].mean(1)
        skin = I < 100
        v = t.v_at_x(c[skin, 2], np.abs(c[skin, 0]), 7.0, 9.0)
        sel = np.zeros(len(F), bool)
        sel[np.where(skin)[0]] = v > 8.0
        M[sel] = MAT['trim_black']
