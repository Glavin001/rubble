"""The art spec's car: a generic modern mid-size four-door sedan, our own
design (no make's styling, badges or logos).

Spec targets: length 4,880, width 1,860, height 1,450, wheelbase 2,830,
track 1,600 front and rear, front overhang ~970, ground clearance 140,
tyre 235/45 R18 (683 mm). Framed doors with a hidden B-pillar, rear doors
cut round the rear wheel opening, wrap-around bumper covers, slim headlamps
either side of a wide grille, wrap-round tail lamps, a short boot.
"""
import numpy as np
from vehgen.curves import table, smoothstep, bump
from vehgen.tube import Tube, key
from vehgen.parts import molding as _molding, surface_bump as _surface_bump  # noqa: F401

NAME = 'sedan'
TITLE = 'Rubble mid-size sedan'
PAINT = (0.22, 0.006, 0.012)          # deep metallic red, linear RGB
PAINT_METALLIC = 0.5
PAINT_ROUGHNESS = 0.24

WHEELBASE = 2.830
TRACK = 1.600
ZF_AX = WHEELBASE / 2
ZR_AX = -WHEELBASE / 2
TYRE_D = 0.683
TYRE_W = 0.235
RIM_D = 18 * 0.0254
RIM_W = 8 * 0.0254
WHEEL_Y = TYRE_D / 2
RIM_STYLE = 'alloy'
DISC_D = 0.32
DOORS = ['FL', 'FR', 'RL', 'RR']
ZFRONT = 2.385
ZREAR = -2.495

_W = table([(2.42, 0.835), (2.34, 0.885), (2.15, 0.915), (1.415, 0.929), (0.0, 0.930),
            (-1.415, 0.930), (-2.10, 0.916), (-2.38, 0.890), (-2.52, 0.850)])
_ZS = np.linspace(-2.6, 2.5, 52)


def _off(d):
    return [(z, float(_W(z)) - d) for z in _ZS]


K = [
    key(0.0, 0.195),
    # 1 sill bottom / bumper bottom
    key(_off(0.070), [(2.45, 0.300), (1.90, 0.300), (1.0, 0.205), (-1.0, 0.205),
                      (-1.90, 0.320), (-2.55, 0.320)]),
    # 2 door bottom
    key(_off(0.034), [(2.45, 0.360), (1.90, 0.360), (1.0, 0.272), (-1.0, 0.272),
                      (-1.90, 0.400), (-2.55, 0.420)]),
    # 3 widest line / bumper top front
    key(_off(0.0), [(2.45, 0.500), (1.0, 0.480), (-1.0, 0.500), (-2.55, 0.560)]),
    # 4 waist crease (rising), headlamp bottom on the nose
    key(_off(0.004), [(2.45, 0.665), (2.20, 0.700), (1.80, 0.735), (1.415, 0.760), (0.0, 0.785),
                      (-1.415, 0.805), (-2.55, 0.820)]),
    # 5 shoulder
    key(_off(0.036), [(2.45, 0.775), (2.20, 0.805), (1.80, 0.840), (1.415, 0.865), (0.88, 0.905),
                      (0.0, 0.930), (-1.415, 0.950), (-2.00, 0.958), (-2.55, 0.948)]),
    # 6 belt (glass base) / bonnet shut line / boot lid edge
    key([(2.45, 0.600), (2.20, 0.660), (1.80, 0.720), (0.88, 0.790), (0.78, 0.800),
         (0.0, 0.808), (-1.0, 0.818), (-1.60, 0.818), (-2.55, 0.800)],
        [(2.45, 0.790), (2.20, 0.825), (1.80, 0.865), (1.415, 0.900), (0.88, 0.940), (0.78, 0.975),
         (0.0, 0.990), (-1.0, 1.010), (-1.60, 1.022), (-2.0, 1.012), (-2.55, 1.000)]),
    # 7 roof rail outer / daylight-opening top / C-pillar outer
    key([(2.45, 0.480), (2.0, 0.560), (1.415, 0.620), (0.88, 0.665), (0.78, 0.705),
         (0.0, 0.650), (-0.5, 0.645), (-1.0, 0.665), (-1.3, 0.705), (-1.70, 0.770),
         (-2.0, 0.640), (-2.55, 0.580)],
        [(2.45, 0.798), (2.20, 0.834), (1.80, 0.874), (1.415, 0.908), (0.88, 0.945), (0.78, 0.985), (0.30, 1.235),
         (-0.10, 1.355), (-0.5, 1.385), (-1.0, 1.345), (-1.3, 1.250), (-1.70, 1.066),
         (-2.0, 1.058), (-2.55, 1.052)]),
    # 8 windscreen / back-light edge
    key([(2.45, 0.400), (2.0, 0.500), (1.415, 0.560), (0.88, 0.600), (0.78, 0.645),
         (0.0, 0.595), (-0.5, 0.590), (-1.0, 0.595), (-1.3, 0.625), (-1.70, 0.665),
         (-2.0, 0.520), (-2.55, 0.470)],
        [(2.45, 0.802), (2.20, 0.838), (1.80, 0.878), (1.415, 0.913), (0.88, 0.950), (0.78, 0.990), (0.30, 1.250),
         (-0.10, 1.385), (-0.5, 1.412), (-1.0, 1.372), (-1.3, 1.262), (-1.70, 1.075),
         (-2.0, 1.064), (-2.55, 1.058)]),
    # 9 centre line
    key(0.0, [(2.45, 0.808), (2.20, 0.845), (1.80, 0.886), (1.415, 0.922), (0.88, 0.958), (0.78, 0.995),
              (0.30, 1.262), (-0.10, 1.415), (-0.40, 1.450), (-0.70, 1.440), (-1.00, 1.395),
              (-1.30, 1.272), (-1.70, 1.082), (-2.00, 1.070), (-2.55, 1.064)]),
]

HANDLES = [(1 / 3, 1 / 3), (0.30, 0.30), (0.33, 0.33), (0.36, 0.36), (0.18, 0.18),
           (0.22, 0.22), (0.24, 0.26), (0.33, 0.33), (0.33, 0.33), (1 / 3, 1 / 3)]

ARCH_F = dict(center=(ZF_AX, WHEEL_Y), R=0.398)
ARCH_R = dict(center=(ZR_AX, WHEEL_Y), R=0.398)


def _arch_lip(P, N, z, v):
    out = np.zeros_like(P)
    side = np.abs(N[:, 0]) > 0.5
    for a in (ARCH_F, ARCH_R):
        zc, yc = a['center']
        r = np.hypot(z - zc, P[:, 1] - yc)
        lip = 0.006 * bump(r, a['R'] + 0.010, 0.050) * (P[:, 1] > yc - 0.1)
        out[:, 0] += np.sign(P[:, 0]) * lip * side
    return out


def _lower_sculpt(P, N, z, v):
    """A lower body-side surface drawn in between the wheels, lifting to
    the rear: the modern 'rising lower line'."""
    out = np.zeros_like(P)
    side = np.clip((np.abs(N[:, 0]) - 0.6) / 0.3, 0, 1)
    zw = np.clip((z + 1.0) / 0.2, 0, 1) * np.clip((1.0 - z) / 0.2, 0, 1)
    yl = 0.40 + 0.04 * (1.0 - z) / 2.0
    prof = -0.007 * bump(P[:, 1], yl, 0.10)
    out[:, 0] = np.sign(P[:, 0]) * prof * side * zw
    return out


def _bonnet_vee(P, N, z, v):
    """Two soft creases running back from the grille corners."""
    out = np.zeros_like(P)
    ax = np.abs(P[:, 0])
    w = smoothstep(1.0, 1.2, z) * smoothstep(2.40, 2.25, z) * (np.abs(N[:, 1]) > 0.6)
    out[:, 1] = 0.006 * bump(ax, 0.30, 0.14) * w
    return out


def tube():
    return Tube(K, HANDLES, displace=[_arch_lip, _lower_sculpt, _bonnet_vee])


ZLINES = [('F', 2.37), ('HL', 2.06), ('AF1', 1.81), ('AF2', ZF_AX), ('AF3', 1.02), ('DF', 0.88),
          ('WB', 0.78), ('WT', -0.10), ('FG', -0.17), ('BP', -0.22), ('RG', -0.27),
          ('RQ', -0.97), ('AR1', -1.02), ('AR2', ZR_AX), ('AR3', -1.81), ('TL', -2.20), ('T', -2.47)]
VLINES = [0, 1, 2, 3, 4, 5, 5.33, 5.67, 6, 6.85, 7, 8, 9]


def front_z(x, y):
    x = np.abs(x)
    lean = 0.075 * np.clip((0.58 - y) / 0.28, 0, 1) ** 1.3   # bumper tucks under
    top = 0.085 * np.clip((y - 0.62) / 0.19, 0, 1) ** 1.6    # nose rolls back to the bonnet
    corner = 0.150 * (np.clip(x / 0.88, 0, 1)) ** 3
    return ZFRONT - lean - top - corner


def rear_z(x, y):
    x = np.abs(x)
    tuck = 0.060 * np.clip((0.44 - y) / 0.14, 0, 1)
    top = 0.030 * np.clip((y - 0.90) / 0.12, 0, 1)
    corner = 0.120 * (np.clip(x / 0.89, 0, 1)) ** 3
    return ZREAR + tuck + top + corner


LENS_TINT = {'headlight_L_lens': (1, 1, 1, 1), 'headlight_R_lens': (1, 1, 1, 1),
             'taillight_L_lens': (0.78, 0.02, 0.02, 1), 'taillight_R_lens': (0.78, 0.02, 0.02, 1)}


def assign(zf, zt, lo, hi, side):
    """v-line indices: 0 floor, 1 sill, 2 door bottom, 3 widest, 4 waist
    crease, 5 shoulder, 6-7 shoulder strip, 8 belt, 9 door-frame line,
    10 roof rail, 11 glass edge, 12 centre."""
    s = side
    I = zf
    front = I in ('F', 'HL', 'AF1', 'AF2', 'AF3')
    fdoor = I in ('DF', 'WB', 'WT', 'FG')
    rdoor = I in ('BP', 'RG', 'RQ')
    if lo == 0:
        return 'body_shell' if (fdoor or rdoor) else None
    if lo <= 7:
        if I == 'F':
            if lo == 4:
                return 'headlight_%s_lens' % s
            return 'bumper_front' if lo <= 3 else 'fender_F' + s
        if I == 'HL':
            return 'bumper_front' if lo <= 3 else 'fender_F' + s
        if I == 'TL' and lo == 4:
            return 'taillight_%s_lens' % s
        if I in ('AR3', 'TL'):
            return 'bumper_rear' if lo <= 2 else 'body_shell'
        if front:
            return 'fender_F' + s
        if fdoor:
            return 'body_shell' if lo == 1 else 'door_F' + s
        if rdoor:
            return 'body_shell' if lo == 1 else 'door_R' + s
        if I == 'AR3':
            return 'bumper_rear' if lo <= 2 else 'body_shell'
        return 'body_shell'
    if front:
        return 'hood'
    if I in ('AR3', 'TL'):
        return 'trunk_lid'
    if lo == 8:
        if I in ('WB', 'WT'):
            return 'glass_door_F' + s
        if I in ('DF', 'FG'):
            return 'door_F' + s
        if I == 'RG':
            return 'glass_door_R' + s
        if I in ('BP', 'RQ'):
            return 'door_R' + s
        if I == 'AR1':
            return 'glass_quarter_' + s
        return 'body_shell'
    if lo == 9:
        if fdoor:
            return 'door_F' + s
        if rdoor:
            return 'door_R' + s
        return 'body_shell'
    if lo == 11:
        if I == 'WB':
            return 'glass_windshield'
        if I in ('RQ', 'AR1', 'AR2'):
            return 'glass_rear'
    return 'body_shell'


def build_layout(cb):
    zi = cb.zi
    # back light, quarter glass and C-pillar
    for name, zs in (('RQ', {11: -1.00, 12: -1.00}),
                     ('AR1', {11: -1.20, 12: -1.20}),
                     ('AR2', {8: -1.32, 9: -1.22, 10: -1.18, 11: -1.45, 12: -1.45}),
                     ('AR3', {8: -1.80, 9: -1.75, 10: -1.70, 11: -1.70, 12: -1.70}),
                     ('DF', {}), ):
        for jl, z in zs.items():
            cb.set_z(zi[name], jl, z)
    cb.end_fit(zi['F'], front_z)
    cb.end_fit(zi['T'], rear_z)
    cb.arch(ARCH_F['center'], ARCH_F['R'], (zi['AF1'], zi['AF2'], zi['AF3']), [1, 2, 3, 4],
            [10, 22, 40, 62, 90, 118, 145, 172, 196.5], (0.6, 5.2))
    cb.arch(ARCH_R['center'], ARCH_R['R'], (zi['AR1'], zi['AR2'], zi['AR3']), [1, 2, 3, 4],
            [-16.5, 8, 35, 62, 90, 118, 145, 162, 177], (0.6, 5.2))


def build_caps(cb):
    zi = cb.zi
    i0, iT = zi['F'], zi['T']
    J = cb.J
    nF = (lambda j: cb.nid(i0, j % J))
    nT = (lambda j: cb.nid(iT, j % J))
    # nose: bumper cover with the headlamps under the bonnet edge and the
    # grille opening between them
    left = [nF(k) for k in range(1, 6)]
    right = [nF(J - k) for k in range(1, 6)]
    top = [nF(k) for k in range(5, J - 4)]          # K5 .. centre .. K5'
    C = len(top) - 1
    y0 = cb.L.node_pos(nF(1))[1]
    xs = [cb.L.node_pos(n)[0] for n in top]
    bottom = [nF(1)] + [(x, y0) for x in xs[1:-1]] + [nF(J - 1)]
    hl = {(3, c) for c in range(0, 3)}
    hr = {(3, c) for c in range(C - 3, C)}
    gr = {(3, c) for c in range(3, C - 3)} | {(1, c) for c in range(4, C - 4)}

    def fpart(r, c):
        if (r, c) in hl:
            return 'headlight_L_lens'
        if (r, c) in hr:
            return 'headlight_R_lens'
        if (r, c) in gr:
            return None
        return 'bumper_front'
    cb.cap('capF', front_z, 1, left, right, bottom, top, fpart)
    cb.grille_cols = (xs[3], xs[C - 3], xs[4], xs[C - 4])
    # tail: bumper cover low, tail panel, wrap-round lamps under the lid
    left = [nT(k) for k in range(1, 6)]
    right = [nT(J - k) for k in range(1, 6)]
    y0 = cb.L.node_pos(nT(1))[1]
    xk1 = cb.L.node_pos(nT(1))[0]
    DROP = 0.0125
    xsr = [0.55, 0.28, 0.0, -0.28, -0.55]
    lid = []
    for x in xsr:
        v = cb.tube.v_at_x(-2.47, abs(x), 6.0, 9.0)[0]
        lid.append(cb.tube.eval(-2.47, v)[0][1] - DROP)
    pk = cb.L.node_pos(nT(8))
    top = ([nT(5), nT(6), nT(7), (pk[0], pk[1] - DROP)] + [(x, y) for x, y in zip(xsr, lid)]
           + [(-pk[0], pk[1] - DROP), nT(J - 7), nT(J - 6), nT(J - 5)])
    fan = [xk1 - (xk1 - 0.55) * k / 4 for k in (1, 2, 3)]
    bottom = ([nT(1)] + [(x, y0) for x in fan] + [(x, y0) for x in xsr]
              + [(-x, y0) for x in reversed(fan)] + [nT(J - 1)])
    lidp = np.array([cb.L.nodes[nT(j)][1] for j in range(8, J - 7)])
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
    tx = [pk[0]] + xsr + [-pk[0]]
    arcs = {((4, 3 + k), (4, 4 + k)): lid_curve(tx[k], tx[k + 1]) for k in range(len(tx) - 1)}
    C = len(top) - 1

    def rpart(r, c):
        if r == 3 and c <= 3:
            return 'taillight_L_lens'
        if r == 3 and c >= C - 4:
            return 'taillight_R_lens'
        if r <= 1:
            return 'bumper_rear'
        return 'body_shell'
    cb.cap('capR', rear_z, -1, left, right, bottom, top, rpart, arcs=arcs)


# --- parts ------------------------------------------------------------------------
def parts(cb):
    from vehgen import parts as P
    from vehgen.mesh import (Mesh, rbox, box, sweep, rrect, circle, closed_lathe,
                             orient_outward, catmull, grid, rot_x, rot_y, loft)
    from vehgen.panels import inner_panel, boundary_loops
    tube = cb.tube
    out, extras, mk = {}, {}, {}
    D = __import__(__name__)
    VM = tube.VM
    J = cb.J

    for tag, z in (('F', ZF_AX), ('R', ZR_AX)):
        for s, sg in (('L', 1), ('R', -1)):
            rim, ty, di, c = P.wheel(D, sg, z)
            out['rim_' + tag + s] = rim
            out['tyre_' + tag + s] = ty
            out['brake_disc_' + tag + s] = di
            mk['wheel_center_' + tag + s] = (c, (sg, 0, 0))

    # lamp housings: a reflector shell following each lens 30 mm behind it,
    # chrome projector cups in the headlamps
    from vehgen.panels import inner_panel as _ip
    for s, sg in (('L', 1), ('R', -1)):
        for kind in ('headlight', 'taillight'):
            sk = cb.skins['%s_%s_lens' % (kind, s)].copy()
            isb = np.zeros(len(sk.v), bool)
            for lp in boundary_loops(sk):
                isb[lp] = True
            h = _ip(sk, isb, depth=0.032, rings=1, mat='lamp_emissive', island=-10**6)
            h.flip()
            if kind == 'headlight':
                lo, hi = sk.v.min(0), sk.v.max(0)
                n = sk.vertex_normals().mean(0)
                n /= np.linalg.norm(n)
                c0 = sk.v.mean(0)
                vn = sk.vertex_normals()
                for k in (-1, 1):
                    q = c0 + np.array([k * sg * 0.075, 0.0, 0.0])
                    i_ = int(np.argmin(np.linalg.norm(sk.v - q, axis=1)))
                    n = vn[i_]
                    pc = sk.v[i_] - n * 0.034
                    cup = closed_lathe([(0.0, 0.002), (0.020, 0.002), (0.024, -0.008), (0.008, -0.016),
                                        (0.0, -0.016)], 24, 'chrome')
                    cup.transform(P.rot_to(n), pc)
                    h.add(cup)
            out['%s_%s_housing' % (kind, s)] = h

    # grille: honeycomb following the nose behind the slim upper opening and
    # the lower intake, each with a dark back plate
    g = Mesh()
    gu, _, gl, _ = cb.grille_cols
    for (gx, ya, yb) in ((abs(gu) - 0.025, 0.680, 0.765), (abs(gl) - 0.025, 0.375, 0.485)):
        xs_ = np.linspace(-gx, gx, 40)
        ys_ = np.linspace(ya, yb, 6)
        X, Y = np.meshgrid(xs_, ys_, indexing='ij')
        Z = front_z(X.reshape(-1), Y.reshape(-1)).reshape(X.shape) - 0.045
        plate = grid(np.stack([X, Y, Z], -1), mat='trim_black')
        g.add(plate)
        for i, y in enumerate(np.arange(ya + 0.012, yb - 0.006, 0.024)):
            for x in np.arange(-gx + 0.02 + (0.016 if i % 2 else 0), gx - 0.015, 0.032):
                zc = float(front_z(np.array([x]), np.array([y]))[0]) - 0.026
                g.add(box((x, y, zc), (0.011, 0.007, 0.008), 'trim_black'))
    out['grille'] = g

    # mirrors on the door sail, handles, wipers
    for s, sg in (('L', 1), ('R', -1)):
        v = 6.4 if sg > 0 else VM - 6.4
        base, bn = tube.eval(0.80, v, with_normal=True)
        base, bn = base[0], bn[0]
        m = Mesh()
        m.add(rbox(base + bn * 0.026, (0.010, 0.018, 0.035), 0.008, 2, 'paint'))
        hc = base + bn * 0.016 + [sg * 0.095, 0.060, -0.020]
        m.add(rbox(hc, (0.085, 0.055, 0.045), 0.035, 4, 'paint'))
        m.add(rbox(hc + [0, 0, -0.046], (0.072, 0.044, 0.0015), 0.0014, 1, 'chrome'))
        m.add(rbox(base + bn * 0.016 + [sg * 0.045, 0.040, -0.010], (0.040, 0.012, 0.020), 0.010, 2, 'trim_black'))
        out['mirror_' + s] = m
        mk['mount_mirror_' + s] = (base, (sg, 0, 0))
        out['handle_F' + s] = P.door_handle(tube, 0.05, 0.865, sg, length=0.16)
        out['handle_R' + s] = P.door_handle(tube, -0.80, 0.880, sg, length=0.16)
    for s, x0, x1 in (('L', 0.62, 0.08), ('R', 0.02, -0.52)):
        zs = 0.72
        pts = []
        for x in np.linspace(x0, x1, 8):
            v = tube.v_at_x(zs, abs(x), 8.0, 9.0)[0]
            if x < 0:
                v = VM - v
            Pp, Nn = tube.eval(zs, v, with_normal=True)
            pts.append(Pp[0] + Nn[0] * 0.004)
        out['wiper_' + s] = P.wiper(np.array(pts))

    # body trim: black B-pillar applique band under the door frames, sill
    # strips, windscreen and back-light reveals
    bx = Mesh()
    for side in (1, -1):
        f = (lambda v: v) if side > 0 else (lambda v: VM - v)
        sv = np.linspace(0.80, -0.95, 34)
        Pp, Nn = P.path_on_surface(tube, sv, 0.240, side, off=0.004)
        bx.add(orient_outward(sweep(Pp, rrect(0.004, 0.022, 0.0015, 1), mat='trim_black')))
    ws = ([(0.78, 8.0), (0.34, 8.0), (-0.10, 8.0), (-0.10, 9.0), (-0.10, 10.0), (0.34, 10.0),
           (0.78, 10.0), (0.78, 9.0)])
    bx.add(_molding(tube, ws, off=0.0015, sec=(0.008, 0.004), closed=True, n=6))
    extras['body_shell'] = bx

    # interior
    hf = 0.215
    for s, sg in (('L', 1), ('R', -1)):
        out['seat_F' + s] = P.bucket_seat(sg * 0.38, -0.40, hf + 0.02)
        mk['seat_H_point_F' + s] = ((sg * 0.38, hf + 0.20, -0.53), (1, 0, 0))
    rs = Mesh()
    rs.add(rbox((0, hf + 0.22, -1.08), (0.52, 0.07, 0.24), 0.05, 3, 'interior_fabric'))
    back = rbox((0, 0, 0), (0.52, 0.29, 0.06), 0.05, 3, 'interior_fabric')
    back.transform(rot_x(np.radians(-22)), (0, hf + 0.56, -1.38))
    rs.add(back)
    out['seat_rear'] = rs
    dash = Mesh()
    dash.add(rbox((0, 0.78, 0.48), (0.76, 0.12, 0.16), 0.06, 3, 'interior_plastic'))
    dash.add(rbox((0.38, 0.92, 0.42), (0.18, 0.03, 0.08), 0.025, 2, 'interior_plastic'))
    dash.add(rbox((0, 0.86, 0.33), (0.13, 0.08, 0.02), 0.01, 2, 'trim_black'))
    out['dash'] = dash
    out['steering_wheel'] = P.steering_wheel((0.38, 0.84, 0.17), R=0.185, tilt_deg=24, column=0.13)
    out['console'] = rbox((0, 0.42, -0.15), (0.11, 0.12, 0.45), 0.03, 2, 'interior_plastic')
    for s in ('FL', 'FR', 'RL', 'RR'):
        sk = cb.skins['door_' + s].copy()
        isb = np.zeros(len(sk.v), bool)
        for lp in boundary_loops(sk):
            isb[lp] = True
        out['door_trim_' + s] = inner_panel(sk, isb, depth=0.085, rings=2, mat='interior_plastic',
                                            island=-10**6)
    zz = np.linspace(-0.05, -0.95, 46)
    vv = np.linspace(7.35, VM - 7.35, 60)
    Z, V = np.meshgrid(zz, vv, indexing='ij')
    Pp, Nn = tube.eval(Z.reshape(-1), V.reshape(-1), with_normal=True)
    Pp = (Pp - Nn * 0.040).reshape(len(zz), len(vv), 3)
    hl_ = grid(Pp, mat='interior_fabric')
    if np.mean(hl_.vertex_normals()[:, 1]) > 0:
        hl_.flip()
    out['headliner'] = hl_

    # bay: transverse four with gearbox, radiator, firewall
    eb = Mesh()
    eb.add(rbox((0.05, 0.50, 1.55), (0.30, 0.17, 0.13), 0.03, 2, 'metal_bare'))
    eb.add(rbox((0.05, 0.71, 1.55), (0.28, 0.04, 0.10), 0.02, 2, 'trim_black'))
    eb.add(rbox((-0.36, 0.42, 1.52), (0.12, 0.13, 0.12), 0.04, 2, 'metal_bare'))
    eb.add(rbox((0.05, 0.62, 1.74), (0.30, 0.07, 0.04), 0.02, 2, 'trim_black'))
    out['engine_block'] = eb
    rad = Mesh()
    rad.add(box((0, 0.56, 2.12), (0.36, 0.16, 0.024), 'trim_black'))
    rad.add(box((0, 0.735, 2.12), (0.37, 0.014, 0.030), 'metal_bare'))
    out['radiator'] = rad
    out['firewall'] = box((0, 0.60, 0.94), (0.60, 0.28, 0.003), 'trim_black')
    ub = Mesh()
    ub.add(box((0, 0.150, 1.60), (0.46, 0.003, 0.45), 'trim_black'))
    ub.add(box((0, 0.330, -2.05), (0.50, 0.004, 0.28), 'trim_black'))
    ub.add(rbox((-0.09, 0.30, -1.90), (0.21, 0.05, 0.22), 0.02, 2, 'metal_bare'))
    out['underbody'] = ub
    ex = Mesh()
    pts = [(0.10, 0.36, 1.30), (0.12, 0.25, 1.12), (0.10, 0.168, 0.95), (0.10, 0.166, -0.70),
           (0.16, 0.175, -1.20), (0.30, 0.20, -1.60), (0.42, 0.25, -2.10), (0.48, 0.26, -2.42)]
    ex.add(P.tube_pipe(pts, 0.022, n=5))
    ex.add(rbox((0.12, 0.168, -0.20), (0.09, 0.016, 0.25), 0.012, 2, 'metal_bare'))
    ex.add(rbox((0.36, 0.230, -1.92), (0.20, 0.06, 0.13), 0.04, 2, 'metal_bare'))
    out['exhaust'] = ex
    for tag, zc in (('F', ZF_AX), ('R', ZR_AX)):
        for s, sg in (('L', 1), ('R', -1)):
            out['wheel_well_%s%s' % (tag, s)] = P.wheel_well((0, WHEEL_Y, zc), 0.375, 0.56, 0.86, sg,
                                                             a_front=92, a_rear=92)

    # markers
    for s, sg in (('L', 1), ('R', -1)):
        for nm, y in (('upper', 0.82), ('lower', 0.46)):
            Pp, Nn = P.surface_point(tube, 0.86, y, sg)
            mk['hinge_door_F%s_%s' % (s, nm)] = (Pp - Nn * 0.035, (sg, 0, 0))
            Pp, Nn = P.surface_point(tube, -0.24, y, sg)
            mk['hinge_door_R%s_%s' % (s, nm)] = (Pp - Nn * 0.035, (sg, 0, 0))
        Pp, Nn = P.surface_point(tube, -0.20, 0.64, sg)
        mk['latch_door_F' + s] = (Pp - Nn * 0.035, (sg, 0, 0))
        Pp, Nn = P.surface_point(tube, -1.00, 0.70, sg)
        mk['latch_door_R' + s] = (Pp - Nn * 0.035, (sg, 0, 0))
        mk['hinge_hood_' + s] = ((sg * 0.68, 0.905, 0.92), (1, 0, 0))
        mk['hinge_trunk_' + s] = ((sg * 0.55, 1.02, -1.75), (1, 0, 0))
        for k, (z, y) in enumerate(((2.15, 0.83), (1.80, 0.86), (1.30, 0.89), (0.95, 0.91))):
            x = float(K[6][0](np.array([z]))[0]) + 0.03
            mk['bolt_fender_F%s_%d' % (s, k + 1)] = ((sg * x, y - 0.03, z), (1, 0, 0))
    for k, x in enumerate((-0.70, -0.35, 0.35, 0.70)):
        zz = float(front_z(np.array([abs(x)]), np.array([0.42]))[0])
        mk['bolt_bumper_front_%d' % (k + 1)] = ((x, 0.42, zz - 0.03), (1, 0, 0))
        zz = float(rear_z(np.array([abs(x)]), np.array([0.45]))[0])
        mk['bolt_bumper_rear_%d' % (k + 1)] = ((x, 0.45, zz + 0.03), (1, 0, 0))
    mk['latch_hood'] = ((0.0, 0.80, 2.30), (1, 0, 0))
    mk['latch_trunk'] = ((0.0, 0.98, -2.45), (1, 0, 0))
    return out, extras, mk
