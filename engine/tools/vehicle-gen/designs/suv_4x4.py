"""A boxy off-road 4x4 SUV, our own design: two-box body with an upright
windscreen, long cargo windows, a tailgate carrying the spare wheel, flared
wheel openings, steel wheels on 32-inch tyres, a roof rack and rock sliders.
The kind of tall, flat-panelled vehicle that crumples dramatically.

Targets: length ~4.6 m (body), width 1.86 m, height ~1.86 m (1.92 over the
rack), wheelbase 2.70 m, track 1.56 m, tyre 265/75 R16 (802 mm), ground
clearance ~240 mm.
"""
import numpy as np
from vehgen.curves import table, smoothstep, bump
from vehgen.tube import Tube, key
from vehgen.parts import molding as _molding, surface_bump as _surface_bump  # noqa: F401

NAME = 'suv_4x4'
TITLE = 'Rubble off-road 4x4'
PAINT = (0.10, 0.16, 0.06)             # olive green, linear RGB
PAINT_METALLIC = 0.15
PAINT_ROUGHNESS = 0.38

WHEELBASE = 2.700
TRACK = 1.560
ZF_AX = WHEELBASE / 2
ZR_AX = -WHEELBASE / 2
TYRE_D = 0.802
TYRE_W = 0.265
RIM_D = 16 * 0.0254
RIM_W = 7 * 0.0254
WHEEL_Y = TYRE_D / 2
RIM_STYLE = 'steel'
DISC_D = 0.30
DOORS = ['FL', 'FR', 'RL', 'RR']
TAILGATE = True
ZFRONT = 2.150
ZREAR = -2.300

_W = table([(2.20, 0.860), (2.10, 0.905), (1.90, 0.925), (1.35, 0.930), (0.0, 0.930),
            (-1.35, 0.930), (-2.10, 0.925), (-2.25, 0.905), (-2.35, 0.870)])
_ZS = np.linspace(-2.4, 2.3, 48)


def _off(d):
    return [(z, float(_W(z)) - d) for z in _ZS]


K = [
    key(0.0, 0.400),
    key(_off(0.055), [(2.30, 0.560), (1.85, 0.560), (0.95, 0.450), (-0.95, 0.450),
                      (-1.85, 0.560), (-2.40, 0.560)]),
    key(_off(0.026), [(2.30, 0.630), (1.85, 0.630), (0.95, 0.520), (-0.95, 0.520),
                      (-1.85, 0.640), (-2.40, 0.640)]),
    key(_off(0.0), [(2.30, 0.760), (1.0, 0.740), (-1.0, 0.740), (-2.40, 0.760)]),
    key(_off(0.002), [(2.30, 0.970), (1.35, 0.985), (0.0, 0.985), (-1.35, 0.990), (-2.40, 0.990)]),
    key(_off(0.024), [(2.30, 1.100), (1.35, 1.120), (0.80, 1.130), (0.0, 1.135), (-2.40, 1.140)]),
    key([(2.30, 0.740), (2.05, 0.770), (1.60, 0.795), (0.80, 0.810), (0.72, 0.840),
         (0.0, 0.850), (-2.40, 0.850)],
        [(2.30, 1.115), (1.35, 1.135), (0.80, 1.150), (0.72, 1.170), (0.0, 1.170), (-2.40, 1.170)]),
    key([(2.30, 0.620), (1.35, 0.680), (0.80, 0.730), (0.72, 0.800), (0.30, 0.790), (0.0, 0.790),
         (-2.40, 0.790)],
        [(2.30, 1.120), (1.35, 1.140), (0.80, 1.155), (0.72, 1.175), (0.30, 1.770), (0.0, 1.790),
         (-2.40, 1.790)]),
    key([(2.30, 0.500), (1.35, 0.580), (0.80, 0.650), (0.72, 0.730), (0.30, 0.740), (0.0, 0.740),
         (-2.40, 0.740)],
        [(2.30, 1.124), (1.35, 1.144), (0.80, 1.158), (0.72, 1.180), (0.30, 1.825), (0.0, 1.845),
         (-2.40, 1.845)]),
    key(0.0, [(2.30, 1.130), (1.35, 1.150), (0.80, 1.165), (0.72, 1.190), (0.30, 1.840),
              (0.0, 1.860), (-2.40, 1.860)]),
]

HANDLES = [(1 / 3, 1 / 3), (0.30, 0.30), (0.33, 0.33), (0.36, 0.36), (0.15, 0.15),
           (0.14, 0.14), (0.20, 0.22), (0.30, 0.30), (0.20, 0.20), (1 / 3, 1 / 3)]

ARCH_F = dict(center=(ZF_AX, WHEEL_Y), R=0.460)
ARCH_R = dict(center=(ZR_AX, WHEEL_Y), R=0.460)


def _flares(P, N, z, v):
    """Box flares round the wheel openings."""
    out = np.zeros_like(P)
    side = np.clip((np.abs(N[:, 0]) - 0.5) / 0.3, 0, 1)
    for a in (ARCH_F, ARCH_R):
        zc, yc = a['center']
        r = np.hypot(z - zc, P[:, 1] - yc)
        fl = 0.022 * smoothstep(a['R'] + 0.13, a['R'] + 0.03, r) * (P[:, 1] > yc - 0.15)
        out[:, 0] += np.sign(P[:, 0]) * fl * side
    return out


def _bonnet_ribs(P, N, z, v):
    out = np.zeros_like(P)
    ax = np.abs(P[:, 0])
    w = smoothstep(0.85, 1.0, z) * smoothstep(2.20, 2.05, z) * (np.abs(N[:, 1]) > 0.6)
    out[:, 1] = 0.008 * bump(ax, 0.36, 0.08) * w
    return out


def tube():
    return Tube(K, HANDLES, displace=[_flares, _bonnet_ribs])


ZLINES = [('F', 2.14), ('HL', 1.98), ('AF1', 1.81), ('AF2', ZF_AX), ('AF3', 0.89), ('DF', 0.80),
          ('WB', 0.72), ('WT', 0.30), ('FG', 0.0), ('BP', -0.05), ('RG', -0.10),
          ('RQ', -0.80), ('AR1', -0.89), ('AR2', ZR_AX), ('AR3', -1.81), ('QG', -2.06), ('T', -2.28)]
VLINES = [0, 1, 2, 3, 4, 5, 5.33, 5.67, 6, 6.85, 7, 8, 9]


def front_z(x, y):
    x = np.abs(x)
    lean = 0.030 * np.clip((0.80 - y) / 0.25, 0, 1)
    top = 0.040 * np.clip((y - 1.02) / 0.10, 0, 1) ** 1.5
    corner = 0.090 * (np.clip(x / 0.90, 0, 1)) ** 5
    return ZFRONT - lean - top - corner


def rear_z(x, y):
    x = np.abs(x)
    tuck = 0.030 * np.clip((0.66 - y) / 0.10, 0, 1)
    top = 0.040 * np.clip((y - 1.70) / 0.15, 0, 1) ** 1.5
    corner = 0.070 * (np.clip(x / 0.90, 0, 1)) ** 5
    return ZREAR + tuck + top + corner


LENS_TINT = {'headlight_L_lens': (1, 1, 1, 1), 'headlight_R_lens': (1, 1, 1, 1),
             'taillight_L_lens': (0.78, 0.02, 0.02, 1), 'taillight_R_lens': (0.78, 0.02, 0.02, 1)}


def assign(zf, zt, lo, hi, side):
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
            return 'bumper_front' if lo <= 2 else 'fender_F' + s
        if I == 'HL':
            return 'bumper_front' if lo <= 2 else 'fender_F' + s
        if front:
            return 'fender_F' + s
        if fdoor:
            return 'body_shell' if lo == 1 else 'door_F' + s
        if rdoor:
            return 'body_shell' if lo == 1 else 'door_R' + s
        if I in ('AR3', 'QG'):
            return 'bumper_rear' if lo <= 1 else 'body_shell'
        return 'body_shell'
    if front:
        return 'hood'
    if lo == 8:
        if I in ('WB', 'WT'):
            return 'glass_door_F' + s
        if I in ('DF', 'FG'):
            return 'door_F' + s
        if I == 'RG':
            return 'glass_door_R' + s
        if I in ('BP', 'RQ'):
            return 'door_R' + s
        if I in ('AR1', 'AR2', 'AR3'):
            return 'glass_quarter_' + s
        return 'body_shell'
    if lo == 9:
        if fdoor:
            return 'door_F' + s
        if rdoor:
            return 'door_R' + s
        return 'body_shell'
    if lo == 11 and I == 'WB':
        return 'glass_windshield'
    return 'body_shell'


def build_layout(cb):
    zi = cb.zi
    cb.end_fit(zi['F'], front_z)
    cb.end_fit(zi['T'], rear_z)
    cb.arch(ARCH_F['center'], ARCH_F['R'], (zi['AF1'], zi['AF2'], zi['AF3']), [1, 2, 3, 4],
            [16, 26, 42, 62, 90, 118, 145, 168, 192], (0.6, 5.2))
    cb.arch(ARCH_R['center'], ARCH_R['R'], (zi['AR1'], zi['AR2'], zi['AR3']), [1, 2, 3, 4],
            [-12, 12, 35, 62, 90, 118, 145, 168, 192], (0.6, 5.2))


def build_caps(cb):
    zi = cb.zi
    i0, iT = zi['F'], zi['T']
    J = cb.J
    nF = (lambda j: cb.nid(i0, j % J))
    nT = (lambda j: cb.nid(iT, j % J))
    # nose: square lamps either side of a tall grille, apron below
    left = [nF(k) for k in range(1, 6)]
    right = [nF(J - k) for k in range(1, 6)]
    top = [nF(k) for k in range(5, J - 4)]
    C = len(top) - 1
    y0 = cb.L.node_pos(nF(1))[1]
    xs = [cb.L.node_pos(n)[0] for n in top]
    bottom = [nF(1)] + [(x, y0) for x in xs[1:-1]] + [nF(J - 1)]
    hl = {(3, c) for c in range(0, 5)}
    hr = {(3, c) for c in range(C - 5, C)}
    gr = {(r, c) for r in (2, 3) for c in range(5, C - 5)}

    def fpart(r, c):
        if (r, c) in hl:
            return 'headlight_L_lens'
        if (r, c) in hr:
            return 'headlight_R_lens'
        if (r, c) in gr:
            return None
        if r <= 1:
            return 'bumper_front'
        return 'fender_FL' if c < C / 2 else 'fender_FR'
    cb.cap('capF', front_z, 1, left, right, bottom, top, fpart)
    cb.grille_cols = (xs[5], xs[C - 5])
    # tail: D-pillars with tall lamps, the tailgate between them carrying
    # the back light, bumper below
    left = [nT(k) for k in range(1, 11)]
    right = [nT(J - k) for k in range(1, 11)]
    top = [nT(10), nT(11), nT(12), nT(J - 11), nT(J - 10)]
    C = len(top) - 1
    y0 = cb.L.node_pos(nT(1))[1]
    xg = 0.70
    xs = [cb.L.node_pos(n)[0] for n in top]
    bottom = [nT(1), (xg, y0), (0.0, y0), (-xg, y0), nT(J - 1)]
    R = len(left) - 1
    ov = {}
    for r in range(1, R):
        yr = cb.L.node_pos(left[r])[1]
        ov[(r, 1)] = (xg, yr)
        ov[(r, 2)] = (0.0, yr)
        ov[(r, 3)] = (-xg, yr)

    def rpart(r, c):
        if r == 0:
            return 'bumper_rear'
        if c in (0, 3):
            if r in (2, 3):
                return 'taillight_%s_lens' % ('L' if c == 0 else 'R')
            return 'body_shell'
        if r == 7:
            return 'glass_rear'
        if r == R - 1 and False:
            return 'body_shell'
        return 'tailgate'
    cb.cap('capR', rear_z, -1, left, right, bottom, top, rpart, overrides=ov)


# --- parts ------------------------------------------------------------------------
def parts(cb):
    from vehgen import parts as P
    from vehgen.mesh import (Mesh, rbox, box, sweep, rrect, circle, closed_lathe,
                             orient_outward, grid, rot_x, rot_y)
    from vehgen.panels import inner_panel, boundary_loops
    tube = cb.tube
    out, extras, mk = {}, {}, {}
    D = __import__(__name__)
    VM = tube.VM

    for tag, z in (('F', ZF_AX), ('R', ZR_AX)):
        for s, sg in (('L', 1), ('R', -1)):
            rim, ty, di, c = P.wheel(D, sg, z)
            out['rim_' + tag + s] = rim
            out['tyre_' + tag + s] = ty
            out['brake_disc_' + tag + s] = di
            mk['wheel_center_' + tag + s] = (c, (sg, 0, 0))

    # lamp housings: reflector shells behind the lenses, chrome cups in front
    for s, sg in (('L', 1), ('R', -1)):
        for kind in ('headlight', 'taillight'):
            sk = cb.skins['%s_%s_lens' % (kind, s)].copy()
            isb = np.zeros(len(sk.v), bool)
            for lp in boundary_loops(sk):
                isb[lp] = True
            h = inner_panel(sk, isb, depth=0.034, rings=1, mat='lamp_emissive', island=-10**6)
            h.flip()
            if kind == 'headlight':
                vn = sk.vertex_normals()
                c0 = sk.v.mean(0)
                i_ = int(np.argmin(np.linalg.norm(sk.v - c0, axis=1)))
                n = vn[i_]
                cup = closed_lathe([(0.0, 0.002), (0.045, 0.002), (0.050, -0.010), (0.016, -0.022),
                                    (0.0, -0.022)], 32, 'chrome')
                cup.transform(P.rot_to(n), sk.v[i_] - n * 0.036)
                h.add(cup)
            out['%s_%s_housing' % (kind, s)] = h

    # grille: vertical slats in the opening, a dark back plate
    g = Mesh()
    gx = abs(cb.grille_cols[0]) - 0.03
    ya, yb = 0.785, 1.080
    X, Y = np.meshgrid(np.linspace(-gx, gx, 30), np.linspace(ya, yb, 8), indexing='ij')
    Z = front_z(X.reshape(-1), Y.reshape(-1)).reshape(X.shape) - 0.050
    g.add(grid(np.stack([X, Y, Z], -1), mat='trim_black'))
    for x in np.arange(-gx + 0.02, gx - 0.01, 0.045):
        zc = float(front_z(np.array([x]), np.array([(ya + yb) / 2]))[0]) - 0.030
        g.add(box((x, (ya + yb) / 2, zc), (0.010, (yb - ya) / 2 - 0.010, 0.012), 'trim_black'))
    out['grille'] = g

    # steel bumpers: a square tube bar on brackets, front and rear
    def bar(zf, y, sign, half_w):
        b = Mesh()
        b.add(rbox((0, y, zf + sign * 0.095), (half_w, 0.065, 0.055), 0.012, 2, 'trim_black'))
        for x in (-0.55, 0.55):
            b.add(rbox((x, y - 0.02, zf + sign * 0.030), (0.04, 0.04, 0.022), 0.006, 1, 'trim_black'))
        return b
    extras['bumper_front'] = bar(float(front_z(np.array([0.5]), np.array([0.60]))[0]), 0.600, 1, 0.92)
    extras['bumper_rear'] = bar(float(rear_z(np.array([0.5]), np.array([0.60]))[0]), 0.600, -1, 0.90)

    # spare wheel on the tailgate
    zt = float(rear_z(np.array([0.0]), np.array([1.05]))[0])
    sp_t = P.tyre(TYRE_D, TYRE_W, RIM_D, RIM_W)
    sp_r = P.rim_steel(RIM_D, RIM_W)
    spare = Mesh().add(sp_r).add(sp_t)
    spare.transform(rot_y(np.pi / 2), (0, 1.09, zt - TYRE_W / 2 - 0.045))
    spare.add(rbox((0, 1.09, zt - 0.024), (0.06, 0.06, 0.018), 0.01, 1, 'metal_bare'))
    out['spare_wheel'] = spare

    # mirrors, handles, wipers
    for s, sg in (('L', 1), ('R', -1)):
        v = 5.8 if sg > 0 else VM - 5.8
        base, bn = tube.eval(0.62, v, with_normal=True)
        base, bn = base[0], bn[0]
        m = Mesh()
        m.add(rbox(base + bn * 0.028, (0.012, 0.022, 0.035), 0.008, 2, 'trim_black'))
        hc = base + bn * 0.028 + [sg * 0.11, 0.07, -0.02]
        m.add(rbox(hc, (0.10, 0.075, 0.035), 0.020, 3, 'trim_black'))
        m.add(rbox(hc + [0, 0, -0.036], (0.088, 0.064, 0.0015), 0.0014, 1, 'chrome'))
        m.add(rbox(base + bn * 0.028 + [sg * 0.05, 0.045, -0.01], (0.045, 0.012, 0.016), 0.008, 2, 'trim_black'))
        out['mirror_' + s] = m
        mk['mount_mirror_' + s] = (base, (sg, 0, 0))
        out['handle_F' + s] = P.door_handle(tube, 0.04, 1.06, sg, length=0.15)
        out['handle_R' + s] = P.door_handle(tube, -0.72, 1.06, sg, length=0.15)
    for s, x0, x1 in (('L', 0.62, 0.06), ('R', 0.02, -0.54)):
        zs = 0.66
        pts = []
        for x in np.linspace(x0, x1, 8):
            v = tube.v_at_x(zs, abs(x), 8.0, 9.0)[0]
            if x < 0:
                v = VM - v
            Pp, Nn = tube.eval(zs, v, with_normal=True)
            pts.append(Pp[0] + Nn[0] * 0.005)
        out['wiper_' + s] = P.wiper(np.array(pts))

    # body extras: rock sliders, roof rack, windscreen reveal
    bx = Mesh()
    for sg in (1, -1):
        zs_ = np.linspace(0.78, -0.82, 24)
        Pp, Nn = P.path_on_surface(tube, zs_, 0.50, sg, off=0.060)
        bx.add(P.tube_pipe(Pp, 0.030, n=1))
        for z in (0.55, -0.10, -0.65):
            Q, Nq = P.surface_point(tube, z, 0.50, sg)
            bx.add(P.tube_pipe([Q + Nq * 0.002, Q + Nq * 0.045], 0.016, n=1))
        rail = []
        for z in np.linspace(0.10, -2.12, 26):
            v = 7.55 if sg > 0 else VM - 7.55
            Q, Nq = tube.eval(z, v, with_normal=True)
            rail.append(Q[0] + np.array([0, 0.055, 0]))
        bx.add(P.tube_pipe(np.array(rail), 0.016, n=1))
        for z in (0.0, -2.05):
            v = 7.55 if sg > 0 else VM - 7.55
            Q = tube.eval(z, v)[0]
            bx.add(P.tube_pipe([Q + [0, 0.004, 0], Q + [0, 0.055, 0]], 0.014, n=1))
    for z in (-0.35, -0.95, -1.55):
        vl, vr = 7.55, VM - 7.55
        A_ = tube.eval(z, vl)[0] + [0, 0.055, 0]
        B_ = tube.eval(z, vr)[0] + [0, 0.055, 0]
        bx.add(P.tube_pipe([A_, (A_ + B_) / 2 + [0, 0.012, 0], B_], 0.014, n=2))
    for k in range(len(bx.qm)):
        pass
    bx.set_mat('trim_black')
    ws = ([(0.72, 8.0), (0.51, 8.0), (0.30, 8.0), (0.30, 9.0), (0.30, 10.0), (0.51, 10.0),
           (0.72, 10.0), (0.72, 9.0)])
    bx.add(_molding(tube, ws, off=0.0015, sec=(0.010, 0.005), closed=True, n=6).set_mat('trim_black'))
    extras['body_shell'] = bx

    # interior
    hf = 0.430
    for s, sg in (('L', 1), ('R', -1)):
        out['seat_F' + s] = P.bucket_seat(sg * 0.38, -0.30, hf)
        mk['seat_H_point_F' + s] = ((sg * 0.38, hf + 0.18, -0.43), (1, 0, 0))
    rs = Mesh()
    rs.add(rbox((0, hf + 0.20, -1.05), (0.52, 0.07, 0.24), 0.05, 3, 'interior_fabric'))
    back = rbox((0, 0, 0), (0.52, 0.30, 0.06), 0.05, 3, 'interior_fabric')
    back.transform(rot_x(np.radians(-12)), (0, hf + 0.56, -1.32))
    rs.add(back)
    out['seat_rear'] = rs
    dash = Mesh()
    dash.add(rbox((0, 1.00, 0.52), (0.76, 0.13, 0.14), 0.04, 3, 'interior_plastic'))
    dash.add(rbox((0.38, 1.15, 0.46), (0.17, 0.03, 0.07), 0.02, 2, 'interior_plastic'))
    out['dash'] = dash
    out['steering_wheel'] = P.steering_wheel((0.38, 1.04, 0.22), R=0.19, tilt_deg=30, column=0.12)
    out['console'] = rbox((0, 0.62, -0.05), (0.11, 0.10, 0.42), 0.03, 2, 'interior_plastic')
    for s in ('FL', 'FR', 'RL', 'RR'):
        sk = cb.skins['door_' + s].copy()
        isb = np.zeros(len(sk.v), bool)
        for lp in boundary_loops(sk):
            isb[lp] = True
        out['door_trim_' + s] = inner_panel(sk, isb, depth=0.085, rings=2, mat='interior_plastic',
                                            island=-10**6)
    zz = np.linspace(0.24, -2.15, 80)
    vv = np.linspace(7.35, VM - 7.35, 60)
    Z, V = np.meshgrid(zz, vv, indexing='ij')
    Pp, Nn = tube.eval(Z.reshape(-1), V.reshape(-1), with_normal=True)
    Pp = (Pp - Nn * 0.040).reshape(len(zz), len(vv), 3)
    hl_ = grid(Pp, mat='interior_fabric')
    if np.mean(hl_.vertex_normals()[:, 1]) > 0:
        hl_.flip()
    out['headliner'] = hl_

    # bay: inline six, radiator, firewall; ladder frame, solid axles,
    # transfer case and skid plates under it all
    eb = Mesh()
    eb.add(rbox((0, 0.78, 1.38), (0.17, 0.20, 0.42), 0.03, 2, 'metal_bare'))
    eb.add(rbox((0, 1.00, 1.38), (0.13, 0.03, 0.40), 0.015, 2, 'trim_black'))
    eb.add(rbox((0.24, 0.85, 1.38), (0.06, 0.08, 0.32), 0.02, 2, 'metal_bare'))
    eb.add(rbox((0, 0.70, 0.92), (0.17, 0.17, 0.11), 0.04, 2, 'metal_bare'))
    out['engine_block'] = eb
    rad = Mesh()
    rad.add(box((0, 0.82, 1.92), (0.36, 0.18, 0.028), 'trim_black'))
    rad.add(box((0, 1.015, 1.92), (0.37, 0.015, 0.032), 'metal_bare'))
    out['radiator'] = rad
    out['firewall'] = box((0, 0.84, 0.775), (0.60, 0.28, 0.003), 'trim_black')
    ub = Mesh()
    for sg in (1, -1):
        ub.add(box((sg * 0.42, 0.345, 0.0), (0.035, 0.045, 2.05), 'trim_black'))
    for z in (1.95, 0.70, -0.60, -2.10):
        ub.add(box((0, 0.345, z), (0.385, 0.035, 0.03), 'trim_black'))
    ub.add(box((0, 0.300, 1.45), (0.32, 0.004, 0.40), 'metal_bare'))
    for zc in (ZF_AX, ZR_AX):
        ax_ = closed_lathe([(0.0, 0.62), (0.050, 0.62), (0.050, -0.62), (0.0, -0.62)], 20, 'metal_bare')
        ax_.transform(None, (0, WHEEL_Y, zc))
        ub.add(ax_)
        ub.add(rbox((0.08, WHEEL_Y, zc), (0.13, 0.12, 0.11), 0.06, 3, 'metal_bare'))
    ub.add(rbox((0, 0.315, 0.40), (0.12, 0.060, 0.16), 0.03, 2, 'metal_bare'))
    ub.add(P.tube_pipe([(0.03, 0.300, 0.20), (0.06, 0.290, -0.60), (0.08, 0.300, -1.24)], 0.030, n=2))
    ub.add(P.tube_pipe([(0.03, 0.300, 0.56), (0.06, 0.300, 1.24)], 0.028, n=1))
    ub.add(rbox((-0.20, 0.30, -1.75), (0.30, 0.05, 0.25), 0.03, 2, 'metal_bare'))
    out['underbody'] = ub
    ex = Mesh()
    pts = [(0.24, 0.66, 1.30), (0.28, 0.40, 1.08), (0.25, 0.282, 0.85), (0.25, 0.280, -0.70),
           (0.30, 0.282, -1.15), (0.34, 0.285, -1.55), (0.38, 0.290, -2.05), (0.50, 0.292, -2.25)]
    ex.add(P.tube_pipe(pts, 0.026, n=5))
    ex.add(rbox((0.25, 0.282, -0.10), (0.10, 0.026, 0.28), 0.02, 2, 'metal_bare'))
    out['exhaust'] = ex
    for tag, zc in (('F', ZF_AX), ('R', ZR_AX)):
        for s, sg in (('L', 1), ('R', -1)):
            out['wheel_well_%s%s' % (tag, s)] = P.wheel_well((0, WHEEL_Y, zc), 0.440, 0.53, 0.85, sg,
                                                             a_front=92, a_rear=92)

    # markers
    for s, sg in (('L', 1), ('R', -1)):
        for nm, y in (('upper', 1.02), ('lower', 0.66)):
            Pp, Nn = P.surface_point(tube, 0.78, y, sg)
            mk['hinge_door_F%s_%s' % (s, nm)] = (Pp - Nn * 0.035, (sg, 0, 0))
            Pp, Nn = P.surface_point(tube, -0.07, y, sg)
            mk['hinge_door_R%s_%s' % (s, nm)] = (Pp - Nn * 0.035, (sg, 0, 0))
        Pp, Nn = P.surface_point(tube, -0.03, 0.86, sg)
        mk['latch_door_F' + s] = (Pp - Nn * 0.035, (sg, 0, 0))
        Pp, Nn = P.surface_point(tube, -0.86, 0.86, sg)
        mk['latch_door_R' + s] = (Pp - Nn * 0.035, (sg, 0, 0))
        mk['hinge_hood_' + s] = ((sg * 0.68, 1.12, 0.84), (1, 0, 0))
        mk['hinge_tailgate_' + s] = ((sg * 0.45, 1.80, -2.24), (1, 0, 0))
        for k, (z, y) in enumerate(((2.00, 1.09), (1.60, 1.10), (1.20, 1.11), (0.88, 1.12))):
            x = float(K[6][0](np.array([z]))[0]) + 0.03
            mk['bolt_fender_F%s_%d' % (s, k + 1)] = ((sg * x, y - 0.03, z), (1, 0, 0))
    for k, x in enumerate((-0.55, -0.25, 0.25, 0.55)):
        zz = float(front_z(np.array([abs(x)]), np.array([0.60]))[0])
        mk['bolt_bumper_front_%d' % (k + 1)] = ((x, 0.60, zz - 0.03), (1, 0, 0))
        zz = float(rear_z(np.array([abs(x)]), np.array([0.60]))[0])
        mk['bolt_bumper_rear_%d' % (k + 1)] = ((x, 0.60, zz + 0.03), (1, 0, 0))
    mk['latch_hood'] = ((0.0, 1.10, 2.08), (1, 0, 0))
    mk['latch_tailgate'] = ((0.0, 0.70, -2.27), (1, 0, 0))
    return out, extras, mk
