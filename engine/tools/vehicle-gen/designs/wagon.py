"""Estate wagon: the mid-size sedan's nose, doors and wheels with the roof
carried to the tail, long cargo windows behind the rear doors, a near
vertical tailgate carrying the back light between tall lamps, and roof
rails. Derived from designs/sedan.py (the tail from designs/suv_4x4.py's
tailgate layout); only the differences are here.
"""
import numpy as np
from vehgen.design import derive
from vehgen.tube import key

derive('sedan', globals())

NAME = 'wagon'
TITLE = 'Rubble estate wagon'
PAINT = (0.30, 0.31, 0.32)            # silver
PAINT_METALLIC = 0.8
PAINT_ROUGHNESS = 0.30
TAILGATE = True
ZREAR = -2.470

# roof, rail, glass edge and centre line: the sedan's forward of the B-pillar,
# then level to the tail instead of the fastback drop
K[7] = key([(2.45, 0.480), (2.0, 0.560), (1.415, 0.620), (0.88, 0.665), (0.78, 0.705),  # noqa: F821
            (0.0, 0.650), (-0.5, 0.645), (-1.2, 0.648), (-2.0, 0.640), (-2.55, 0.625)],
           [(2.45, 0.798), (2.20, 0.834), (1.80, 0.874), (1.415, 0.908), (0.88, 0.945), (0.78, 0.985),
            (0.30, 1.235), (-0.10, 1.355), (-0.5, 1.385), (-1.2, 1.380), (-2.0, 1.360), (-2.55, 1.340)])
K[8] = key([(2.45, 0.400), (2.0, 0.500), (1.415, 0.560), (0.88, 0.600), (0.78, 0.645),  # noqa: F821
            (0.0, 0.595), (-0.5, 0.590), (-1.2, 0.592), (-2.0, 0.585), (-2.55, 0.570)],
           [(2.45, 0.802), (2.20, 0.838), (1.80, 0.878), (1.415, 0.913), (0.88, 0.950), (0.78, 0.990),
            (0.30, 1.250), (-0.10, 1.385), (-0.5, 1.412), (-1.2, 1.408), (-2.0, 1.388), (-2.55, 1.368)])
K[9] = key(0.0, [(2.45, 0.808), (2.20, 0.845), (1.80, 0.886), (1.415, 0.922), (0.88, 0.958),  # noqa: F821
                 (0.78, 0.995), (0.30, 1.262), (-0.10, 1.415), (-0.40, 1.450), (-1.2, 1.445),
                 (-2.0, 1.425), (-2.55, 1.405)])


def rear_z(x, y):
    """Tailgate face: leans forward 0.12 m from the bumper to the roof's
    rear edge."""
    x = np.abs(x)
    tuck = 0.060 * np.clip((0.44 - y) / 0.14, 0, 1)
    lean = 0.120 * np.clip((y - 0.90) / 0.50, 0, 1) ** 1.4
    corner = 0.090 * (np.clip(x / 0.89, 0, 1)) ** 4
    return ZREAR + tuck + lean + corner


_sedan_assign = assign  # noqa: F821


def assign(zf, zt, lo, hi, side):
    s = side
    I = zf
    # cargo windows from the rear door to the D-pillar; no boot lid
    if I in ('AR1', 'AR2', 'AR3') and lo == 8:
        return 'glass_quarter_' + s
    if I in ('RQ', 'AR1', 'AR2', 'AR3', 'TL') and lo >= 8:
        return 'body_shell'
    return _sedan_assign(zf, zt, lo, hi, side)


def build_layout(cb):
    zi = cb.zi
    cb.end_fit(zi['F'], front_z)  # noqa: F821
    cb.end_fit(zi['T'], rear_z)
    cb.arch(ARCH_F['center'], ARCH_F['R'], (zi['AF1'], zi['AF2'], zi['AF3']), [1, 2, 3, 4],  # noqa: F821
            [10, 22, 40, 62, 90, 118, 145, 172, 196.5], (0.6, 5.2))
    cb.arch(ARCH_R['center'], ARCH_R['R'], (zi['AR1'], zi['AR2'], zi['AR3']), [1, 2, 3, 4],  # noqa: F821
            [-16.5, 8, 35, 62, 90, 118, 145, 162, 177], (0.6, 5.2))


def build_tail(cb):
    """Tall lamps in the D-pillar columns, the tailgate between them with
    the back light, bumper cover below (the 4x4's tail layout)."""
    zi = cb.zi
    iT = zi['T']
    J = cb.J
    nT = (lambda j: cb.nid(iT, j % J))
    left = [nT(k) for k in range(1, 11)]
    right = [nT(J - k) for k in range(1, 11)]
    top = [nT(10), nT(11), nT(12), nT(J - 11), nT(J - 10)]
    y0 = cb.L.node_pos(nT(1))[1]
    xg = 0.66
    bottom = [nT(1), (xg, y0), (0.0, y0), (-xg, y0), nT(J - 1)]
    R = len(left) - 1
    ov = {}
    # rows up to the belt run level; the back-light and frame rows keep the
    # pillar's distances below the roof edge, so no cap row is taller in the
    # middle than at the pillar (cap rows share chord counts with the body
    # bands they meet, all along the car)
    yl = [cb.L.node_pos(n)[1] for n in left]
    xl = [cb.L.node_pos(n)[0] for n in left]
    ytop = [cb.L.node_pos(n)[1] for n in top]
    for r in range(1, R):
        # the D-pillar narrows going up with the roof edge: the tailgate's
        # side line stays 50 mm inside the body's outline
        x1 = min(xg, xl[r] - 0.05)
        for c, x in ((1, x1), (2, 0.0), (3, -x1)):
            if r >= 7:
                yt = np.interp(x, sorted([p_[0] for p_ in map(cb.L.node_pos, top)]),
                               [y for _, y in sorted(zip([cb.L.node_pos(n)[0] for n in top], ytop))])
                ov[(r, c)] = (x, yt - (yl[R] - yl[r]))
            else:
                ov[(r, c)] = (x, yl[r])

    def rpart(r, c):
        if r <= 1:
            return 'bumper_rear'
        if c in (0, 3):
            if r in (3, 4, 5, 6):
                return 'taillight_%s_lens' % ('L' if c == 0 else 'R')
            return 'body_shell'
        if r == 7:
            return 'glass_rear'
        if r == R - 1:
            return 'body_shell'          # roof header above the back light
        return 'tailgate'
    cb.cap('capR', rear_z, -1, left, right, bottom, top, rpart, overrides=ov)


_sedan_parts = parts  # noqa: F821


def parts(cb):
    from vehgen import parts as P
    from vehgen.mesh import Mesh, rbox
    out, extras, mk = _sedan_parts(cb)
    tube = cb.tube
    # roof rails on two posts each side
    rails = Mesh()
    for sg in (1, -1):
        pts = []
        for z in np.linspace(0.05, -2.15, 12):
            v = tube.v_at_x(z, 0.56, 7.0, 9.0)[0]
            p = tube.eval(z, v)[0]
            pts.append((sg * 0.56, p[1] + 0.055, z))
        rails.add(P.tube_pipe(pts, 0.013, mat='trim_black', n=3))
        for z in (0.0, -2.10):
            v = tube.v_at_x(z, 0.56, 7.0, 9.0)[0]
            yb = float(tube.eval(z, v)[0][1])
            rails.add(rbox((sg * 0.56, yb + 0.030, z), (0.018, 0.026, 0.045), 0.010, 2, 'trim_black'))
    out['roof_rails'] = rails
    # the boot-lid markers become tailgate markers
    for s in ('L', 'R'):
        mk.pop('hinge_trunk_' + s, None)
    mk.pop('latch_trunk', None)
    for s, sg in (('L', 1), ('R', -1)):
        mk['hinge_tailgate_' + s] = ((sg * 0.45, 1.36, -2.40), (1, 0, 0))
    mk['latch_tailgate'] = ((0.0, 0.62, float(rear_z(np.array([0.0]), np.array([0.62]))[0]) + 0.03),
                            (1, 0, 0))
    return out, extras, mk
