"""Generic generators for the non-panel parts. Each returns a Mesh in glTF
coordinates; designs choose styles and placements."""
import numpy as np
from .mesh import (Mesh, closed_lathe, lathe, rbox, box, sweep, rrect, circle, loft,
                   orient_outward, rot_x, rot_y, rot_z, catmull, grid, MAT)

AX_TO_Z = rot_y(-np.pi / 2)    # lathe axis +X -> +Z (forward)
AX_TO_MZ = rot_y(np.pi / 2)    # lathe axis +X -> -Z (rearward)


# -- wheels ----------------------------------------------------------------------

def tyre(D_out, W, rim_d, rim_w, segs=96, grooves=3):
    """Closed tyre: beads seated inside the rim flanges, sidewalls bulging
    to the section width, rounded shoulders, circumferential grooves."""
    R = D_out / 2
    r = rim_d / 2
    hw = W / 2
    hb = rim_w / 2 - 0.015            # bead axial position, inside the flanges
    rb = r - 0.0015                   # bead seat just over the barrel
    rf = r + 0.026                    # clears the flange top
    prof = [(rb, -hb), (rf, -hb)]
    for t in np.linspace(0, 1, 9)[1:]:
        rr = rf + (R - 0.030 - rf) * t
        a = -hb - (hw - hb) * np.sin(np.pi / 2 * min(1.0, t * 1.6)) + 0.010 * t * t
        prof.append((rr, a))
    for k in np.linspace(0, 1, 5)[1:]:
        th = np.pi / 2 * k
        prof.append((R - 0.030 + 0.030 * np.sin(th), -hw + 0.010 + 0.022 - 0.022 * np.cos(th)))
    gw = 0.008
    prof.append((R, -hw + 0.034))
    for g in range(grooves):
        c = -hw + 0.034 + (W - 0.068) * (g + 1) / (grooves + 1)
        prof += [(R, c - gw), (R - 0.008, c - gw * 0.6), (R - 0.008, c + gw * 0.6), (R, c + gw)]
    prof.append((R, hw - 0.034))
    for k in np.linspace(0, 1, 5)[:-1]:
        th = np.pi / 2 * (1 - k)
        prof.append((R - 0.030 + 0.030 * np.sin(th), hw - 0.010 - 0.022 + 0.022 * np.cos(th)))
    for t in np.linspace(1, 0, 9)[1:]:
        rr = rf + (R - 0.030 - rf) * t
        a = hb + (hw - hb) * np.sin(np.pi / 2 * min(1.0, t * 1.6)) - 0.010 * t * t
        prof.append((rr, a))
    prof += [(rf, hb), (rb, hb)]
    return closed_lathe(prof, segs, 'rubber')


def rim_magnum(rim_d, rim_w, segs=72, spokes=5):
    """Late-60s polished five-spoke: a chrome dished face with five dark
    window cut-outs (the spokes are the chrome left between them), a narrow
    polished lip and a small domed centre cap."""
    r = rim_d / 2
    hw = rim_w / 2
    m = Mesh()
    barrel = [(r + 0.012, -hw - 0.006), (r + 0.012, -hw + 0.004), (r - 0.004, -hw + 0.008),
              (r - 0.010, -hw + 0.030), (r - 0.010, hw - 0.020), (r - 0.004, hw - 0.006),
              (r - 0.008, hw - 0.006), (r - 0.014, hw - 0.020), (r - 0.014, -hw + 0.030),
              (r - 0.008, -hw + 0.006)]
    m.add(closed_lathe(barrel, segs, 'metal_bare'))
    lip = [(r + 0.014, hw - 0.010), (r + 0.016, hw + 0.002), (r + 0.008, hw + 0.010),
           (r - 0.026, hw + 0.004), (r - 0.030, hw - 0.004), (r - 0.010, hw - 0.012)]
    m.add(closed_lathe(lip, segs, 'chrome'))
    r0, r1 = 0.046, r - 0.033
    a_in, a_out = hw - 0.024, hw - 0.004

    def ax(rr):
        return a_in + (a_out - a_in) * np.clip((rr - r0) / (r1 - r0), 0, 1) ** 0.8
    dish = [(r0, a_in), (r1, a_out), (r1, a_out - 0.006), (r0, a_in - 0.006)]
    m.add(closed_lathe(dish, segs, 'chrome'))
    cap = [(0.0, a_in + 0.020), (0.022, a_in + 0.018), (0.040, a_in + 0.008), (0.048, a_in - 0.002),
           (0.0, a_in - 0.002)]
    m.add(closed_lathe(cap, 48, 'chrome'))
    # windows: between spoke centre lines, spokes 30 mm wide at the cap and
    # 44 mm at the rim
    ri, ro = r0 + 0.016, r1 - 0.012
    for k in range(spokes):
        c0 = 2 * np.pi * k / spokes
        c1 = 2 * np.pi * (k + 1) / spokes
        pts = []
        for rr, wid in ((ri, 0.015), (ro, 0.022)):
            pass
        # left edge (offset from spoke at c0), outer arc, right edge, inner arc
        def edge(c, sgn, rr, wid):
            d = np.array([np.cos(c), np.sin(c)])
            n = np.array([-np.sin(c), np.cos(c)]) * sgn
            # point at radius rr along the offset line
            t = np.sqrt(max(rr * rr - wid * wid, 1e-9))
            return d * t + n * wid
        poly = []
        for rr, wid in ((ri, 0.015), (ro, 0.022)):
            poly.append(edge(c0, 1, rr, wid))
        a0_ = np.arctan2(*poly[-1][::-1])
        a1_ = np.arctan2(*edge(c1, -1, ro, 0.022)[::-1])
        if a1_ < a0_:
            a1_ += 2 * np.pi
        for a_ in np.linspace(a0_, a1_, 9)[1:-1]:
            poly.append(np.array([ro * np.cos(a_), ro * np.sin(a_)]))
        poly.append(edge(c1, -1, ro, 0.022))
        poly.append(edge(c1, -1, ri, 0.015))
        b0_ = np.arctan2(*poly[-1][::-1])
        b1_ = np.arctan2(*poly[0][::-1])
        if b1_ > b0_:
            b1_ -= 2 * np.pi
        for a_ in np.linspace(b0_, b1_, 5)[1:-1]:
            poly.append(np.array([ri * np.cos(a_), ri * np.sin(a_)]))
        poly = np.array(poly)
        rad = np.linalg.norm(poly, axis=1)
        # a single dark panel just proud of the dish (open surface)
        from .mesh import zigzag_cap
        front = np.c_[ax(rad) + 0.0010, poly[:, 0], poly[:, 1]]
        T = zigzag_cap(np.arange(len(front)), front, np.array([1.0, 0, 0]))
        m.add(Mesh.from_faces(front, tris=T, mat='trim_black'))
    return m


def rim_alloy(rim_d, rim_w, segs=72, pairs=5):
    """Modern cast wheel: five split spokes in a concave face, polished lip."""
    r = rim_d / 2
    hw = rim_w / 2
    m = Mesh()
    barrel = [(r + 0.012, -hw - 0.006), (r + 0.012, -hw + 0.004), (r - 0.004, -hw + 0.008),
              (r - 0.012, -hw + 0.040), (r - 0.012, hw - 0.020), (r - 0.004, hw - 0.006),
              (r - 0.009, hw - 0.006), (r - 0.017, hw - 0.020), (r - 0.017, -hw + 0.040),
              (r - 0.009, -hw + 0.006)]
    m.add(closed_lathe(barrel, segs, 'metal_bare'))
    lip = [(r + 0.013, hw - 0.010), (r + 0.015, hw + 0.002), (r + 0.004, hw + 0.008),
           (r - 0.018, hw + 0.004), (r - 0.020, hw - 0.004), (r - 0.008, hw - 0.012)]
    m.add(closed_lathe(lip, segs, 'chrome'))
    hub = [(0.0, hw - 0.020), (0.030, hw - 0.021), (0.065, hw - 0.028), (0.075, hw - 0.040),
           (0.075, hw - 0.052), (0.0, hw - 0.052)]
    m.add(closed_lathe(hub, 40, 'chrome'))
    m.add(closed_lathe([(0.0, hw - 0.018), (0.026, hw - 0.018), (0.028, hw - 0.022), (0.0, hw - 0.022)],
                       24, 'trim_black'))
    for k in range(pairs):
        for d in (-1, 1):
            a = 2 * np.pi * k / pairs + d * 0.11
            path = []
            for t in np.linspace(0, 1, 8):
                rr = 0.060 + (r - 0.020 - 0.060) * t
                aa = a - d * 0.07 * t
                ax = hw - 0.040 + 0.034 * t ** 0.7
                path.append([ax, rr * np.cos(aa), rr * np.sin(aa)])
            path = np.array(path)
            sc = np.stack([np.linspace(0.026, 0.020, len(path)), np.full(len(path), 1.0)], 1)
            sp = sweep(path, rrect(1.0, 0.016, 0.005, 2), mat='chrome', up=(1, 0, 0), scale=sc,
                       twist_up=np.tile([1.0, 0, 0], (len(path), 1)))
            m.add(orient_outward(sp))
    for k in range(5):
        a = 2 * np.pi * (k + 0.5) / 5
        nut = closed_lathe([(0.0, 0.006), (0.008, 0.005), (0.009, -0.006), (0.0, -0.006)], 12, 'chrome')
        nut.transform(None, (hw - 0.026, 0.045 * np.cos(a), 0.045 * np.sin(a)))
        m.add(nut)
    return m


def rim_steel(rim_d, rim_w, segs=64):
    """Painted steel wheel with a small hub cap (work trucks)."""
    r = rim_d / 2
    hw = rim_w / 2
    m = Mesh()
    prof = [(r + 0.012, -hw - 0.006), (r + 0.012, -hw + 0.004), (r - 0.006, -hw + 0.008),
            (r - 0.010, hw - 0.030), (r - 0.002, hw - 0.012), (r + 0.012, hw - 0.006),
            (r + 0.012, hw + 0.004), (r - 0.012, hw - 0.002), (r - 0.016, hw - 0.030),
            (r - 0.040, hw - 0.034), (0.11, hw - 0.010), (0.10, hw - 0.020), (r - 0.044, hw - 0.046),
            (r - 0.020, hw - 0.044), (r - 0.014, -hw + 0.010), (r - 0.010, -hw + 0.006)]
    m.add(closed_lathe(prof, segs, 'metal_bare'))
    m.add(closed_lathe([(0.0, hw + 0.006), (0.060, hw + 0.002), (0.085, hw - 0.012), (0.0, hw - 0.012)],
                       40, 'chrome'))
    return m


def brake_disc(d=0.28, segs=48):
    r = d / 2
    m = closed_lathe([(0.075, -0.052), (r, -0.052), (r, -0.028), (0.075, -0.028)], segs, 'metal_bare')
    m.add(closed_lathe([(0.040, -0.030), (0.085, -0.030), (0.085, 0.004), (0.040, 0.004)], 32, 'metal_bare'))
    # caliper at the rear, inboard
    cal = rbox((-0.040, 0.0, 0.0), (0.030, 0.040, 0.075), 0.012, 2, 'trim_black')
    cal.transform(rot_x(np.radians(-110)), (0, 0, 0))
    cal.v[:, 0] -= 0.0
    # place on the disc's rim, wrapping it
    cal2 = rbox((-0.040, r - 0.022, 0.0), (0.024, 0.030, 0.060), 0.010, 2, 'trim_black')
    cal2.transform(rot_x(np.radians(-120)))
    m.add(cal2)
    return m


def wheel(D, side, z_axle):
    """(rim, tyre, disc, centre) for one corner. side +1 left, -1 right."""
    x = side * D.TRACK / 2
    c = np.array([x, D.WHEEL_Y, z_axle])
    ty = tyre(D.TYRE_D, D.TYRE_W, D.RIM_D, D.RIM_W)
    style = getattr(D, 'RIM_STYLE', 'magnum')
    ri = (rim_magnum(D.RIM_D, D.RIM_W) if style == 'magnum' else
          rim_alloy(D.RIM_D, D.RIM_W) if style == 'alloy' else
          rim_steel(D.RIM_D, D.RIM_W))
    di = brake_disc(getattr(D, 'DISC_D', 0.28))
    out = []
    for mm in (ri, ty, di):
        if side < 0:
            mm.mirror_x()
        mm.transform(None, c)
        out.append(mm)
    return out[0], out[1], out[2], c


# -- lamps --------------------------------------------------------------------------

def rot_to(axis):
    """Rotation taking +X to `axis` (up kept as close to +Y as possible)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    up = np.array([0.0, 1.0, 0.0])
    z = np.cross(a, up)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 0.0, 1.0])
    z /= np.linalg.norm(z)
    y = np.cross(z, a)
    return np.stack([a, y, z], 1)


def graph_normal(fz, x, y, facing=1, h=1e-4):
    """Outward normal of the end face z = fz(x, y)."""
    fx = (fz(np.array([x + h]), np.array([y])) - fz(np.array([x - h]), np.array([y])))[0] / (2 * h)
    fy = (fz(np.array([x]), np.array([y + h])) - fz(np.array([x]), np.array([y - h])))[0] / (2 * h)
    n = np.array([-fx, -fy, 1.0]) * facing
    return n / np.linalg.norm(n)


def round_lamp(center, R, facing=1, tint=(1, 1, 1, 1), depth=0.07, axis=None):
    """(lens, housing) of a round sealed-beam lamp facing +z (facing=1) or
    along `axis`."""
    depth = max(depth, 0.014)
    Rot = rot_to(axis) if axis is not None else (AX_TO_Z if facing > 0 else AX_TO_MZ)
    Rl = R - 0.005
    lens = closed_lathe([(0.0, 0.016), (Rl * 0.5, 0.013), (Rl * 0.85, 0.007), (Rl, 0.0),
                         (Rl, -0.004), (0.0, -0.004)], 48, 'lamp_lens')
    lens.transform(Rot, center)
    lens.tint(tint)
    hous = Mesh()
    bezel = closed_lathe([(R - 0.002, 0.002), (R + 0.010, 0.006), (R + 0.016, 0.0),
                          (R + 0.014, -0.010), (R - 0.002, -0.008)], 48, 'chrome')
    cup = closed_lathe([(R - 0.002, -0.008), (R * 0.7, -depth * 0.6), (0.02, -depth),
                        (0.0, -depth), (0.0, -depth - 0.004), (0.02, -depth - 0.004),
                        (R * 0.7 + 0.004, -depth * 0.6 - 0.004), (R + 0.002, -0.010)], 40, 'lamp_emissive')
    hous.add(bezel).add(cup)
    hous.transform(Rot, center)
    return lens, hous


# -- helpers ----------------------------------------------------------------------

def surface_point(tube, z, y, side=1, vrange=(1.5, 5.5)):
    v = tube.v_at_height(z, y, *vrange)
    if side < 0:
        v = tube.VM - v
    P, N = tube.eval(z, v, with_normal=True)
    return P[0], N[0]


def path_on_surface(tube, zs, ys, side=1, off=0.0, vrange=(1.5, 5.5)):
    zs = np.atleast_1d(zs)
    ys = np.broadcast_to(np.atleast_1d(ys), zs.shape)
    v = tube.v_at_height(zs, ys, *vrange)
    if side < 0:
        v = tube.VM - v
    P, N = tube.eval(zs, v, with_normal=True)
    return P + N * off, N


def door_handle(tube, z, y, side=1, length=0.13):
    P, N = surface_point(tube, z, y, side)
    N = N / np.linalg.norm(N)
    Ry = rot_y(np.arctan2(-N[2], N[0]))
    m = rbox((0, 0, 0), (0.009, 0.013, length / 2), 0.006, 2, 'chrome')
    plate = rbox((0, 0, 0), (0.0015, 0.020, length / 2 + 0.012), 0.0014, 1, 'trim_black')
    m.transform(Ry, P + N * 0.019)
    plate.transform(Ry, P + N * 0.0060)
    return m.add(plate)


def sport_mirror(base, side=1, mat='paint'):
    """Bullet racing mirror: pad on the door top, short stem, rounded head."""
    b = np.asarray(base, float)
    m = Mesh()
    pad = rbox(b + [-side * 0.006, 0.017, 0], (0.018, 0.005, 0.038), 0.004, 2, mat)
    stem = sweep(np.array([b + [0, 0.020, 0.0], b + [side * 0.025, 0.050, -0.005],
                           b + [side * 0.050, 0.070, -0.010]]), circle(0.008, 12), mat='chrome')
    head_c = b + [side * 0.080, 0.085, -0.015]
    head = rbox(head_c, (0.050, 0.036, 0.042), 0.034, 4, mat)
    glass = rbox(head_c + [0, 0, -0.043], (0.040, 0.027, 0.0015), 0.0014, 1, 'chrome')
    m.add(pad).add(orient_outward(stem)).add(head).add(glass)
    return m


def wiper(points, mat='trim_black'):
    """Arm + blade along a polyline lying on the glass."""
    P = np.asarray(points, float)
    arm = sweep(P, rrect(0.012, 0.006, 0.002, 1), mat=mat)
    blade = sweep(P + [0, -0.004, 0], rrect(0.008, 0.010, 0.002, 1), mat='rubber')
    return orient_outward(arm).add(orient_outward(blade))


# -- interior -----------------------------------------------------------------------

def bucket_seat(x, z, h_floor, mat='interior_fabric', high_back=True):
    """Low-back bucket with an integrated head restraint (late-60s style)."""
    m = Mesh()
    top = h_floor + 0.175
    m.add(rbox((x, top - 0.065, z), (0.25, 0.065, 0.27), 0.05, 3, mat))
    back_h = 0.27
    back = rbox((0, 0, 0), (0.25, back_h, 0.06), 0.05, 3, mat)
    lean = np.radians(-14)
    back.transform(rot_x(lean), (x, top + back_h - 0.03, z - 0.30))
    m.add(back)
    if high_back:
        hr = rbox((0, 0, 0), (0.13, 0.085, 0.05), 0.04, 2, mat)
        hr.transform(rot_x(lean), (x, top + 2 * back_h + 0.05, z - 0.30 - 0.15))
        m.add(hr)
    m.add(box((x, h_floor + 0.035, z), (0.20, 0.03, 0.22), 'metal_bare'))
    return m


def steering_wheel(center, R=0.19, tilt_deg=22, mat='interior_plastic', column=0.30):
    rim = closed_lathe([(R - 0.014, 0.0), (R, 0.012), (R + 0.014, 0.0), (R, -0.012)], 64, mat)
    rim = closed_lathe([(R + 0.014 * np.cos(a), 0.014 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)], 64, mat)
    hub = closed_lathe([(0.0, 0.02), (0.05, 0.015), (0.055, -0.03), (0.0, -0.03)], 24, mat)
    m = Mesh().add(rim).add(hub)
    for k, a in enumerate((np.radians(-90), np.radians(30), np.radians(150))):
        p0 = np.array([0.0, 0.05 * np.cos(a), 0.05 * np.sin(a)])
        p1 = np.array([0.006, (R - 0.010) * np.cos(a), (R - 0.010) * np.sin(a)])
        sp = sweep(np.linspace(p0, p1, 4), rrect(0.022, 0.008, 0.003, 1), mat='chrome', up=(1, 0, 0))
        m.add(orient_outward(sp))
    col = closed_lathe([(0.0, -0.02), (0.032, -0.02), (0.032, -column), (0.0, -column)], 20, mat)
    m.add(col)
    # axis +X -> towards the driver (-Z), tilted up
    R1 = AX_TO_MZ
    R2 = rot_x(np.radians(tilt_deg))
    m.transform(R2 @ R1, center)
    return m


# -- underbody and bay ---------------------------------------------------------------

def wheel_well(center, R, x_in, x_out, side=1, a_front=100, a_rear=100):
    """Half-drum liner over a wheel: roof (cylinder) + inner wall."""
    prof = [(R, x_out), (R, x_in), (R * 0.55, x_in)]
    m = lathe(prof, 40, mat='trim_black', a0=np.radians(-a_rear), a1=np.radians(a_front))
    # lathe: axis x, angle 0 -> +y, +90 -> +z. Profile above runs outward ->
    # inward; faces should look at the wheel (inwards).
    m.weld(1e-7)
    c = np.asarray(center, float)
    if side < 0:
        m.mirror_x()
    # make faces point towards the axle
    tri = m.triangles()
    a, b, cc = m.v[tri[:, 0]], m.v[tri[:, 1]], m.v[tri[:, 2]]
    fn = np.cross(b - a, cc - a)
    cen = (a + b + cc) / 3
    to_axis = -cen * np.array([0, 1, 1])
    if np.mean(np.einsum('ij,ij->i', fn, to_axis)) < 0:
        m.flip()
    m.transform(None, c)
    return m


def tube_pipe(points, r, mat='metal_bare', n=6, segs=14):
    P = catmull(points, n)
    return orient_outward(sweep(P, circle(r, segs), mat=mat))


def plan_outline(tube, fz, y, sign=1, z_guess=2.3, side_len=0.40, n_end=28, n_side=10):
    """Plan outline (x >= 0) of the body at height y round one end: the end
    face z = fz(x, y) from the centre line to the corner, then along the side
    (the tube at height y) away from the end. sign +1 front, -1 rear."""
    def xs(z):
        z = np.atleast_1d(z)
        v = tube.v_at_height(z, np.full(len(z), y), 1.0, 5.0)
        return tube.eval(z, v)[:, 0]
    z = z_guess
    for _ in range(30):
        x = xs(z)[0]
        z = float(fz(np.array([x]), np.array([y]))[0])
    xc = xs(z)[0]
    xe = np.linspace(0, xc, n_end)
    ze = fz(xe, np.full(n_end, y))
    zsd = z - sign * np.linspace(0, side_len, n_side + 1)[1:]
    xsd = xs(zsd)
    P = np.concatenate([np.c_[xe, ze], np.c_[xsd, zsd]])
    return P


def offset_outline(P, d, sign=1):
    """Offset a centre->corner->side outline outward by d (2D), keeping the
    distance at corners (miter, capped)."""
    seg = np.diff(P, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    sn = np.stack([-seg[:, 1], seg[:, 0]], 1) * sign
    vn = np.zeros_like(P)
    vn[:-1] += sn
    vn[1:] += sn
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    sc = np.ones(len(P))
    sc[1:-1] = 1.0 / np.clip(np.einsum('ij,ij->i', vn[1:-1], sn[1:]), 0.6, 1.0)
    return P + vn * (d * sc)[:, None]


def chaikin(P, it=3):
    P = np.asarray(P, float)
    for _ in range(it):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q.append(P[-1])
        P = np.array(Q)
    return P


def _seg_dist(Q, A, B):
    """Distance from points Q (n,2) to segments A->B (m,2): (n,)"""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((Q[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    C = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(Q[:, None, :] - C, axis=2).min(1)


def densify(poly, maxlen):
    """Closed 2D polygon with no edge longer than maxlen."""
    P = np.asarray(poly, float)
    out = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / maxlen)))
        for t in np.arange(k) / k:
            out.append(a + (b - a) * t)
    return np.array(out)


def fit_path(outlines, d, sign=1, it=3, step=None):
    """A smooth path at least d outside every outline (2D polylines, the
    body's plan at several heights): miter offset of the middle one, rounded
    by Chaikin, pushed out wherever it comes closer than d."""
    base = offset_outline(outlines[len(outlines) // 2], d, sign)
    segs = [(o[:-1], o[1:]) for o in outlines]
    for _ in range(it):
        base = chaikin(base, 1)
        T = np.gradient(base, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        N = np.stack([-T[:, 1], T[:, 0]], 1) * sign
        dist = np.min([_seg_dist(base, a, b) for a, b in segs], axis=0)
        base = base + N * np.maximum(0, d - dist)[:, None]
    if step:
        from .curves import resample_polyline
        L = np.linalg.norm(np.diff(base, axis=0), axis=1).sum()
        base = resample_polyline(base, max(2, int(np.ceil(L / step))))
    return base


def molding(tube, zv, off=0.002, sec=(0.010, 0.005), closed=False, n=4):
    from .mesh import sweep, rrect, catmull, orient_outward
    zv = np.asarray(zv, float)
    pts = catmull(zv, n, closed=closed)
    P, N = tube.eval(pts[:, 0], pts[:, 1], with_normal=True)
    P = P + N * off
    m = sweep(P, rrect(sec[0], sec[1], min(sec) * 0.45, 2), mat='chrome', closed=closed,
              twist_up=N)
    return orient_outward(m)


def surface_bump(tube, z0, z1, y0, y1, h, side, mat='paint'):
    """A side scoop: a rounded wedge standing on the skin, deepest (h) at its
    open front z0, tapering to 5 mm at z1."""
    from .mesh import loft, orient_outward
    rings = []
    ym = (y0 + y1) / 2
    for t in np.linspace(0, 1, 22):
        z = z0 + (z1 - z0) * t
        th = 0.010 + (h - 0.010) * (1 - t) ** 0.9
        hh = (y1 - y0) * (1 - 0.35 * t)
        P_, N_ = surface_point(tube, z, ym, side)
        N_ = N_ / np.linalg.norm(N_)
        up = np.array([0.0, 1.0, 0.0])
        up = up - N_ * np.dot(up, N_)
        up /= np.linalg.norm(up)
        # superellipse section, sampled evenly by arc length
        ang = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        ca, sa = np.cos(ang), np.sin(ang)
        e = 0.5
        raw = np.stack([th / 2 * np.sign(ca) * np.abs(ca) ** e, hh / 2 * np.sign(sa) * np.abs(sa) ** e], 1)
        from .curves import resample_polyline
        sec = resample_polyline(np.concatenate([raw, raw[:1]]), 16)[:-1]
        ring = P_[None] + N_[None] * (sec[:, 0:1] + th / 2 - 0.0012) + up[None] * sec[:, 1:2]
        rings.append(ring)
    return orient_outward(loft(np.array(rings), mat=mat))


