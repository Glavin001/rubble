"""Police interceptor: the mid-size sedan in black on steel wheels, with a
roof light bar, a push bar on the nose, an A-pillar spotlight and a boot-lid
antenna. Derived from designs/sedan.py; only the differences are here.
"""
import numpy as np
from vehgen.design import derive

derive('sedan', globals())

NAME = 'police_sedan'
TITLE = 'Rubble police sedan'
PAINT = (0.010, 0.010, 0.012)          # gloss black
PAINT_METALLIC = 0.2
PAINT_ROUGHNESS = 0.18
RIM_STYLE = 'steel'

RED = (0.85, 0.03, 0.02, 1.0)
BLUE = (0.02, 0.10, 0.90, 1.0)

_sedan_parts = parts  # noqa: F821 (from the base design)


def parts(cb):
    from vehgen import parts as P
    from vehgen.mesh import Mesh, rbox, closed_lathe
    out, extras, mk = _sedan_parts(cb)
    tube = cb.tube

    def roof_y(z, x):
        v = tube.v_at_x(z, abs(x), 7.5, 9.0)[0]
        return float(tube.eval(z, v)[0][1])

    # light bar across the roof over the front seats: black base on four
    # feet, red lens on the left, blue on the right
    zc = -0.28
    yb = roof_y(zc, 0.0) + 0.045
    bar = Mesh()
    lens = Mesh()
    bar.add(rbox((0, yb, zc), (0.60, 0.016, 0.125), 0.012, 2, 'trim_black'))
    for x in (-0.45, 0.45):
        for dz in (-0.08, 0.08):
            ys = roof_y(zc + dz, x)
            bar.add(rbox((x, (ys + 0.004 + yb) / 2, zc + dz), (0.022, (yb - ys - 0.004) / 2, 0.022), 0.006, 1,
                         'trim_black'))
    for sg, tint in ((1, RED), (-1, BLUE)):
        l_ = rbox((sg * 0.300, yb + 0.049, zc), (0.282, 0.030, 0.112), 0.024, 3, 'lamp_lens')
        l_.tint(tint)
        lens.add(l_)
    bar.add(rbox((0, yb + 0.049, zc), (0.010, 0.030, 0.105), 0.005, 1, 'trim_black'))
    out['lightbar'] = bar
    out['lightbar_lens'] = lens

    # push bar: two uprights following the nose, two cross tubes, rubber
    # pads, brackets back to the bumper beam
    pb = Mesh()
    ys = np.linspace(0.30, 0.74, 9)
    for x in (-0.32, 0.32):
        zs = front_z(np.full(len(ys), abs(x)), ys) + 0.075  # noqa: F821
        pb.add(P.tube_pipe(np.c_[np.full(len(ys), x), ys, zs], 0.024, mat='trim_black', n=3))
        pb.add(rbox((x, 0.52, float(front_z(np.array([abs(x)]), np.array([0.52]))[0]) + 0.112),  # noqa: F821
                    (0.030, 0.17, 0.010), 0.008, 1, 'rubber'))
        zf = float(front_z(np.array([abs(x)]), np.array([0.40]))[0])  # noqa: F821
        pb.add(rbox((x, 0.40, zf + 0.030), (0.030, 0.025, 0.040), 0.006, 1, 'trim_black'))
    for y in (0.45, 0.68):
        xs = np.linspace(-0.32, 0.32, 9)
        zs = front_z(np.abs(xs), np.full(len(xs), y)) + 0.075  # noqa: F821
        zs = np.maximum(zs, zs.max() - 0.010)
        pb.add(P.tube_pipe(np.c_[xs, np.full(len(xs), y), zs], 0.020, mat='trim_black', n=3))
    out['push_bar'] = pb

    # A-pillar spotlight (driver's side) and a boot-lid antenna
    sp = Mesh()
    base, bn = P.surface_point(tube, 0.66, 0.985, 1, vrange=(5.5, 7.5))
    c = base + bn * 0.065 + np.array([0, 0.02, 0])
    can = closed_lathe([(0.0, 0.040), (0.050, 0.040), (0.056, 0.0), (0.040, -0.075), (0.0, -0.075)],
                       28, 'chrome')
    can.transform(P.rot_to(np.array([0.15, 0.05, 1.0])), c)
    sp.add(can)
    sp.add(P.tube_pipe([base + bn * 0.004, base + bn * 0.035, c - np.array([0, 0.0, 0.03])], 0.010,
                       mat='chrome', n=2))
    out['spotlight'] = sp
    lens_ = closed_lathe([(0.0, 0.006), (0.044, 0.006), (0.044, 0.002), (0.0, 0.002)], 28, 'lamp_lens')
    lens_.transform(P.rot_to(np.array([0.15, 0.05, 1.0])), c + np.array([0.15, 0.05, 1.0]) /
                    np.linalg.norm([0.15, 0.05, 1.0]) * 0.040)
    out['lightbar_lens'].add(lens_)
    za, xa = -2.05, 0.35
    ya = float(tube.eval(za, tube.v_at_x(za, xa, 6.0, 9.0)[0])[0][1])
    ant = Mesh()
    ant.add(rbox((xa, ya + 0.014, za), (0.018, 0.010, 0.018), 0.006, 1, 'trim_black'))
    ant.add(P.tube_pipe([(xa, ya + 0.024, za), (xa, ya + 0.35, za - 0.02), (xa, ya + 0.70, za - 0.06)],
                        0.0025, mat='trim_black', n=3))
    out['antenna'] = ant
    return out, extras, mk
