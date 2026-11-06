"""UV0 atlas for the painted panels: every island keeps its metric scale (even
texel density) and is shelf-packed into [0, 1]^2 without overlap."""
import numpy as np
from .mesh import MAT, cube_uv

PANEL_PREFIX = ('body_shell', 'hood', 'trunk_lid', 'door_', 'fender_', 'bumper_', 'tailgate',
                'bed_', 'cab_')


def is_panel(name):
    return name.startswith(PANEL_PREFIX)


def pack(parts, margin=0.012):
    isl = []
    for name, m in parts.items():
        if not is_panel(name):
            continue
        # anything without an island gets a box projection
        noq = m.qi < 0
        nt = m.ti < 0
        if noq.any() or nt.any():
            tmp = m.copy()
            cube_uv(tmp, island_base=50000)
            m.quv[noq] = tmp.quv[noq]
            m.qi[noq] = tmp.qi[noq]
            m.tuv[nt] = tmp.tuv[nt]
            m.ti[nt] = tmp.ti[nt]
        ids = np.unique(np.concatenate([m.qi, m.ti]))
        for i in ids:
            qs = np.where(m.qi == i)[0]
            ts = np.where(m.ti == i)[0]
            uv = np.concatenate([m.quv[qs].reshape(-1, 2), m.tuv[ts].reshape(-1, 2)])
            lo, hi = uv.min(0), uv.max(0)
            isl.append(dict(part=name, id=i, qs=qs, ts=ts, lo=lo, size=hi - lo))
    area = sum(float(np.prod(i['size'] + margin)) for i in isl)
    widest = max(float(i['size'][0]) for i in isl)
    W = max(np.sqrt(area * 1.25), widest + 2 * margin)
    isl.sort(key=lambda i: -i['size'][1])
    x = y = margin
    row_h = 0.0
    for i in isl:
        w, h = i['size']
        if x + w + margin > W:
            x = margin
            y += row_h + margin
            row_h = 0.0
        i['off'] = np.array([x, y])
        x += w + margin
        row_h = max(row_h, h)
    H = y + row_h + margin
    S = max(W, H)
    for i in isl:
        m = parts[i['part']]
        sh = i['off'] - i['lo']
        m.quv[i['qs']] = (m.quv[i['qs']] + sh) / S
        m.tuv[i['ts']] = (m.tuv[i['ts']] + sh) / S
    return S


def normalise_other(parts):
    """Non-panel parts: box projection scaled into [0, 1] per part."""
    for name, m in parts.items():
        if is_panel(name):
            continue
        cube_uv(m, 0)
        uv = np.concatenate([m.quv.reshape(-1, 2), m.tuv.reshape(-1, 2)])
        if not len(uv):
            continue
        lo, hi = uv.min(0), uv.max(0)
        s = max(float((hi - lo).max()), 1e-6)
        m.quv = (m.quv - lo) / s
        m.tuv = (m.tuv - lo) / s
