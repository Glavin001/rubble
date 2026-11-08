"""Design module -> finished part meshes."""
import numpy as np
from .carbody import CarBody
from .panels import panel, glass_from
from .mesh import Mesh

CLOSURE_INNER = {'hood': 0.030, 'trunk_lid': 0.030}
CHART_ISLAND = {'tube': 0}


def body_parts(D, h=0.02, log=print):
    tube = D.tube()
    cb = CarBody(tube, D.ZLINES, D.VLINES, h=h)
    D.build_layout(cb)
    cb.build_nodes()
    cb.tube_blocks(D.assign)
    D.build_caps(cb)
    cb.L.solve()
    zmin = min(z for _, z in D.ZLINES) - 0.1
    zmax = max(z for _, z in D.ZLINES) + 0.1
    tube.girth_table(zmin, zmax)
    groups = {}
    for bl in cb.L.blocks:
        g, Pp, P3 = cb.L.mesh_block(bl)
        groups.setdefault(bl.part, []).append((g, bl.chart))
    parts = {}
    skins = {}
    from collections import Counter
    from .panels import boundary_loops
    bkeys = Counter()
    for name, blocks in groups.items():
        if name.startswith('glass') or name.endswith('_lens'):
            sk = Mesh()
            for g, _ in blocks:
                sk.add(g.copy())
            sk.weld()
            ks = set()
            for lp in boundary_loops(sk):
                for k in np.round(sk.v[lp] / 1e-6).astype(np.int64):
                    ks.add(tuple(k))
            bkeys.update(ks)
    shared_glass = {k for k, c in bkeys.items() if c > 1}
    # border vertices a panel shares with another panel: true shut lines
    pkeys = Counter()
    for name, blocks in groups.items():
        if name.startswith('glass') or name.endswith('_lens'):
            continue
        sk = Mesh()
        for g, _ in blocks:
            sk.add(g.copy())
        sk.weld()
        ks = set()
        for lp in boundary_loops(sk):
            for k in np.round(sk.v[lp] / 1e-6).astype(np.int64):
                ks.add(tuple(k))
        pkeys.update(ks)
    shut = {k for k, c in pkeys.items() if c > 1}
    for name, blocks in groups.items():
        sk = Mesh()
        for g, _ in blocks:
            sk.add(g.copy())
        skins[name] = sk.weld()
        charts = sorted({c for _, c in blocks})
        isl = {c: k for k, c in enumerate(charts)}
        bm = [(g, isl[c]) for g, c in blocks]
        if name.startswith('glass'):
            parts[name] = glass_from(bm, shared=shared_glass)
            continue
        if name.endswith('_lens'):
            lens = glass_from(bm, depth=0.013, grow=0.003)
            lens.set_mat('lamp_lens')
            tint = getattr(D, 'LENS_TINT', {}).get(name, (1, 1, 1, 1))
            parts[name] = lens.tint(tint)
            continue
        inner = None
        if name in CLOSURE_INNER:
            inner = CLOSURE_INNER[name]
        elif name.startswith('door_'):
            inner = 0.035
        parts[name] = panel(bm, inner=inner, shut=shut)
        log(f'  {name}: {parts[name].ntris()} tris')
    cb.skins = skins
    return parts, cb
