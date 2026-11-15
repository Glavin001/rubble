"""The art spec's automatic acceptance checks, run on a delivered .glb.

    Blender -b --python check.py -- <file.glb> [--doors FL,FR] [--json out.json]

Runs inside Blender only for mathutils' BVH (nearest-point queries); the glb
is parsed directly, so this checks the file, not the scene that wrote it.
Definitions follow the spec: panel edge p95 <= 30 mm, triangle aspect
(lmax^2 / 2A) <= 6, shut-line gaps measured per 2 cm cell of each panel
border, interpenetration > 1 mm, UV0 overlap rasterised at 2048^2.
"""
import sys
import json
import struct
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
path = argv[0]
doors = (argv[argv.index('--doors') + 1] if '--doors' in argv else 'FL,FR,RL,RR').split(',')
json_out = argv[argv.index('--json') + 1] if '--json' in argv else None
md_out = argv[argv.index('--md') + 1] if '--md' in argv else None
spec_dims = '--sedan' in argv

MATERIALS = ['paint', 'chrome', 'trim_black', 'rubber', 'glass', 'lamp_lens',
             'lamp_emissive', 'interior_fabric', 'interior_plastic', 'metal_bare']
PANEL = ('body_shell', 'hood', 'trunk_lid', 'tailgate', 'door_F', 'door_R', 'fender_', 'bumper_')
INTERIOR = ('dash', 'steering', 'seat', 'console', 'door_trim', 'headliner')


def is_panel(n):
    return n.startswith(PANEL)


def required_parts():
    v = ['body_shell', 'hood', 'fender_FL', 'fender_FR', 'bumper_front', 'bumper_rear', 'grille',
         'glass_windshield', 'glass_rear', 'glass_quarter_L', 'glass_quarter_R', 'mirror_L',
         'mirror_R', 'wiper_L', 'wiper_R', 'dash', 'steering_wheel', 'seat_FL', 'seat_FR',
         'seat_rear', 'console', 'headliner', 'engine_block', 'radiator', 'firewall', 'underbody',
         'exhaust']
    if '--tailgate' in argv:
        v.append('tailgate')
    else:
        v.append('trunk_lid')
    for w in ('FL', 'FR', 'RL', 'RR'):
        for p in ('rim', 'tyre', 'brake_disc', 'wheel_well'):
            v.append(f'{p}_{w}')
    for w in doors:
        for p in ('door', 'glass_door', 'handle', 'door_trim'):
            v.append(f'{p}_{w}')
    for s in ('L', 'R'):
        for l in ('headlight', 'taillight'):
            v += [f'{l}_{s}_lens', f'{l}_{s}_housing']
    return v


def required_markers():
    v = []
    for w in ('FL', 'FR', 'RL', 'RR'):
        v.append(f'wheel_center_{w}')
    for w in doors:
        v += [f'hinge_door_{w}_upper', f'hinge_door_{w}_lower', f'latch_door_{w}']
    v += ['hinge_hood_L', 'hinge_hood_R', 'latch_hood', 'mount_mirror_L', 'mount_mirror_R',
          'seat_H_point_FL', 'seat_H_point_FR']
    if '--tailgate' in argv:
        v += ['hinge_tailgate_L', 'hinge_tailgate_R', 'latch_tailgate']
    else:
        v += ['hinge_trunk_L', 'hinge_trunk_R', 'latch_trunk']
    return v


# -- glb ---------------------------------------------------------------------------
def read_glb(p):
    b = open(p, 'rb').read()
    magic, ver, length = struct.unpack_from('<III', b, 0)
    assert magic == 0x46546C67
    off = 12
    js = None
    bin_ = None
    while off < length:
        clen, ctype = struct.unpack_from('<II', b, off)
        chunk = b[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk)
        elif ctype == 0x004E4942:
            bin_ = chunk
        off += 8 + clen
    return js, bin_


CT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def accessor(js, bin_, i):
    a = js['accessors'][i]
    bv = js['bufferViews'][a['bufferView']]
    dt = CT[a['componentType']]
    n = NC[a['type']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = bv.get('byteStride', 0)
    cnt = a['count']
    if stride and stride != n * np.dtype(dt).itemsize:
        raw = np.frombuffer(bin_, np.uint8, cnt * stride, off).reshape(cnt, stride)
        arr = raw[:, :n * np.dtype(dt).itemsize].copy().view(dt).reshape(cnt, n)
    else:
        arr = np.frombuffer(bin_, dt, cnt * n, off).reshape(cnt, n)
    return arr.astype(np.float64 if dt == np.float32 else np.int64)


def quat_rot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


js, bin_ = read_glb(path)
nodes = js['nodes']
scene = js['scenes'][js.get('scene', 0)]
roots = scene['nodes']
rows = []
fails = 0


def row(name, ok, detail):
    global fails
    if not ok:
        fails += 1
    rows.append((name, ok, detail))


row('one root named `vehicle`', len(roots) == 1 and nodes[roots[0]].get('name') == 'vehicle',
    'roots ' + str([nodes[r].get('name') for r in roots]))
root = nodes[roots[0]]
children = root.get('children', [])
flat = all('children' not in nodes[c] for c in children)
row('flat hierarchy', flat, f'{len(children)} children of `vehicle`')

parts = {}
markers = {}
mats = [m.get('name', '?') for m in js.get('materials', [])]
for c in children:
    nd = nodes[c]
    t = np.array(nd.get('translation', [0, 0, 0]), float)
    r = nd.get('rotation', [0, 0, 0, 1])
    if 'mesh' in nd:
        if nd.get('scale') and np.any(np.array(nd['scale']) < 0):
            row('no negative scale', False, nd['name'])
        me = js['meshes'][nd['mesh']]
        P, N, T, TM, UV, S = [], [], [], [], [], []
        base = 0
        for pr in me['primitives']:
            pos = accessor(js, bin_, pr['attributes']['POSITION'])
            pos = np.array([quat_rot(r, p) for p in pos]) + t if (nd.get('rotation') or nd.get('translation')) else pos
            nrm = accessor(js, bin_, pr['attributes']['NORMAL'])
            uv = (accessor(js, bin_, pr['attributes']['TEXCOORD_0']) if 'TEXCOORD_0' in pr['attributes']
                  else np.zeros((len(pos), 2)))
            idx = accessor(js, bin_, pr['indices']).reshape(-1, 3)
            P.append(pos)
            N.append(nrm)
            UV.append(uv)
            # solid parts tag every vertex: 0 outer skin, 1 edge wall, 2 inner face
            S.append(accessor(js, bin_, pr['attributes']['_SURFACE']).reshape(-1).astype(int)
                     if '_SURFACE' in pr['attributes'] else np.zeros(len(pos), int))
            T.append(idx + base)
            TM += [mats[pr.get('material', 0)]] * len(idx)
            base += len(pos)
        parts[nd['name']] = dict(p=np.concatenate(P), n=np.concatenate(N), t=np.concatenate(T),
                                 tm=np.array(TM), uv=np.concatenate(UV), s=np.concatenate(S))
    else:
        xa = quat_rot(r, np.array([1.0, 0, 0]))
        markers[nd['name']] = (t, xa)

missing = [p for p in required_parts() if p not in parts]
row('every part present', not missing, f'{len(parts)} parts; missing {missing}')
mm = [m for m in required_markers() if m not in markers]
bf = sum(1 for k in markers if k.startswith('bolt_fender_FL_'))
bb = sum(1 for k in markers if k.startswith('bolt_bumper_front_'))
row('every marker present', not mm and 3 <= bf <= 5 and 4 <= bb <= 6,
    f'{len(markers)} markers; missing {mm}; fender bolts {bf}, bumper bolts {bb}')
bad = [m for m in mats if m not in MATERIALS]
row('materials use only the spec names', not bad, f'{len(mats)} materials; others {bad}')
gm = [m for m in js.get('materials', []) if m.get('name') == 'glass']
glass_ok = all(m.get('pbrMetallicRoughness', {}).get('roughnessFactor', 1) <= 0.05 and
               (m.get('alphaMode') == 'BLEND' or 'KHR_materials_transmission' in m.get('extensions', {}))
               for m in gm)
row('glass transparent, roughness <= 0.05', glass_ok and bool(gm), str([(m.get('alphaMode'), m.get('pbrMetallicRoughness', {}).get('roughnessFactor')) for m in gm]))

# -- dimensions ----------------------------------------------------------------------
allp = np.concatenate([v['p'] for k, v in parts.items() if not k.startswith('mirror')])
lo, hi = allp.min(0), allp.max(0)
wc = {w: markers.get('wheel_center_' + w, (np.zeros(3), None))[0] for w in ('FL', 'FR', 'RL', 'RR')}
wb = (wc['FL'][2] + wc['FR'][2]) / 2 - (wc['RL'][2] + wc['RR'][2]) / 2
trf = wc['FL'][0] - wc['FR'][0]
trr = wc['RL'][0] - wc['RR'][0]
fz = (wc['FL'][2] + wc['FR'][2]) / 2
body = np.concatenate([v['p'] for k, v in parts.items()
                       if not k.startswith(('tyre', 'rim', 'brake', 'mirror'))])
gc = body[:, 1].min()
tyre_d = np.mean([parts['tyre_' + w]['p'][:, 1].max() - parts['tyre_' + w]['p'][:, 1].min()
                  for w in wc if 'tyre_' + w in parts])
tmin = [parts['tyre_' + w]['p'][:, 1].min() for w in wc if 'tyre_' + w in parts]
cen_err = []
for w in wc:
    if 'tyre_' + w in parts:
        q = parts['tyre_' + w]['p']
        cen_err.append(np.linalg.norm((q.min(0) + q.max(0)) / 2 - wc[w]))
mid = (wc['FL'][2] + wc['RL'][2]) / 2
dims = dict(length=hi[2] - lo[2], width=hi[0] - lo[0], height=hi[1], wheelbase=wb, track_f=trf,
            track_r=trr, front_overhang=hi[2] - fz, ground_clearance=gc, tyre_d=tyre_d,
            origin_mid_axles=mid, origin_centre_x=(hi[0] + lo[0]) / 2)
row('dimensions (measured)', True, ', '.join(f'{k} {v * 1000:.0f} mm' for k, v in dims.items()))
if spec_dims:
    tgt = dict(length=(4.880, .05), width=(1.860, .03), height=(1.450, .03), wheelbase=(2.830, .02),
               track_f=(1.600, .02), track_r=(1.600, .02), front_overhang=(0.970, .05),
               ground_clearance=(0.140, .015), tyre_d=(0.683, .005))
    off = [f'{k} {dims[k] * 1000:.0f} (target {t * 1000:.0f} ± {e * 1000:.0f})' for k, (t, e) in tgt.items()
           if abs(dims[k] - t) > e]
    row('dimensions within the sedan tolerances', not off, '; '.join(off) or 'all in tolerance')
row('origin on the ground, centred, midway between axles',
    abs(mid) < 0.005 and abs(dims['origin_centre_x']) < 0.005 and lo[1] > -0.002,
    f'axle midpoint z {mid * 1000:.1f} mm, centre x {dims["origin_centre_x"] * 1000:.1f} mm, min y {lo[1] * 1000:.1f} mm')
row('tyres touch y = 0; wheel centres within 5 mm of markers',
    all(abs(t) < 0.001 for t in tmin) and all(e < 0.005 for e in cen_err),
    f'tyre min y {[round(t * 1000, 1) for t in tmin]} mm; centre error {[round(e * 1000, 1) for e in cen_err]} mm')

# -- topology ---------------------------------------------------------------------------
nonman = {}
openb = {}
where_inv = {}
where_asp = {}
zero = {}
inverted = {}
edges_panel = []
aspect_panel = []
over6 = {}
groups = {'body': 0, 'wheel': {}, 'interior': 0, 'other': 0}
total = 0
for name, d in parts.items():
    p, t, n = d['p'], d['t'], d['n']
    total += len(t)
    if name.startswith(('rim', 'tyre', 'brake')):
        w = name.split('_')[-1]
        groups['wheel'][w] = groups['wheel'].get(w, 0) + len(t)
    elif is_panel(name):
        groups['body'] += len(t)
    elif name.startswith(INTERIOR):
        groups['interior'] += len(t)
    else:
        groups['other'] += len(t)
    key = np.round(p / 1e-6).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    tw = inv[t]
    e = np.concatenate([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]])
    e = np.sort(e, 1)
    e = e[e[:, 0] != e[:, 1]]
    _, cnt = np.unique(e, axis=0, return_counts=True)
    nonman[name] = int((cnt > 2).sum())
    openb[name] = int((cnt == 1).sum())
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    zero[name] = int((area < 1e-10).sum())
    vn = n[t].mean(1)
    invm = np.einsum('ij,ij->i', cr, vn) < 0
    inverted[name] = int(invm.sum())
    if invm.any():
        where_inv[name] = np.round(((a + b + c) / 3)[invm][0], 3).tolist()
    if is_panel(name):
        L = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], 1)
        paint = d['tm'] == 'paint'
        edges_panel.append(L[paint].reshape(-1))
        lmax = L.max(1)
        asp = lmax * lmax / (2 * np.maximum(area, 1e-14))
        aspect_panel.append(asp[paint])
        k = int((asp[paint] > 6).sum())
        if k:
            over6[name] = k
            w = np.argmax(np.where(paint, asp, 0))
            cen = ((a + b + c) / 3)[paint & (asp > 6)]
            from collections import Counter
            cl = Counter(tuple((np.round(x / 0.1) / 10).tolist()) for x in cen).most_common(3)
            where_asp[name] = (round(float(asp[w]), 1), np.round((a[w] + b[w] + c[w]) / 3, 3).tolist(), cl)
row('no non-manifold edges', sum(nonman.values()) == 0, str({k: v for k, v in nonman.items() if v}))
solid = [k for k in parts if parts[k]['s'].any()]
row('solid parts are closed (no open border)', all(openb[k] == 0 for k in solid),
    f'{len(solid)} solid parts; open edges {dict((k, openb[k]) for k in solid if openb[k])}')
row('no zero-area triangles', sum(zero.values()) == 0, str({k: v for k, v in zero.items() if v}))
row('no inverted normals', sum(inverted.values()) == 0, str({k: (v, where_inv.get(k)) for k, v in inverted.items() if v}))
E = np.concatenate(edges_panel)
row('panel edge p95 <= 30 mm', np.percentile(E, 95) <= 0.030,
    f'p5 {np.percentile(E, 5) * 1000:.1f}, p50 {np.percentile(E, 50) * 1000:.1f}, p95 {np.percentile(E, 95) * 1000:.1f} mm')
A = np.concatenate(aspect_panel)
row('panel triangle aspect <= 6:1', A.max() <= 6.0,
    f'p99 {np.percentile(A, 99):.2f}, max {A.max():.2f}; over 6: {over6}; worst {where_asp}')
wmax = max(groups['wheel'].values()) if groups['wheel'] else 0
ok = (150000 <= groups['body'] <= 300000 and wmax <= 15000 and groups['interior'] <= 80000
      and groups['other'] <= 60000 and total <= 500000)
row('triangle budget', ok, f"body {groups['body']} (150k-300k), wheel max {wmax} (<= 15k), "
    f"interior {groups['interior']} (<= 80k), lamps/glass/trim/bay {groups['other']} (<= 60k), total {total} (<= 500k)")

# -- gaps and interpenetration between panels (BVH) -----------------------------------------
from mathutils.bvhtree import BVHTree  # noqa: E402
from mathutils import Vector  # noqa: E402
panels = [k for k in parts if is_panel(k)]
trees = {k: BVHTree.FromPolygons([tuple(x) for x in parts[k]['p']], [tuple(x) for x in parts[k]['t']])
         for k in panels}
cell = 0.02
cells = {}
pen_max = (0.0, '')


def border_zone(d, rings=3):
    """Vertices within `rings` of an open border (the hem): where shut lines are.
    A solid part has no open border: its shut lines are where its skin meets its edge
    walls, so the zone is the skin and the walls (not the inner face) within `rings` of
    the skin's border, the same rows as a sheet's hem and skin edge."""
    p, t = d['p'], d['t']
    solid = d['s'].any()
    t_all = t[(d['s'][t] < 2).all(1)]
    t = t[(d['s'][t] == 0).all(1)]
    key = np.round(p / 1e-6).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    tw = inv[t]
    e = np.concatenate([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]])
    es = np.sort(e, 1)
    u, cnt = np.unique(es, axis=0, return_counts=True)
    zone = np.zeros(inv.max() + 1, bool)
    zone[u[cnt == 1].reshape(-1)] = True
    if solid:
        # spread over the skin and the walls two rings: the skin's border rows and the hem
        # roll (deeper wall rows part where the shut line turns a corner)
        tw = inv[t_all]
        e = np.concatenate([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]])
        es = np.sort(e, 1)
        rings = 2
    for _ in range(rings):
        hit = zone[es].any(1)
        zone[es[hit].reshape(-1)] = True
    return zone[inv]


zones = {k: border_zone(parts[k]) for k in panels}
ztrees = {}
for k in panels:
    t = parts[k]['t']
    sel = zones[k][t].all(1)
    if sel.any():
        ztrees[k] = BVHTree.FromPolygons([tuple(x) for x in parts[k]['p']], [tuple(x) for x in t[sel]])
for a_ in panels:
    p = parts[a_]['p']
    n = parts[a_]['n']
    others = [k for k in panels if k != a_]
    for i in range(len(p)):
        q = Vector(p[i])
        if zones[a_][i]:
            best = (1e9, None)
            for b_ in others:
                if b_ not in ztrees:
                    continue
                loc, nrm, fi, dist = ztrees[b_].find_nearest(q, 0.012)
                if loc is not None and dist < best[0]:
                    best = (dist, b_)
            if best[1] is not None:
                ck = (tuple(np.floor(p[i] / cell).astype(int)), min(a_, best[1]), max(a_, best[1]))
                cells[ck] = min(cells.get(ck, 1e9), best[0])
        for b_ in others:
            loc, nrm, fi, dist = trees[b_].find_nearest(q, 0.012)
            if loc is None:
                continue
            into = -(q - loc).dot(nrm)
            if 0.001 < into < 0.01 and Vector(n[i]).dot(nrm) > 0.5:
                tri = parts[b_]['t'][fi]
                P3 = parts[b_]['p'][tri]
                nn = np.array(nrm)
                proj = np.array(q) - nn * np.dot(np.array(q) - P3[0], nn)
                if np.linalg.norm(proj - np.array(loc)) < 1e-5 and into > pen_max[0]:
                    pen_max = (into, f'{a_} into {b_} at {np.round(p[i], 3).tolist()}')
# a shut line's gap in a cell is the least gap seen in it or its 26
# neighbours: cells holding only a flange (7 mm under the skin) otherwise
# read as wide gaps
pairs_ = {}
for (c, a_, b_), d in cells.items():
    pairs_.setdefault((a_, b_), {})[c] = d
cells2 = {}
for (a_, b_), cm in pairs_.items():
    for c, d in cm.items():
        best = d
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    k = (c[0] + dx, c[1] + dy, c[2] + dz)
                    if k in cm and cm[k] < best:
                        best = cm[k]
        cells2[(c, a_, b_)] = best
cells = cells2
g = np.array(list(cells.values())) if cells else np.array([0.0])
tight = {}
wide = {}
where_gap = {}
for (c, a_, b_), d in cells.items():
    if d < 0.003:
        tight[f'{a_}/{b_}'] = tight.get(f'{a_}/{b_}', 0) + 1
        where_gap.setdefault(f'<3 {a_}/{b_}', np.round(np.array(c) * cell, 2).tolist())
    if d > 0.006:
        wide[f'{a_}/{b_}'] = wide.get(f'{a_}/{b_}', 0) + 1
        where_gap.setdefault(f'>6 {a_}/{b_}', np.round(np.array(c) * cell, 2).tolist())
row('shut-line gaps 3-6 mm', np.percentile(g, 5) >= 0.003 and np.percentile(g, 95) <= 0.006,
    f'{len(g)} border cells: min {g.min() * 1000:.1f}, p5 {np.percentile(g, 5) * 1000:.1f}, p50 {np.percentile(g, 50) * 1000:.1f}, '
    f'p95 {np.percentile(g, 95) * 1000:.1f} mm; under 3 mm: {dict(sorted(tight.items(), key=lambda x: -x[1])[:6])}; '
    f'over 6 mm: {dict(sorted(wide.items(), key=lambda x: -x[1])[:6])}; e.g. {where_gap}')
row('no panel interpenetration over 1 mm', pen_max[0] <= 0.001, f'deepest {pen_max[0] * 1000:.1f} mm {pen_max[1]}')

# all part pairs: triangles that cross (informational at any depth)
alltrees = {k: BVHTree.FromPolygons([tuple(x) for x in parts[k]['p']], [tuple(x) for x in parts[k]['t']])
            for k in parts}
names = sorted(parts)
bb = {k: (parts[k]['p'].min(0) - 0.002, parts[k]['p'].max(0) + 0.002) for k in names}
cross = {}
where_cross = {}
for i, a_ in enumerate(names):
    for b_ in names[i + 1:]:
        if np.any(bb[a_][1] < bb[b_][0]) or np.any(bb[b_][1] < bb[a_][0]):
            continue
        ov = alltrees[a_].overlap(alltrees[b_])
        if ov:
            cross[f'{a_}/{b_}'] = len(ov)
            # where: centroid of the crossing triangles of the first part
            ta = np.array([i_ for i_, _ in ov])
            where_cross[f'{a_}/{b_}'] = np.round(parts[a_]['p'][parts[a_]['t'][ta]].mean((0, 1)), 2).tolist()
top_cross = sorted(cross.items(), key=lambda x: -x[1])[:12]
row('no part pairs cross (any depth)', not cross,
    f'{len(cross)} pairs: ' + str({k: (n, where_cross[k]) for k, n in top_cross}))

# -- UV0 on painted panels --------------------------------------------------------------------
res = 2048
cover = np.zeros((res, res), np.int32)
over = 0
covered = 0
inside = True
pairs = {}
uv_where = {}
for pi, name in enumerate(panels):
    d = parts[name]
    uv = d['uv']
    if np.any(uv < -1e-6) or np.any(uv > 1 + 1e-6):
        inside = False
    sel = d['tm'] == 'paint'
    T = d['t'][sel]
    U = uv[T] * res
    for tri in U:
        a, b, c = tri
        ar = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(ar) < 1e-6:
            continue
        x0, x1 = int(max(np.floor(tri[:, 0].min()), 0)), int(min(np.ceil(tri[:, 0].max()), res - 1))
        y0, y1 = int(max(np.floor(tri[:, 1].min()), 0)), int(min(np.ceil(tri[:, 1].max()), res - 1))
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)

        def e(p, q):
            return (q[0] - p[0]) * (Y - p[1]) - (q[1] - p[1]) * (X - p[0])
        w0, w1, w2 = e(b, c), e(c, a), e(a, b)
        ins = ((w0 > 0) & (w1 > 0) & (w2 > 0)) if ar > 0 else ((w0 < 0) & (w1 < 0) & (w2 < 0))
        if not ins.any():
            continue
        sub = cover[y0:y1 + 1, x0:x1 + 1]
        dup = ins & (sub > 0)
        if dup.any():
            over += int(dup.sum())
            for o in np.unique(sub[dup]):
                key = f'{panels[o - 1]}/{name}'
                pairs[key] = pairs.get(key, 0) + 1
                if key not in uv_where:
                    uv_where[key] = (np.round(tri.mean(0) / res, 4).tolist())
        covered += int((ins & (sub == 0)).sum())
        sub[ins] = pi + 1
row('UV0 non-overlapping on painted panels', over == 0 and inside,
    f'{covered} texels at 2048^2, {over} covered twice {dict(sorted(pairs.items(), key=lambda x: -x[1])[:6])} at uv {uv_where}; all in [0,1]: {inside}')

# texture memory: we ship no textures
tex = len(js.get('images', []))
row('texture memory within 64 MB', tex == 0, f'{tex} images (materials are flat PBR factors)')

print()
print(path)
print()
print('| check | result | detail |')
print('|---|---|---|')
for n_, ok, d in rows:
    print(f'| {n_} | {"pass" if ok else "FAIL"} | {d} |')
print(f'\n{fails} failing')
if md_out:
    with open(md_out, 'w') as f:
        f.write('| check | result | detail |\n|---|---|---|\n')
        for n_, ok, d in rows:
            f.write(f'| {n_} | {"pass" if ok else "FAIL"} | {d} |\n')
if json_out:
    json.dump(dict(file=path, dims={k: float(v) for k, v in dims.items()},
                   groups=dict(body=groups['body'], wheel=groups['wheel'], interior=groups['interior'],
                               other=groups['other'], total=total),
                   rows=[dict(check=n_, ok=bool(ok), detail=d) for n_, ok, d in rows]),
              open(json_out, 'w'), indent=1)
