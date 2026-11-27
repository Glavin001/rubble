#!/usr/bin/env python3
"""Part outlines that make shape differences between Rubble and the reference footage
obvious (engine/docs/crash-reference.md, "Outlines").

Inputs:
- Rubble: `outlines.json` from the CPU tool
  (`cargo run --release -p engine-softbody-lab --example crash_outlines -- S02`, written to
  `engine/out/crash-reference/outlines/<ID>-edge/` with its own stills of the collision
  surface from each scenario camera at each moment);
- the recorder's stills (`crash-ref-record.mjs --all-cams`, `<moment>-<camera>.png`);
- tracings of the reference frames (`engine/docs/crash-reference/tracings/*.json`, points
  in normalised image coordinates; correct them with `trace.html`);
- the reference frames themselves (third-party footage, never committed; read from
  `~/Development/rubble-artifacts/crash-reference/`).

    python3 engine/tools/crash-reference/outlines.py all [S01 S02]
        [--out engine/out/crash-reference/outlines] [--stills DIR] [--frames DIR]
    python3 engine/scripts/with-gpu-lock.py --limit 300 -- \\
        python3 engine/tools/crash-reference/outlines.py rasterise DIR [DIR ...]

`all` writes SVGs (overlays on the recorded and the CPU stills, onion skins, annotated
reference frames, common-frame and change figures) and `shape-numbers.md`; `rasterise`
turns every SVG in the folders into a PNG with headless Chrome over http (the only SVG
renderer this machine has; Chrome is run without the GPU, but under the lock as every
headless Chrome is). Standard library only.

The common frame (side view): x along the car (forward), y up. S01's frames show the whole
car: origin the rear wheel's centre, unit the wheelbase in that source's own "before"
frame. S02's frames show only the front half (no rear wheel): origin the front door's
front edge at the sill (the cabin does not deform there), unit the front wheel's radius
in the source's "before" frame. Shape numbers are reported comparisons, not criteria.
"""
import base64
import http.server
import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1]
sys.path.insert(0, str(HERE))
import sheet  # noqa: E402  (the reference moment -> Rubble moment rule)

ARTIFACTS = Path.home() / 'Development' / 'rubble-artifacts' / 'crash-reference'
TRACINGS = ENGINE / 'docs' / 'crash-reference' / 'tracings'

COLOURS = {
    'hood': '#ff9a1f', 'fender_l': '#38b6ff', 'fender_r': '#38b6ff', 'bumper_f': '#e04be0', 'bumper_r': '#e04be0',
    'door_fl': '#58e070', 'door_fr': '#58e070', 'door_rl': '#2fa04a', 'door_rr': '#2fa04a', 'trunk': '#e8e040',
    'roof': '#b8b8ff', 'greenhouse': '#8f8fd8', 'sill_l': '#ff5050', 'sill_r': '#ff5050', 'wheel': '#f0f0f0',
}
SIDE_GROUPS = ['bumper_f', 'hood', 'fender_l', 'door_fl', 'door_rl', 'greenhouse', 'roof', 'sill_l', 'trunk']
TOP_GROUPS = ['bumper_f', 'hood', 'fender_l', 'fender_r', 'door_fl', 'door_fr', 'greenhouse', 'roof', 'trunk']
SOURCE_COLOURS = {'real': '#ff4d4d', 'beamng': '#3d8bff', 'rubble': '#2ee07a'}
SOURCE_NAMES = {'real': 'real test', 'beamng': 'BeamNG', 'rubble': 'Rubble'}
# A crease: a turn of the hood line sharper than this (degrees) at one of its points.
CREASE_DEG = 20.0


# ---------------------------------------------------------------- SVG helpers

def data_uri(path):
    ext = Path(path).suffix.lower().lstrip('.')
    mime = {'png': 'image/png', 'webp': 'image/webp', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg'}[ext]
    return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode()


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def pts(line):
    return ' '.join(f'{x:.1f},{y:.1f}' for x, y in line)


def polyline(line, colour, width=2.0, dash=None, opacity=1.0, closed=False):
    if len(line) < 2:
        return ''
    tag = 'polygon' if closed else 'polyline'
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<{tag} points="{pts(line)}" fill="none" stroke="{colour}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{d}/>')


def circle(c, r, colour, width=2.0, dash=None, opacity=1.0):
    if c is None or r is None:
        return ''
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<circle cx="{c[0]:.1f}" cy="{c[1]:.1f}" r="{r:.1f}" fill="none" stroke="{colour}" stroke-width="{width}" opacity="{opacity}"{d}/>'


def label(x, y, text, colour, size=15):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{colour}" font-size="{size}" font-family="Helvetica, Arial, sans-serif" '
            f'font-weight="700" stroke="#000" stroke-width="3" paint-order="stroke">{esc(text)}</text>')


def svg(w, h, body, bg=None, bg_opacity=1.0, view=None):
    vb = view or (0, 0, w, h)
    img = f'<image href="{bg}" x="0" y="0" width="{w}" height="{h}" opacity="{bg_opacity}"/>' if bg else ''
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{vb[0]} {vb[1]} {vb[2]} {vb[3]}">'
            f'<rect x="{vb[0]}" y="{vb[1]}" width="{vb[2]}" height="{vb[3]}" fill="#10141c"/>{img}{"".join(body)}</svg>\n')


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------- Rubble overlays

def rubble_lines(cam, group, which):
    g = cam['parts'].get(group)
    return g[which]['lines'] if g else []


def outline_layer(cam, groups, which, dash=None, width=2.5, opacity=1.0, labels=True, colour=None):
    body = []
    for g in groups:
        lines = rubble_lines(cam, g, which)
        c = colour or COLOURS.get(g, '#fff')
        for ln in lines:
            body.append(polyline(ln, c, width, dash, opacity))
        if labels and lines and which == 'now':
            ln = max(lines, key=len)
            top = min(ln, key=lambda p: p[1])
            body.append(label(top[0] + 4, top[1] - 6, g.replace('_', ' '), c))
    hc = cam['hood_centreline'][which]
    if hc:
        body.append(polyline(hc, colour or '#ffffff', width + 1.5, dash, opacity))
        if which == 'now':
            body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLOURS["hood"]}" stroke="#000"/>' for x, y in hc]
    for wh in cam['wheels']:
        if (wh['name'].endswith('l') and groups is SIDE_GROUPS) or groups is TOP_GROUPS:
            w = wh[which]
            body.append(circle(w['c'], w['r'], colour or COLOURS['wheel'], width * 0.8, dash, opacity))
    return body


def legend(lines, x=16, y=24):
    out = [f'<rect x="{x - 8}" y="{y - 18}" width="440" height="{22 * len(lines) + 10}" rx="6" fill="#000" opacity="0.55"/>']
    for k, (text, colour) in enumerate(lines):
        out.append(label(x, y + 22 * k, text, colour, 15))
    return out


def rubble_overlays(sid, doc, out, stills):
    w, h = doc['projection']['width'], doc['projection']['height']
    cpu_dir = out / f'{sid}-edge'
    made = []
    moments = doc['moments']
    before = next((m for m in moments if m['name'] == 'before'), moments[0])
    for cam_name in [c['name'] for c in doc['cameras'] if c['name'] in ('side', 'top')]:
        groups = SIDE_GROUPS if cam_name == 'side' else TOP_GROUPS
        for m in moments:
            cam = m['bodies'][0]['cameras'][cam_name]
            head = [(f'{sid} {m["name"]} ({m["after_impact_ms"]:+g} ms), {cam_name} view, CPU reference, edge contact', '#fff'),
                    ('solid: part outline now; dashed: rest shape carried with the cabin', '#ddd'),
                    ('white: hood centreline (dots: its nodes); circles: wheels', '#ddd')]
            body = outline_layer(cam, groups, 'ghost', dash='7 5', width=2, opacity=0.8, labels=False) + outline_layer(cam, groups, 'now')
            for bg_name, bg_path in (('recorded', stills / f'{m["name"]}-{cam_name}.png'), ('cpu', cpu_dir / f'{m["name"]}-{cam_name}.png')):
                if not bg_path.exists():
                    continue
                note = ('still: the web page (WebGPU) at the same step; its physics is not bit-equal to the CPU run' if bg_name == 'recorded'
                        else 'still: the CPU run\'s collision surface from the same camera (exact)')
                made.append(write(out / sid / 'overlay' / f'{m["name"]}-{cam_name}-on-{bg_name}.svg',
                                  svg(w, h, body + legend(head + [(note, '#9fd')]), data_uri(bg_path), 0.85)))
            if m is not before:
                b = before['bodies'][0]['cameras'][cam_name]
                onion = outline_layer(b, groups, 'now', width=2, opacity=0.9, labels=False, colour='#00e5ff') + outline_layer(cam, groups, 'now')
                bg_path = cpu_dir / f'{m["name"]}-{cam_name}.png'
                made.append(write(out / sid / 'overlay' / f'{m["name"]}-{cam_name}-onion.svg',
                                  svg(w, h, onion + legend([(f'{sid} onion skin, {cam_name} view: before (cyan) and {m["name"]} (colours)', '#fff'),
                                                            ('image space: the car also moved between the two', '#ddd')]),
                                      data_uri(bg_path) if bg_path.exists() else None, 0.45)))
    return made


# ---------------------------------------------------------------- tracings

def load_tracings(sid):
    out = []
    for p in sorted(TRACINGS.glob(f'{sid}_*.json')):
        out.append(json.loads(p.read_text()))
    return out


def to_px(t, p):
    return [p[0] * t['image']['width'], p[1] * t['image']['height']]


def annotated_frame(t, frames, out):
    src = frames / t['frame']
    W, H = t['image']['width'], t['image']['height']
    body = []
    for half, hv in t['halves'].items():
        col = SOURCE_COLOURS[half]
        body.append(polyline([to_px(t, p) for p in hv['crop']], col, 3, '12 8', 0.9, closed=True))
        for name, line in hv['lines'].items():
            ln = [to_px(t, p) for p in line]
            body.append(polyline(ln, col if name != 'hood' else '#ffb000', 5 if name == 'hood' else 3.5))
            body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="#000"/>' for x, y in ln]
            body.append(label(ln[0][0] + 6, ln[0][1] - 8, name.replace('_', ' '), col, 22))
        for name, wh in hv['wheels'].items():
            c, rim = to_px(t, wh['c']), to_px(t, wh['rim'])
            body.append(circle(c, math.dist(c, rim), col, 3.5))
            body.append(label(c[0] - 40, c[1], f'{name} wheel', col, 22))
    head = [(f'{t["frame"]}: {t["moment"]} ({t["camera"]}); tracing: {t.get("traced_by", "")}', '#fff'),
            ('red: real test half; blue: BeamNG half; dashed: each half\'s crop', '#ddd'),
            ('approximate first pass: correct it in trace.html', '#ffb000')]
    leg = [f'<g transform="scale(1.6)">{"".join(legend(head))}</g>']
    bg = data_uri(src) if src.exists() else None
    return write(out / 'reference' / (Path(t['frame']).stem + '-traced.svg'), svg(W, H, body + leg, bg, 1.0))


# ---------------------------------------------------------------- the common frame

def car_xy(t, p, front_left=True):
    x, y = to_px(t, p)
    return [-x if front_left else x, -y]


class Shape:
    """One source at one moment in its own car-like axes (x forward, y up, raw units)."""

    def __init__(self):
        self.lines = {}     # name -> polyline (feature lines; tracings' names)
        self.loops = {}     # name -> [polyline] (Rubble outlines, drawn only)
        self.wheels = {}    # front/rear -> (centre, radius)
        self.door = None    # front door's front edge at the sill
        self.roof_rear = None


def shape_from_tracing(t, half):
    hv = t['halves'][half]
    s = Shape()
    fl = t.get('front', 'left') == 'left'
    for name, line in hv['lines'].items():
        s.lines[name] = [car_xy(t, p, fl) for p in line]
    for name, wh in hv['wheels'].items():
        c, rim = car_xy(t, wh['c'], fl), car_xy(t, wh['rim'], fl)
        s.wheels[name] = (c, math.dist(c, rim))
    if 'door_front_edge' in s.lines:
        s.door = min(s.lines['door_front_edge'], key=lambda p: p[1])
    if 'roofline' in s.lines:
        s.roof_rear = min(s.lines['roofline'], key=lambda p: p[0])
    return s


def envelope(points, along, take, bins):
    """Per bin of coordinate `along` (0: x, 1: y), the point with the largest (take=+1) or
    smallest (take=-1) other coordinate; in order of `along`."""
    other = 1 - along
    vals = [p[along] for p in points]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return []
    out = []
    for k in range(bins):
        a, b = lo + (hi - lo) * k / bins, lo + (hi - lo) * (k + 1) / bins
        sel = [p for p in points if a - 1e-9 <= p[along] <= b + 1e-9]
        if sel:
            out.append(max(sel, key=lambda p: take * p[other]))
    return out


def shape_from_rubble(m):
    """Rubble's outlines in its car frame, and from them the tracings' feature lines: the
    hood's centreline; the greenhouse's front edge (windscreen), the roof's top (roofline),
    the left fender's top, the bumper's and the front door's front edges, the left sill's
    bottom (each the envelope of the part's outline on the side the footage shows)."""
    car = m['bodies'][0]['car_frame']
    s = Shape()
    for g, v in car['parts'].items():
        s.loops[g] = v['now']['lines']
    s.lines['hood'] = car['hood_centreline']['now']
    pts_of = lambda g: [p for ln in s.loops.get(g, []) for p in ln]  # noqa: E731
    feature = {'windscreen': ('greenhouse', 1, 1, 6), 'roofline': ('roof', 0, 1, 8), 'fender_top': ('fender_l', 0, 1, 6),
               'bumper_front': ('bumper_f', 1, 1, 5), 'door_front_edge': ('door_fl', 1, 1, 5), 'sill': ('sill_l', 0, -1, 6)}
    for name, (g, along, take, bins) in feature.items():
        p = pts_of(g)
        if name == 'windscreen' and p:
            # (its front half: the pillars and roof behind it reach other heights)
            mid = sorted(q[0] for q in p)[len(p) // 2]
            p = [q for q in p if q[0] >= mid]
        if p:
            line = envelope(p, along, take, bins)
            s.lines[name] = sorted(line, key=lambda q: -q[0]) if along == 0 else line
    roof = pts_of('roof')
    if roof:
        s.roof_rear = min(roof, key=lambda p: p[0])
    # The door's corner as the cabin carries it (its ghost): the door panel itself can open
    # or bulge, the cabin's corner where it hangs does not move in the crash.
    door = [p for ln in car['parts'].get('door_fl', {}).get('ghost', {}).get('lines', []) for p in ln]
    if door:
        lo = min(p[1] for p in door)
        s.door = max((p for p in door if p[1] <= lo + 0.1), key=lambda p: p[0])
    for wh in car['wheels']:
        if wh['name'] == 'wheel_fl':
            s.wheels['front'] = (wh['now'], wh['r'])
        if wh['name'] == 'wheel_rl':
            s.wheels['rear'] = (wh['now'], wh['r'])
    return s


def normaliser(mode, before):
    """(origin of a shape, unit) from a source's before shape."""
    if mode == 'rear_wheel':
        unit = math.dist(before.wheels['front'][0], before.wheels['rear'][0])
        return (lambda s: s.wheels['rear'][0]), unit
    return (lambda s: s.door), before.wheels['front'][1]


def normalise(s, origin, unit):
    o = origin(s)
    f = lambda p: [(p[0] - o[0]) / unit, (p[1] - o[1]) / unit]  # noqa: E731
    n = Shape()
    n.lines = {k: [f(p) for p in v] for k, v in s.lines.items()}
    n.loops = {k: [[f(p) for p in ln] for ln in v] for k, v in s.loops.items()}
    n.wheels = {k: (f(c), r / unit) for k, (c, r) in s.wheels.items()}
    n.door = f(s.door) if s.door else None
    n.roof_rear = f(s.roof_rear) if s.roof_rear else None
    return n


# ---------------------------------------------------------------- shape numbers

def chord(line):
    return math.dist(line[0], line[-1])


def fold_height(line, chord0):
    (x0, y0), (x1, y1) = line[0], line[-1]
    L = math.hypot(x1 - x0, y1 - y0) or 1e-9
    nx, ny = -(y1 - y0) / L, (x1 - x0) / L
    if ny < 0:
        nx, ny = -nx, -ny
    d = [(p[0] - x0) * nx + (p[1] - y0) * ny for p in line]
    best = max(d, key=abs)
    return best / chord0


def creases(line):
    turns = []
    for a, b, c in zip(line, line[1:], line[2:]):
        t = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
        t = (t + 180) % 360 - 180
        turns.append(t)
    sharp = [t for t in turns if abs(t) > CREASE_DEG]
    changes = sum(1 for a, b in zip(sharp, sharp[1:]) if (a > 0) != (b > 0))
    return len(sharp), changes


def angle(line):
    (x0, y0), (x1, y1) = line[0], line[-1]
    return math.degrees(math.atan2(abs(y1 - y0), abs(x1 - x0)))


def numbers(before, after, mode):
    """The reported shape numbers of one source (normalised shapes)."""
    r = {}
    hb, ha = before.lines.get('hood'), after.lines.get('hood')
    if hb and ha and len(ha) >= 2:
        r['hood fold height (/ chord before)'] = fold_height(ha, chord(hb))
        r['hood chord after / before'] = chord(ha) / chord(hb)
        n, ch = creases(ha)
        r[f'hood creases (turn > {CREASE_DEG:g} deg)'] = n
        r['hood crease sign changes'] = ch
        r['hood points'] = len(ha)
    fb, fa = before.wheels.get('front'), after.wheels.get('front')
    if fb and fa:
        r['front wheel setback'] = fb[0][0] - fa[0][0]
        r['front wheel drop (+ down)'] = fb[0][1] - fa[0][1]
    wb, wa = before.lines.get('windscreen'), after.lines.get('windscreen')
    if wb and wa:
        r['windscreen angle before (deg)'] = angle(wb)
        r['windscreen angle change (deg)'] = angle(wa) - angle(wb)
    if mode == 'rear_wheel' and before.roof_rear and after.roof_rear:
        r['rear ride height change (roof rear over rear wheel)'] = after.roof_rear[1] - before.roof_rear[1]
    # Orientation checks: the roofline's and the sill's tilt, + rear up (nose down).
    for name in ('roofline', 'sill'):
        lb, la = before.lines.get(name), after.lines.get(name)
        if lb and la and len(lb) >= 2 and len(la) >= 2:
            r[f'check: {name} tilt change (deg, + rear up)'] = rear_up(la) - rear_up(lb)
    return r


def rear_up(line):
    """A line's tilt (deg) from its front end to its rear end, + when the rear is higher."""
    f, b = max(line, key=lambda p: p[0]), min(line, key=lambda p: p[0])
    return math.degrees(math.atan2(b[1] - f[1], f[0] - b[0]))


# ---------------------------------------------------------------- figures

def plot_frame(shapes, title, subtitle, unit_name, size=(1200, 620)):
    """Shapes: [(source, Shape, style)]; style 'solid' or 'ghost'. Front at the left, as filmed."""
    W, H = size
    xs, ys = [], []
    for _, s, _ in shapes:
        for ln in list(s.lines.values()) + [p for v in s.loops.values() for p in v]:
            xs += [p[0] for p in ln]
            ys += [p[1] for p in ln]
        for c, r in s.wheels.values():
            xs += [c[0] - r, c[0] + r]
            ys += [c[1] - r, c[1] + r]
    if not xs:
        return svg(W, H, [label(20, 40, title + ': nothing to draw', '#fff')])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.06 * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    top = 90
    sc = min((W - 40) / (x1 - x0), (H - top - 30) / (y1 - y0))
    # front at the left: screen x grows toward the rear
    tx = lambda p: [20 + (x1 - p[0]) * sc, top + (y1 - p[1]) * sc]  # noqa: E731
    body = []
    # axes through the origin, and a grid of 0.1 units
    g = 0.1 if unit_name == 'wheelbases' else 0.5
    k = math.floor(x0 / g)
    while k * g <= x1:
        a = tx([k * g, y0])
        b = tx([k * g, y1])
        body.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#2a3344" stroke-width="{2 if k == 0 else 1}"/>')
        k += 1
    k = math.floor(y0 / g)
    while k * g <= y1:
        a = tx([x0, k * g])
        b = tx([x1, k * g])
        body.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#2a3344" stroke-width="{2 if k == 0 else 1}"/>')
        k += 1
    for src, s, style in shapes:
        col = SOURCE_COLOURS[src] if style != 'before' else '#8a93a6'
        dash = '8 6' if style == 'before' else None
        op = 0.75 if style == 'before' else 1.0
        for name, ln in s.lines.items():
            body.append(polyline([tx(p) for p in ln], col, 5 if name == 'hood' else 2.5, dash, op))
            if name == 'hood':
                body += [f'<circle cx="{tx(p)[0]:.1f}" cy="{tx(p)[1]:.1f}" r="4" fill="{col}"/>' for p in ln]
        for name, lns in s.loops.items():
            if name not in ('fender_l', 'bumper_f', 'door_fl', 'sill_l'):
                continue
            for ln in lns:
                body.append(polyline([tx(p) for p in ln], col, 1.0, dash, 0.3 * op))
        for c, r in s.wheels.values():
            body.append(circle(tx(c), r * sc, col, 2.5, dash, op))
        if s.door:
            d = tx(s.door)
            body.append(f'<rect x="{d[0] - 5:.1f}" y="{d[1] - 5:.1f}" width="10" height="10" fill="none" stroke="{col}" stroke-width="2"/>')
    leg = [(title, '#fff'), (subtitle, '#ccc')]
    srcs = []
    for src, _, style in shapes:
        name = SOURCE_NAMES[src] + (' (before, dashed)' if style == 'before' else '')
        if name not in [x[0] for x in srcs]:
            srcs.append((name, SOURCE_COLOURS[src] if style != 'before' else '#8a93a6'))
    body.append(label(20, 28, title, '#fff', 18))
    body.append(label(20, 52, subtitle, '#ccc', 14))
    x = 20
    for name, col in srcs:
        body.append(label(x, 76, name, col, 15))
        x += 12 + 9 * len(name)
    return svg(W, H, body)


def compare(sid, doc, out):
    tr = [t for t in load_tracings(sid) if t['camera'] == 'side']
    if not tr:
        return [], ''
    moments = doc['moments']
    mode = 'rear_wheel' if all('rear' in t['halves'][h]['wheels'] for t in tr for h in ('real', 'beamng')) else 'door'
    unit_name = 'wheelbases' if mode == 'rear_wheel' else 'front wheel radii'
    tr.sort(key=lambda t: t['video_s'])
    before_t = next((t for t in tr if t['moment'] == 'first contact'), tr[0])
    m_before = next(m for m in moments if m['name'] == 'before')
    raw_before = {'real': shape_from_tracing(before_t, 'real'), 'beamng': shape_from_tracing(before_t, 'beamng'), 'rubble': shape_from_rubble(m_before)}
    norm = {k: normaliser(mode, v) for k, v in raw_before.items()}
    nb = {k: normalise(raw_before[k], *norm[k]) for k in raw_before}
    made, tables = [], []
    origin_txt = ('origin: rear wheel centre; unit: wheelbase (each source\'s own before frame)' if mode == 'rear_wheel'
                  else 'origin: front door\'s front edge at the sill (square); unit: front wheel radius (each source\'s before frame); rear not in the footage')
    for t in tr:
        rm = sheet.pick_moment(sid, t['moment'], moments)
        shapes = {'real': shape_from_tracing(t, 'real'), 'beamng': shape_from_tracing(t, 'beamng'), 'rubble': shape_from_rubble(rm)}
        n = {k: normalise(shapes[k], *norm[k]) for k in shapes}
        tag = re.sub(r'\W+', '_', t['moment'])
        title = f'{sid} {t["moment"]} (video {t["video_s"]} s; Rubble {rm["name"]}, {rm["after_impact_ms"]:+g} ms): side profile, front at the left'
        made.append(write(out / sid / 'common' / f'{tag}.svg', plot_frame([(k, n[k], 'now') for k in ('real', 'beamng', 'rubble')], title, origin_txt + f'; grid {"0.1" if mode == "rear_wheel" else "0.5"} {unit_name}', unit_name)))
        # change: each source against its own before, three panels
        panels = []
        for k in ('real', 'beamng', 'rubble'):
            panels.append(plot_frame([(k, nb[k], 'before'), (k, n[k], 'now')], f'{SOURCE_NAMES[k]}: {t["moment"]} against before', origin_txt.split(';')[0], unit_name, (800, 520)))
        body = ''.join(f'<g transform="translate({800 * i},0)">{p[p.index(">") + 1:p.rindex("</svg>")]}</g>' for i, p in enumerate(panels))
        made.append(write(out / sid / 'change' / f'{tag}.svg', f'<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="520" viewBox="0 0 2400 520">{body}</svg>\n'))
        if t is not before_t:
            nums = {k: numbers(nb[k], n[k], mode) for k in n}
            nums['rubble']['check: Rubble cabin and rear pitch from the fit (deg, + rear up)'] = rm['bodies'][0]['rear_pitch_deg'] - m_before['bodies'][0]['rear_pitch_deg']
            keys = []
            for k in ('real', 'beamng', 'rubble'):
                keys += [x for x in nums[k] if x not in keys]
            rows = [f'### {sid} {t["moment"]} (video {t["video_s"]} s; Rubble {rm["name"]}, {rm["after_impact_ms"]:+g} ms)', '',
                    f'Unit: {unit_name}. {origin_txt}.', '', '| | real test | BeamNG | Rubble |', '|---|---|---|---|']
            for key in keys:
                cells = []
                for k in ('real', 'beamng', 'rubble'):
                    v = nums[k].get(key)
                    cells.append('n/a' if v is None else (f'{v:d}' if isinstance(v, int) else f'{v:+.3f}' if 'deg' not in key else f'{v:+.1f}'))
                rows.append(f'| {key} | ' + ' | '.join(cells) + ' |')
            tables.append('\n'.join(rows) + '\n')
    return made, '\n'.join(tables)


def top_yaw(sid, doc):
    tr = [t for t in load_tracings(sid) if t['camera'] == 'top']
    if not tr:
        return ''
    tr.sort(key=lambda t: t['video_s'])
    ang = lambda t, h: math.degrees(math.atan2(-(to_px(t, t['halves'][h]['lines']['side_line'][-1])[1] - to_px(t, t['halves'][h]['lines']['side_line'][0])[1]),  # noqa: E731
                                               to_px(t, t['halves'][h]['lines']['side_line'][-1])[0] - to_px(t, t['halves'][h]['lines']['side_line'][0])[0]))
    rows = [f'### {sid} top view: yaw', '', 'Yaw from the side line\'s angle in the frame, against the first frame (+: nose toward the struck side). '
            'Rubble: the cabin and rear\'s fitted yaw (+ toward its left, the struck side).', '',
            '| reference moment | real test | BeamNG | Rubble (moment) |', '|---|---|---|---|']
    for t in tr:
        rm = sheet.pick_moment(sid, t['moment'], doc['moments'])
        rows.append(f'| {t["moment"]} (video {t["video_s"]} s) | {ang(t, "real") - ang(tr[0], "real"):+.1f} | '
                    f'{ang(t, "beamng") - ang(tr[0], "beamng"):+.1f} | {rm["bodies"][0]["rear_yaw_deg"]:+.1f} ({rm["name"]}) |')
    rows += ['', 'Rubble\'s yaw at every moment: ' + ', '.join(f'{m["name"]} ({m["after_impact_ms"]:+g} ms) {m["bodies"][0]["rear_yaw_deg"]:+.1f}°' for m in doc['moments'])
             + '. The video times of the reference frames are approximate (index.json), so "rotation" may be later than 300 ms.']
    return '\n'.join(rows) + '\n'


# ---------------------------------------------------------------- rasterise

def rasterise(dirs):
    chrome = os.environ.get('CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    for d in dirs:
        root = Path(d).resolve()
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **k)  # noqa: E731
        http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        profile = root / '.chrome-profile'
        for f in sorted(root.rglob('*.svg')):
            head = f.read_text()[:400]
            w, h = int(re.search(r'width="(\d+)"', head).group(1)), int(re.search(r'height="(\d+)"', head).group(1))
            png = f.with_suffix('.png')
            url = f'http://127.0.0.1:{srv.server_address[1]}/{f.relative_to(root).as_posix()}'
            subprocess.run([chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--user-data-dir={profile}',
                            f'--window-size={w},{h}', f'--screenshot={png}', url], check=True, capture_output=True, timeout=60)
            print(png)
        srv.shutdown()


# ---------------------------------------------------------------- main

def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if a[0] == 'rasterise':
        rasterise([x for x in a[1:] if not x.startswith('--')])
        return 0
    out = Path(opt('--out', ENGINE / 'out' / 'crash-reference' / 'outlines'))
    frames = Path(opt('--frames', ARTIFACTS))
    ids = [x for x in a[1:] if re.fullmatch(r'S\d\d', x)] or ['S01', 'S02']
    md = ['# Shape numbers: real test, BeamNG and Rubble', '',
          'From the normalised hood line and wheel positions, by the same code for all three sources '
          '(`engine/tools/crash-reference/outlines.py`). Reported comparisons, not criteria. Reference values come '
          'from hand tracings (`engine/docs/crash-reference/tracings/`, an approximate first pass); Rubble\'s from '
          'the CPU reference (run 0, edge contact). Rubble\'s hood line is its centreline nodes (4 rows on the '
          'generated designs); the footage\'s is the hood\'s visible side edge.', '']
    for sid in ids:
        doc = json.loads((out / f'{sid}-edge' / 'outlines.json').read_text())
        stills = Path(opt('--stills', ARTIFACTS / 'rubble-v2' / f'{sid}-edge-run0'))
        made = rubble_overlays(sid, doc, out, stills)
        for t in load_tracings(sid):
            made.append(annotated_frame(t, frames, out / sid))
        figs, table = compare(sid, doc, out)
        made += figs
        md += [f'## {sid}', '', table, top_yaw(sid, doc)]
        print(f'{sid}: {len(made)} SVGs')
    write(out / 'shape-numbers.md', '\n'.join(md))
    print(out / 'shape-numbers.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
