#!/usr/bin/env python3
"""Part outlines that make shape differences between Rubble and the reference footage
obvious (engine/docs/crash-reference.md, "Outlines").

Inputs:
- Rubble: `outlines.json` from the CPU tool
  (`cargo run --release -p engine-softbody-lab --example crash_outlines -- S02`, written to
  `engine/out/crash-reference/outlines/<ID>-edge/` with its own stills of the collision
  surface from each scenario camera at each moment);
- the recorder's stills (`crash-ref-record.mjs --all-cams`, `<moment>-<camera>.png`);
- the Reference Studio's shots, frames and tracings (`apps/reference-studio/workspace/`,
  points only; trace and correct them in `apps/reference-studio`) and its image library
  (third-party footage, never committed: `apps/reference-studio/workspace/library/`).

Normalising and measuring are Rust's (`reference_shape`, the studio's maths: one place for
the page and this script); this script only draws.

    python3 engine/tools/crash-reference/outlines.py all [S01 S02]
        [--out engine/out/crash-reference/outlines] [--stills DIR] [--library DIR]
    python3 engine/scripts/with-gpu-lock.py --limit 300 -- \\
        python3 engine/tools/crash-reference/outlines.py rasterise DIR [DIR ...]

`all` writes SVGs (overlays on the recorded and the CPU stills, onion skins, annotated
reference frames, common-frame and change figures) and `shape-numbers.md`; `rasterise`
turns every SVG in the folders into a PNG with headless Chrome over http (the only SVG
renderer this machine has; Chrome is run without the GPU, but under the lock as every
headless Chrome is). Standard library only.

The common frame is the structure schema's (`studio/structures/<id>.json` `frames`, the
first every source has): for the side view, origin the rear wheel's centre and unit the
wheelbase in each source's own "before" frame when the rear wheel is traced (S01), else
origin the front door's corner at the sill and unit the front wheel's radius (S02's frames
show only the front half). Shape numbers are reported comparisons, not criteria.
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
import studio_data  # noqa: E402  (the studio's data and its Rust maths)

ARTIFACTS = Path.home() / 'Development' / 'rubble-artifacts' / 'crash-reference'

COLOURS = {
    'hood': '#ff9a1f', 'fender_l': '#38b6ff', 'fender_r': '#38b6ff', 'bumper_f': '#e04be0', 'bumper_r': '#e04be0',
    'door_fl': '#58e070', 'door_fr': '#58e070', 'door_rl': '#2fa04a', 'door_rr': '#2fa04a', 'trunk': '#e8e040',
    'roof': '#b8b8ff', 'greenhouse': '#8f8fd8', 'sill_l': '#ff5050', 'sill_r': '#ff5050', 'wheel': '#f0f0f0',
}
SIDE_GROUPS = ['bumper_f', 'hood', 'fender_l', 'door_fl', 'door_rl', 'greenhouse', 'roof', 'sill_l', 'trunk']
TOP_GROUPS = ['bumper_f', 'hood', 'fender_l', 'fender_r', 'door_fl', 'door_fr', 'greenhouse', 'roof', 'trunk']
SOURCE_COLOURS = {'real': '#ff4d4d', 'beamng': '#3d8bff', 'rubble': '#2ee07a', 'other': '#b48cff'}
SOURCE_NAMES = {'real': 'real test', 'beamng': 'BeamNG', 'rubble': 'Rubble', 'other': 'other'}


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


# ---------------------------------------------------------------- tracings (the studio's frames)

def scenario_frames(sid, view=None):
    """The studio's frames of shots linked to scenario `sid` with a traced region, in time order."""
    shot_map, all_frames = studio_data.shots(), studio_data.frames()
    out = [f for f in all_frames.values() if studio_data.scenario_of(f, shot_map) == sid and f.get('image')
           and any(studio_data.traced(r) for r in f['regions']) and (view is None or f['view'] == view)]
    out.sort(key=lambda f: ((f.get('time') or {}).get('video_s', 1e9), f['id']))
    return out, all_frames, shot_map


def annotated_frame(f, schemas, library, out):
    """The frame with each region's crop and traced elements (display only)."""
    W, H = f['image']['width'], f['image']['height']
    body = []
    for r in f['regions']:
        col = SOURCE_COLOURS.get(r['source'], '#b48cff')
        c = r['crop']
        to = lambda p: [(c[0] + p[0] * (c[2] - c[0])) * W, (c[1] + p[1] * (c[3] - c[1])) * H]  # noqa: E731
        body.append(polyline([to(p) for p in ([0, 0], [1, 0], [1, 1], [0, 1])], col, 3, '12 8', 0.9, closed=True))
        for inst in r['tracing']['instances']:
            labels = {e['name']: e['label'] for e in schemas.get(inst['schema'], {}).get('elements', [])}
            for name, v in inst['elements'].items():
                dash = None if v.get('vis', 'clear') == 'clear' else '10 6'
                if v.get('pts'):
                    ln = [to(p) for p in v['pts']]
                    body.append(polyline(ln, col if name != 'hood' else '#ffb000', 5 if name == 'hood' else 3.5, dash))
                    body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="#000"/>' for x, y in ln]
                    body.append(label(ln[0][0] + 6, ln[0][1] - 8, labels.get(name, name), col, 22))
                elif v.get('c') and v.get('rim'):
                    cc, rim = to(v['c']), to(v['rim'])
                    body.append(circle(cc, math.dist(cc, rim), col, 3.5, dash))
                    body.append(label(cc[0] - 40, cc[1], labels.get(name, name), col, 22))
                elif v.get('p'):
                    p = to(v['p'])
                    body.append(f'<rect x="{p[0] - 7:.1f}" y="{p[1] - 7:.1f}" width="14" height="14" fill="none" stroke="{col}" stroke-width="3"/>')
    traced_by = '; '.join(sorted({r['tracing']['traced_by'] for r in f['regions'] if r['tracing']['traced_by']}))
    head = [(f'{f["id"]}: {f["moment"]["label"]} ({f["view"]}); tracing: {traced_by}', '#fff'),
            ('dashed boxes: each region\'s crop (red: real test, blue: BeamNG); dashed lines: blurred or hidden', '#ddd'),
            ('correct it in the Reference Studio (apps/reference-studio)', '#ffb000')]
    leg = [f'<g transform="scale(1.6)">{"".join(legend(head))}</g>']
    src = studio_data.image_path(f, library)
    bg = data_uri(src) if src and src.exists() else None
    return write(out / 'reference' / (f['id'] + '-traced.svg'), svg(W, H, body + leg, bg, 1.0))


# ---------------------------------------------------------------- the common frame (Rust)

def comparison(f, before, schema, outlines_path, sid):
    """Every traced region of `f` against the same region of `before`, and Rubble, normalised
    and measured by Rust (`reference_shape compare`, the studio's maths)."""
    sources, names = [], {}
    for r in f['regions']:
        inst = (r['tracing']['instances'] or [None])[0]
        if not inst or inst['schema'] != schema['id'] or not studio_data.traced(r):
            continue
        br = next((x for x in before['regions'] if x['id'] == r['id'] and x['tracing']['instances']), None) or \
            next((x for x in before['regions'] if x['source'] == r['source'] and x['tracing']['instances']), None)
        if not br:
            continue
        shape = lambda fr, rg: {'tracing': {'instance': rg['tracing']['instances'][0], 'width': studio_data.region_size(fr, rg)[0],  # noqa: E731
                                            'height': studio_data.region_size(fr, rg)[1]}}
        sources.append({'name': r['source'], 'before': shape(before, br), 'now': shape(f, r)})
        names[r['source']] = r['label']
    m_now, cam = studio_data.link(f, sid)
    m_before, cam_b = studio_data.link(before, sid)
    view = lambda c: 'car_frame' if schema.get('rubble_view') == 'car_frame' else f'camera:{c}'  # noqa: E731
    sources.append({'name': 'rubble', 'before': {'rubble': {'outlines': str(outlines_path), 'moment': m_before, 'view': view(cam_b)}},
                    'now': {'rubble': {'outlines': str(outlines_path), 'moment': m_now, 'view': view(cam)}}})
    res = studio_data.rust('compare', schema=schema, sources=sources)
    res['rubble_moment'] = m_now
    res['order'] = [s['name'] for s in sources]
    return res


def plot_frame(shapes, title, subtitle, size=(1200, 620)):
    """Shapes (Rust's normalised JSON): [(source, shape, style)]; style 'now' or 'before'.
    Front at the left, as filmed."""
    W, H = size
    xs, ys = [], []
    for _, s, _ in shapes:
        for ln in list(s['lines'].values()) + [p for v in s.get('loops', {}).values() for p in v]:
            xs += [p[0] for p in ln]
            ys += [p[1] for p in ln]
        for c in s['circles'].values():
            xs += [c['c'][0] - c['r'], c['c'][0] + c['r']]
            ys += [c['c'][1] - c['r'], c['c'][1] + c['r']]
    if not xs:
        return svg(W, H, [label(20, 40, title + ': nothing to draw', '#fff')])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.06 * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    top = 90
    sc = min((W - 40) / (x1 - x0), (H - top - 30) / (y1 - y0))
    tx = lambda p: [20 + (x1 - p[0]) * sc, top + (y1 - p[1]) * sc]  # noqa: E731
    body = []
    g = 0.1 if (x1 - x0) < 3 else 0.5
    for axis in (0, 1):
        lo, hi = (x0, x1) if axis == 0 else (y0, y1)
        k = math.floor(lo / g)
        while k * g <= hi:
            a, b = ((tx([k * g, y0]), tx([k * g, y1])) if axis == 0 else (tx([x0, k * g]), tx([x1, k * g])))
            body.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#2a3344" stroke-width="{2 if k == 0 else 1}"/>')
            k += 1
    for src, s, style in shapes:
        col = SOURCE_COLOURS.get(src, '#b48cff') if style != 'before' else '#8a93a6'
        dash = '8 6' if style == 'before' else None
        op = 0.75 if style == 'before' else 1.0
        for name, ln in s['lines'].items():
            body.append(polyline([tx(p) for p in ln], col, 5 if name == 'hood' else 2.5, dash, op))
            if name == 'hood':
                body += [f'<circle cx="{tx(p)[0]:.1f}" cy="{tx(p)[1]:.1f}" r="4" fill="{col}"/>' for p in ln]
        for name, lns in s.get('loops', {}).items():
            if name in ('fender_l', 'bumper_f', 'door_fl', 'sill_l'):
                for ln in lns:
                    body.append(polyline([tx(p) for p in ln], col, 1.0, dash, 0.3 * op))
        for c in s['circles'].values():
            body.append(circle(tx(c['c']), c['r'] * sc, col, 2.5, dash, op))
        for p in s['points'].values():
            d = tx(p)
            body.append(f'<rect x="{d[0] - 5:.1f}" y="{d[1] - 5:.1f}" width="10" height="10" fill="none" stroke="{col}" stroke-width="2"/>')
    body.append(label(20, 28, title, '#fff', 18))
    body.append(label(20, 52, subtitle, '#ccc', 14))
    x = 20
    seen = []
    for src, _, style in shapes:
        name = SOURCE_NAMES.get(src, src) + (' (before, dashed)' if style == 'before' else '')
        if name not in seen:
            seen.append(name)
            body.append(label(x, 76, name, SOURCE_COLOURS.get(src, '#b48cff') if style != 'before' else '#8a93a6', 15))
            x += 12 + 9 * len(name)
    return svg(W, H, body)


def fmt(cell, key):
    if cell is None:
        return 'n/a'
    if cell.get('int'):
        return f'{int(cell["v"])}'
    v = f'{cell["v"]:+.1f}' if 'deg' in key else f'{cell["v"]:+.3f}'
    return v + ('*' if cell.get('vis', 'clear') != 'clear' else '')


def compare(sid, out):
    """Common-frame and change figures and the shape-number tables of a scenario's side frames."""
    frs, all_frames, shot_map = scenario_frames(sid)
    schemas = studio_data.structures()
    outlines_path = out / f'{sid}-edge' / 'outlines.json'
    made, tables = [], []
    for f in frs:
        inst = next((r['tracing']['instances'][0] for r in f['regions'] if r['tracing']['instances']), None)
        schema = schemas.get(inst['schema']) if inst else None
        before = studio_data.before_of(f, all_frames, shot_map) or f
        if not schema:
            continue
        res = comparison(f, before, schema, outlines_path, sid)
        srcs = res['order']
        frame_txt = res['frame']['label'] if res['frame'] else 'no common frame (raw axes)'
        tag = re.sub(r'\W+', '_', f'{f["view"]}_{f["moment"]["label"]}_{f["id"]}')
        title = f'{sid} {f["moment"]["label"]} ({f["id"]}; Rubble {res["rubble_moment"]}), {schema["id"]}, front at the left'
        made.append(write(out / sid / 'common' / f'{tag}.svg', plot_frame([(k, res['shapes'][k]['now'], 'now') for k in srcs], title, frame_txt)))
        panels = [plot_frame([(k, res['shapes'][k]['before'], 'before'), (k, res['shapes'][k]['now'], 'now')],
                             f'{SOURCE_NAMES.get(k, k)}: {f["moment"]["label"]} against {before["id"]}', frame_txt.split(';')[0], (800, 520)) for k in srcs]
        body = ''.join(f'<g transform="translate({800 * i},0)">{p[p.index(">") + 1:p.rindex("</svg>")]}</g>' for i, p in enumerate(panels))
        made.append(write(out / sid / 'change' / f'{tag}.svg', f'<svg xmlns="http://www.w3.org/2000/svg" width="{800 * len(panels)}" height="520" viewBox="0 0 {800 * len(panels)} 520">{body}</svg>\n'))
        if before is f:
            continue
        rows = [f'### {sid} {f["moment"]["label"]} ({f["id"]}, against {before["id"]}; Rubble {res["rubble_moment"]})', '',
                f'{schema["id"]}. {frame_txt}. `*`: measured from a blurred or hidden element.', '',
                '| | ' + ' | '.join(SOURCE_NAMES.get(k, k) for k in srcs) + ' |', '|---|' + '---|' * len(srcs)]
        for r in res['rows']:
            rows.append(f'| {r["label"]} | ' + ' | '.join(fmt(r['values'].get(k), r['label']) for k in srcs) + ' |')
        tables.append('\n'.join(rows) + '\n')
    return made, '\n'.join(tables)


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
    library = Path(opt('--library', studio_data.LIBRARY))
    ids = [x for x in a[1:] if re.fullmatch(r'S\d\d', x)] or ['S01', 'S02']
    md = ['# Shape numbers: real test, BeamNG and Rubble', '',
          'From the normalised traced elements, by the same Rust code for every source (`reference_shape`, '
          '`engine_softbody::crash_reference::shape`; the Reference Studio shows the same table). Reported comparisons, '
          'not criteria. Reference values come from hand tracings (`apps/reference-studio/workspace/frames/`, '
          'partly an approximate first pass); Rubble\'s from the CPU reference (run 0, edge contact). Rubble\'s hood '
          'line is its centreline nodes (4 rows on the generated designs); the footage\'s is the hood\'s visible side edge.', '']
    for sid in ids:
        doc = json.loads((out / f'{sid}-edge' / 'outlines.json').read_text())
        stills = Path(opt('--stills', ARTIFACTS / 'rubble-v2' / f'{sid}-edge-run0'))
        made = rubble_overlays(sid, doc, out, stills)
        schemas = studio_data.structures()
        for f in scenario_frames(sid)[0]:
            made.append(annotated_frame(f, schemas, library, out / sid))
        figs, table = compare(sid, out)
        made += figs
        md += [f'## {sid}', '', table]
        print(f'{sid}: {len(made)} SVGs')
    write(out / 'shape-numbers.md', '\n'.join(md))
    print(out / 'shape-numbers.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
