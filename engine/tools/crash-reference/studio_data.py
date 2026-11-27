"""The Reference Studio's data for the crash reference scripts (`outlines.py`, `sheet.py`).

Shots, frames and tracings are the studio's (`apps/reference-studio`, README "Data model"):
points-only JSON in `engine/docs/crash-reference/studio/`, images in the gitignored
`apps/reference-studio/library/`. Shape maths (normalising, measures, Rubble's elements, the
reference-moment rule) is Rust's: `reference_shape` (engine-softbody-lab example), called
here as the studio's server calls it. Standard library only.
"""
import json
import subprocess
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2]
REPO = ENGINE.parent
DATA = ENGINE / 'docs' / 'crash-reference' / 'studio'
LIBRARY = REPO / 'apps' / 'reference-studio' / 'library'
BIN = ENGINE / 'target' / 'release' / 'examples' / 'reference_shape'


def _load(kind):
    d = DATA / kind
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob('*.json'))} if d.exists() else {}


def shots():
    return _load('shots')


def frames():
    return _load('frames')


def structures():
    return _load('structures')


def scenario_of(frame, shot_map):
    """The Rubble scenario id a frame's shot is linked to, or None."""
    s = shot_map.get(frame.get('shot') or '')
    return (s or {}).get('rubble', {}) and s['rubble']['scenario']


def image_path(frame, library=LIBRARY):
    img = frame.get('image')
    return Path(library) / img['file'] if img else None


def traced(region):
    return any(v.get('p') or v.get('pts') or v.get('c') for i in region['tracing']['instances'] for v in i['elements'].values())


def rust(op, **body):
    """One request to the Rust shape service (built on first use)."""
    if not BIN.exists():
        subprocess.run(['cargo', 'build', '--release', '-p', 'engine-softbody-lab', '--example', 'reference_shape'],
                       cwd=ENGINE, check=True)
    r = subprocess.run([str(BIN)], input=json.dumps({'op': op, **body}), capture_output=True, text=True, check=True)
    out = json.loads(r.stdout)
    if not out['ok']:
        raise RuntimeError(f'reference_shape {op}: {out["error"]}')
    return out['result']


def link(frame, scenario):
    """(moment, camera) of the scenario a frame is compared with: its own choice, else the rule."""
    own = frame.get('rubble') or {}
    pick = rust('pick', scenario=scenario, moment=frame['moment']['label'] or 'peak', view=frame.get('view') or 'side')
    return own.get('moment') or pick['moment'], own.get('camera') or pick['camera']


def region_size(frame, region):
    c = region['crop']
    return (c[2] - c[0]) * frame['image']['width'], (c[3] - c[1]) * frame['image']['height']


def before_of(frame, all_frames, shot_map):
    """The studio server's rule: the frame's own before, the shot's, else the earliest traced
    frame of the same shot and view (video time, time after impact, a before-like label, id)."""
    if frame.get('before_frame'):
        return all_frames.get(frame['before_frame'])
    shot = shot_map.get(frame.get('shot') or '') or {}
    b = all_frames.get(shot.get('before_frame') or '')
    if b and b['view'] == frame['view']:
        return b
    same = [f for f in all_frames.values() if f.get('shot') and f['shot'] == frame.get('shot') and f['view'] == frame['view']
            and any(traced(r) for r in f['regions'])]
    inf = float('inf')
    early = lambda f: 0 if f['moment']['label'].strip().lower() in ('before', 'setup', 'first contact', 'contact', 'rest before') else 1  # noqa: E731
    same.sort(key=lambda f: ((f.get('time') or {}).get('video_s', inf), f['moment'].get('after_impact_ms') if f['moment'].get('after_impact_ms') is not None else inf, early(f), f['id']))
    return same[0] if same else None
