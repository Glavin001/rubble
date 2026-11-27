#!/usr/bin/env python3
"""Side-by-side sheets of the crash reference suite (engine/docs/crash-reference.md).

For each scenario and each contact setting it has stills for, one PNG: a row per reference
frame of the Reference Studio (`engine/docs/crash-reference/studio/`: the frame itself,
already real | BeamNG halves) beside Rubble's still from the frame's linked camera and
moment (its own choice, else the rule in `engine_softbody::crash_reference::shape::link`), each row labelled with the scenario id,
the reference moment and Rubble's moment (ms after first contact), the camera and the
contact setting.

    python3 engine/tools/crash-reference/sheet.py [S01 S02 ...]
        [--library apps/reference-studio/library]
        [--stills engine/out/crash-reference] [--out engine/out/crash-reference/sheets]
        [--copy-to ~/Development/rubble-artifacts/crash-reference/sheets] [--run 0]

Reference frames are third-party footage: they are read from the studio's gitignored
library (never committed), and the sheets are not committed. Where a scenario has no frames on
disk (S10) the script says "no reference frames" and goes on. Rubble's stills are the
recorder's (`crash-ref-record.mjs`, `<moment>-<camera>.png` in `<ID>-<contact>-run<k>/`).
The moments and cameras are the scenario's registry entry the recorder saved beside its
stills (`scenario.json`). Needs ffmpeg (drawtext) only.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import studio_data  # noqa: E402  (the studio's frames; the moment rule is Rust's)
ROW_H = 540
LABEL_H = 56
FONTS = [
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/System/Library/Fonts/Helvetica.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

def norm(s):
    return re.sub(r'[\s_-]+', ' ', s.strip().lower())


def size(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(',')[:2]
    return int(w), int(h)


def even(x):
    return int(x) // 2 * 2


def ffmpeg(args):
    r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'ffmpeg failed: {r.stderr[-2000:]}')


def text_filter(font, textfile, x, y, size, color='white'):
    f = f"fontfile='{font}':" if font else ''
    return f"drawtext={f}textfile='{textfile}':x={x}:y={y}:fontsize={size}:fontcolor={color}"


def sheet(sid, scen, frames, stills_dir, contact, out, tmp, font):
    rows = []
    for fr in frames:
        mom_name, cam_name = studio_data.link(fr, sid)
        cam = next((c['name'] for c in scen['cameras'] if norm(c['name']) == norm(cam_name)), scen['cameras'][0]['name'])
        mom = next((m for m in scen['moments'] if m['name'] == mom_name), scen['moments'][0])
        still = stills_dir / f"{mom['name']}-{cam.replace(' ', '_')}.png"
        rows.append((fr, cam, mom, still))
    widths = []
    for fr, _, _, still in rows:
        rw, rh = size(fr['path'])
        sw, sh = size(still) if still.exists() else (1280, 720)
        widths.append((even(rw * ROW_H / rh), even(sw * ROW_H / sh)))
    W = max(a + b for a, b in widths) + 8
    parts = []
    for k, ((fr, cam, mom, still), (wr, ws)) in enumerate(zip(rows, widths)):
        label = tmp / f'{sid}-{contact}-{k}.txt'
        label.write_text(f"{sid}  |  reference: {fr['moment']['label']} ({fr['view']}, {fr['id']}; real | BeamNG)  "
                         f"|  Rubble: {mom['name']} ({mom['after_impact_ms']:+g} ms after contact), camera {cam}, contact {contact}")
        part = tmp / f'{sid}-{contact}-{k}.png'
        src = ['-i', str(fr['path'])]
        if still.exists():
            src += ['-i', str(still)]
            right = f'[1:v]scale={ws}:{ROW_H},setsar=1[s];'
        else:
            src += ['-f', 'lavfi', '-i', f'color=c=0x202020:s={ws}x{ROW_H}:d=1']
            miss = tmp / f'{sid}-{contact}-{k}-miss.txt'
            miss.write_text(f'no Rubble still: {still.name}')
            right = f'[1:v]scale={ws}:{ROW_H},setsar=1,{text_filter(font, miss, 20, 20, 24, "0xff8080")}[s];'
        fc = (f'[0:v]scale={wr}:{ROW_H},setsar=1,format=rgb24[r];{right}'
              f'[r][s]hstack=inputs=2,pad={W}:{ROW_H + LABEL_H}:0:{LABEL_H}:color=0x101418,'
              f'{text_filter(font, label, 12, 18, 22)}[o]')
        ffmpeg([*src, '-filter_complex', fc, '-map', '[o]', '-frames:v', '1', str(part)])
        parts.append(part)
    inputs = [x for p in parts for x in ('-i', str(p))]
    fc = ''.join(f'[{i}:v]' for i in range(len(parts))) + f'vstack=inputs={len(parts)}[o]' if len(parts) > 1 else '[0:v]null[o]'
    ffmpeg([*inputs, '-filter_complex', fc, '-map', '[o]', '-frames:v', '1', str(out)])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ids', nargs='*')
    ap.add_argument('--library', default=str(studio_data.LIBRARY))
    ap.add_argument('--stills', default=str(ENGINE / 'out/crash-reference'))
    ap.add_argument('--out', default=str(ENGINE / 'out/crash-reference/sheets'))
    ap.add_argument('--copy-to', default=None)
    ap.add_argument('--run', type=int, default=0)
    a = ap.parse_args()
    if not shutil.which('ffmpeg'):
        sys.exit('needs ffmpeg')
    font = next((f for f in FONTS if os.path.exists(f)), None)
    shot_map, all_frames = studio_data.shots(), studio_data.frames()
    linked = [(studio_data.scenario_of(f, shot_map), f) for f in all_frames.values()]
    ids = a.ids or sorted({sid for sid, _ in linked if sid})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / '.parts'
    tmp.mkdir(exist_ok=True)
    made = []
    for sid in ids:
        frames = []
        for fsid, fr in sorted(linked, key=lambda x: ((x[1].get('time') or {}).get('video_s', 1e9), x[1]['id'])):
            if fsid != sid or not fr.get('image'):
                continue
            p = studio_data.image_path(fr, a.library)
            if p.exists():
                frames.append({**fr, 'path': p})
        if not frames:
            print(f'{sid}: no reference frames')
            continue
        stills = sorted(Path(a.stills).glob(f'{sid}-*-run{a.run}'))
        if not stills:
            print(f'{sid}: no Rubble stills in {a.stills}/{sid}-<contact>-run{a.run}')
            continue
        for d in stills:
            if not (d / 'scenario.json').exists():
                print(f'{d}: no scenario.json (record again with the current crash-ref-record.mjs)')
                continue
            scen = json.loads((d / 'scenario.json').read_text())
            contact = d.name[len(sid) + 1:-len(f'-run{a.run}')]
            dst = out / f'{sid}-{contact}.png'
            rows = sheet(sid, scen, frames, d, contact, dst, tmp, font)
            print(f'{sid} ({contact}): {dst}')
            for fr, cam, mom, still in rows:
                print(f"   {fr['id']}: {fr['moment']['label']} / {fr['view']} -> {still.name}{'' if still.exists() else ' (missing)'}")
            made.append(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    if a.copy_to and made:
        dst = Path(os.path.expanduser(a.copy_to))
        dst.mkdir(parents=True, exist_ok=True)
        for m in made:
            shutil.copy2(m, dst / m.name)
        print(f'copied {len(made)} sheets to {dst}')


if __name__ == '__main__':
    main()
