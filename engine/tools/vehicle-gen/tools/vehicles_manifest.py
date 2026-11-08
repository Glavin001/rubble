"""Write engine/scenes/vehicles/index.json: one entry per built vehicle
(title, glb, thumbnail, base design, measured size and triangle count), the
list the model viewer (viewer/index.html) shows. build.py runs it after each
build; run it by hand after deleting or renaming a vehicle.

    python3 tools/vehicles_manifest.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEH = os.path.normpath(os.path.join(HERE, '..', '..', 'scenes', 'vehicles'))
DESIGNS = os.path.join(HERE, 'designs')


def entry(name):
    d = os.path.join(VEH, name)
    e = {'name': name, 'title': name, 'glb': f'{name}/{name}.glb'}
    thumb = f'renders/{name}_front34.jpg'
    if os.path.exists(os.path.join(d, thumb)):
        e['thumb'] = f'{name}/{thumb}'
    rd = os.path.join(d, 'README.md')
    if os.path.exists(rd):
        for line in open(rd):
            if line.startswith('# '):
                e['title'] = line[2:].strip()
                break
    src = os.path.join(DESIGNS, name + '.py')
    if os.path.exists(src):
        e['design'] = f'engine/tools/vehicle-gen/designs/{name}.py'
        m = re.search(r"derive\('([a-z0-9_]+)'", open(src).read())
        if m:
            e['base'] = m.group(1)
    ck = os.path.join(d, 'checks.md')
    if os.path.exists(ck):
        t = open(ck).read()
        m = re.search(r'length (\d+) mm, width (\d+) mm, height (\d+) mm, wheelbase (\d+) mm', t)
        if m:
            e['size_mm'] = dict(zip(('length', 'width', 'height', 'wheelbase'), map(int, m.groups())))
        m = re.search(r'total (\d+) \(', t)
        if m:
            e['triangles'] = int(m.group(1))
        m = re.search(r'(\d+) parts; missing', t)
        if m:
            e['parts'] = int(m.group(1))
        e['checks_failed'] = t.count('| FAIL |')
    return e


def write():
    names = sorted(n for n in os.listdir(VEH)
                   if os.path.exists(os.path.join(VEH, n, n + '.glb')))
    out = {'vehicles': [entry(n) for n in names]}
    path = os.path.join(VEH, 'index.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    return path, len(names)


if __name__ == '__main__':
    print('wrote %s (%d vehicles)' % write())
