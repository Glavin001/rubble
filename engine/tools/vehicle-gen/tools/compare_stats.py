"""Compare two --stats JSON files (Rust CLI vs tools/stats.py) part by part.

    python3 tools/compare_stats.py a.json b.json
"""
import sys
import json
import math

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
pa = {p['name']: p for p in a['parts']}
pb = {p['name']: p for p in b['parts']}
names = list(dict.fromkeys(list(pa) + list(pb)))
worst = 0.0
bad = 0
order_a = [p['name'] for p in a['parts']]
order_b = [p['name'] for p in b['parts']]
print('part order identical:', order_a == order_b)
print('%-24s %8s %8s %10s %10s  %s' % ('part', 'tris A', 'tris B', 'verts A', 'verts B', 'max |bbox| diff mm / sum diff'))
for n in names:
    x, y = pa.get(n), pb.get(n)
    if not x or not y:
        print('%-24s missing in %s' % (n, 'A' if not x else 'B'))
        bad += 1
        continue
    d = max(abs(u - v) for u, v in zip(x['lo'] + x['hi'], y['lo'] + y['hi']))
    ds = max(abs(u - v) for u, v in zip(x['sum'], y['sum'])) / max(x['verts'], 1)
    same = x['ntris'] == y['ntris'] and x['verts'] == y['verts']
    worst = max(worst, d)
    flag = '' if same and d < 1e-6 and ds < 1e-6 else '  <--'
    if flag:
        bad += 1
    print('%-24s %8d %8d %10d %10d  %.4f / %.2e%s' % (n, x['ntris'], y['ntris'], x['verts'], y['verts'], d * 1000, ds, flag))
ma = {m['name']: m['pos'] for m in a['markers']}
mb = {m['name']: m['pos'] for m in b['markers']}
md = max((math.dist(ma[k], mb[k]) for k in ma if k in mb), default=0)
print('markers: %d vs %d, max distance %.6f mm, missing %s' % (len(ma), len(mb), md * 1000, sorted(set(ma) ^ set(mb))))
print('parts differing: %d of %d; worst bbox difference %.6f mm' % (bad, len(names), worst * 1000))
