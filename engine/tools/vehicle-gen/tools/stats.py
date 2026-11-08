"""Per-part stats of a design built by the Python generator (no Blender):
counts, bounding boxes and vertex checksums, the same JSON the Rust CLI
writes with --stats, to compare the two generators.

    python3 tools/stats.py <design> <out.json>      (Blender's python3.11 has numpy)
"""
import sys
import os
import json
import importlib
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'designs'))
from vehgen import assemble, uvpack  # noqa: E402

D = importlib.import_module(sys.argv[1])
parts, cb = assemble.body_parts(D, log=lambda *a: None)
new, extras, markers = D.parts(cb)
for k, m in extras.items():
    parts[k].add(m)
parts.update(new)
if hasattr(D, 'post'):
    D.post(parts)
uvpack.pack(parts)
uvpack.normalise_other(parts)
out = []
for n, m in parts.items():
    lo, hi = (m.v.min(0), m.v.max(0)) if len(m.v) else (np.zeros(3), np.zeros(3))
    out.append(dict(name=n, verts=len(m.v), quads=len(m.q), tris=len(m.t), ntris=int(m.ntris()),
                    lo=lo.tolist(), hi=hi.tolist(), sum=m.v.sum(0).tolist(), sum2=float((m.v ** 2).sum())))
mk = [dict(name=k, pos=list(map(float, v[0]))) for k, v in markers.items()]
json.dump(dict(parts=out, markers=mk), open(sys.argv[2], 'w'), indent=1)
