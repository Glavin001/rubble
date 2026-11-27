#!/usr/bin/env python3
"""Surface triangles of every vehicle part of a parsed LS-DYNA deck, for viewing.

    python3 -I engine/tools/fe-import/fe_mesh.py CACHE_DIR SECTIONS_JSON OUT_DIR

Shells give two triangles per quad (one per triangle); solids give the faces used by
exactly one element of their part (the part's boundary). Vertices are compacted per
part and written in glTF axes and metres: (X, Y, Z) = (y_left, z_up, x_forward) / 1000.
Writes OUT_DIR/parts.bin (per part: float32 xyz, then uint32 triangle indices) and
OUT_DIR/parts.json (pid, name, section, counts, byte offsets). Standard library only.
"""
import json
import os
import pickle
import sys
from array import array

FACES = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))


def main():
    cache, secpath, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    d = pickle.load(open(os.path.join(cache, "deck.pickle"), "rb"))
    sections = json.load(open(secpath))
    parts = d["parts"]; nid = d["node_id"]; X = d["xyz"]
    tris = {}   # pid -> list of node-id triples
    sp, sn = d["shell_pid"], d["shell_n"]
    for e in range(len(sp)):
        n = sn[4 * e:4 * e + 4]
        t = tris.setdefault(sp[e], [])
        t.append((n[0], n[1], n[2]))
        if n[2] != n[3]:
            t.append((n[0], n[2], n[3]))
    so, s8 = d["solid_pid"], d["solid_n"]
    faces = {}
    for e in range(len(so)):
        n = s8[8 * e:8 * e + 8]
        fp = faces.setdefault(so[e], {})
        for f in FACES:
            q = [n[k] for k in f]
            u = list(dict.fromkeys(q))
            if len(u) < 3:
                continue
            key = tuple(sorted(u))
            if key in fp:
                fp[key] = None
            else:
                fp[key] = u
    for pid, fp in faces.items():
        t = tris.setdefault(pid, [])
        for u in fp.values():
            if u is None:
                continue
            t.append((u[0], u[1], u[2]))
            if len(u) == 4:
                t.append((u[0], u[2], u[3]))
    meta = []
    with open(os.path.join(out, "parts.bin"), "wb") as f:
        for pid in sorted(tris):
            if parts[pid]["source"] != "vehicle":
                continue
            local = {}
            v = array("f"); idx = array("I")
            for tri in tris[pid]:
                for n in tri:
                    j = local.get(n)
                    if j is None:
                        j = local[n] = len(local)
                        i = 3 * nid[n]
                        v.extend((X[i + 1] / 1000, X[i + 2] / 1000, X[i] / 1000))
                    idx.append(j)
            off = f.tell()
            v.tofile(f); idx.tofile(f)
            meta.append(dict(pid=pid, name=parts[pid]["title"], section=sections.get(str(pid), "other"),
                             n_vert=len(local), n_tri=len(idx) // 3, offset=off))
    json.dump(meta, open(os.path.join(out, "parts.json"), "w"))
    print("parts", len(meta), "triangles", sum(m["n_tri"] for m in meta), "vertices", sum(m["n_vert"] for m in meta))


if __name__ == "__main__":
    main()
