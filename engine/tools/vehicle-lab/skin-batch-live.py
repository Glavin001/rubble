#!/usr/bin/env python3
"""Summarise skin-batch-live.sh's runs: per scene, before and after, the seconds while
the derby's cars are apart (no damage yet) and in contact (damage > 0): medians of World steps a second, pictures a second and the drawn frames'
GPU ms (skinning / shadow map / picture), each round, then all rounds together.

    python3 engine/tools/vehicle-lab/skin-batch-live.py OUTDIR
"""
import glob
import os
import re
import statistics
import sys

out = sys.argv[1]
rows = {}
for path in sorted(glob.glob(os.path.join(out, '*.txt'))):
    name = os.path.basename(path)[:-4]
    for line in open(path, errors='replace'):
        p = [x.strip() for x in line.split('|')]
        if len(p) < 11 or p[0] not in ('before', 'after'):
            continue
        dmg = float(p[10].rstrip('%'))
        draw = [float(x) for x in p[9].split('/')] if p[9] != '-' else None
        phase = 'contact' if dmg > 0 else 'apart'
        if not name.startswith('derby'):
            phase = 'all'
        key = (name, phase, p[0])
        rows.setdefault(key, {}).setdefault(p[1], []).append((float(p[4]), float(p[6]), draw))


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float('nan')


def summary(rs):
    d = [r[2] for r in rs if r[2]]
    return (len(rs), med([r[0] for r in rs]), med([r[1] for r in rs]),
            med([x[0] for x in d]), med([x[1] for x in d]), med([x[2] for x in d]))


print('scene | phase | side | round | s | steps/s (% real time) | pictures/s | GPU ms a picture: skin / shadow / picture / sum')
for key in sorted(rows):
    allr = []
    for rnd, rs in sorted(rows[key].items()):
        allr += rs
        n, st, pic, sk, sh, pi = summary(rs)
        print(f'{key[0]} | {key[1]} | {key[2]} | {rnd} | {n} | {st:.0f} ({100 * st / 60:.0f} %) | {pic:.0f} | {sk:.2f} / {sh:.2f} / {pi:.2f} / {sk + sh + pi:.2f}')
    n, st, pic, sk, sh, pi = summary(allr)
    print(f'{key[0]} | {key[1]} | {key[2]} | all | {n} | {st:.0f} ({100 * st / 60:.0f} %) | {pic:.0f} | {sk:.2f} / {sh:.2f} / {pi:.2f} / {sk + sh + pi:.2f}')
