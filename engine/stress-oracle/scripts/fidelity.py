#!/usr/bin/env python3
"""Gameplay fidelity of a method switch: every scene run under two method sets (A, the
reference, and B) with one build, and B judged on the outcomes a player sees.

    scripts/fidelity.py --bin target/release/stress-ref --b true_step --out DIR [--a ""]
                        [--jobs N] [--only SUBSTRING] [--set dotted.path=JSON]...
                        [--b-set dotted.path=JSON]...  (B only: another solve mode, say)

Per scene:
  fails     the outcome flags (any bond broken, collapse, ...) are equal
  where     the detached chunks agree (Jaccard of the detached sets >= 0.8)
  when      first cracks within 2 frames (1/30 s) of each other, or neither cracks
  pieces    fragment counts in the same power-of-two bin
  debris    mean speed of detached chunks within 30 % (both above 0.1 m/s)
  cost      substeps and CPU seconds, B over A

The thresholds are the protocol's (IMPROVEMENTS.md, gameplay fidelity class); a scene
passes when every applicable metric does. A and B run at one thread each, in parallel
across scenes.
"""
import argparse, json, math, os, resource, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scenes(only):
    out = []
    for d in ['scenes', 'scenes/showcases']:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith('.json') and (not only or only in f):
                out.append(os.path.join(d, f))
    return out


def run(binary, methods, scene, out, sets):
    env = dict(os.environ, RAYON_NUM_THREADS='1', STRESS_METHODS=methods)
    args = [binary, 'run', scene, '--out', out]
    for s in sets:
        args += ['--set', s]
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, ru = os.wait4(p.pid, 0)
    return ru.ru_utime + ru.ru_stime, status


def detached(o):
    return {(b, i) for b, chunks in o['bodies'].items() for i, c in enumerate(chunks) if c.get('detached')}


def debris_speed(o):
    v = [math.sqrt(sum(x * x for x in c['velocity'])) for chunks in o['bodies'].values() for c in chunks if c.get('detached')]
    return sum(v) / len(v) if v else 0.0


def judge(a, b):
    va, vb = a['values'], b['values']
    m = {}
    m['fails'] = a.get('flags') == b.get('flags')
    da, db = detached(a), detached(b)
    union = da | db
    m['where'] = (len(da & db) / len(union) >= 0.8) if union else True
    ta, tb = va.get('first_crack_time'), vb.get('first_crack_time')
    m['when'] = (ta is None) == (tb is None) and (ta is None or abs(ta - tb) <= 1.0 / 30.0)
    fa, fb = va.get('fragments', 1.0), vb.get('fragments', 1.0)
    m['pieces'] = int(math.log2(max(fa, 1.0))) == int(math.log2(max(fb, 1.0)))
    sa, sb = debris_speed(a), debris_speed(b)
    m['debris'] = (sa <= 0.1 and sb <= 0.1) or (min(sa, sb) > 0.1 and abs(sb - sa) <= 0.3 * sa)
    return m, (fa, fb, sa, sb, len(da), len(db))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--a', default='', help='STRESS_METHODS of the reference side')
    ap.add_argument('--b', default='', help='STRESS_METHODS of the judged side')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--only')
    ap.add_argument('--set', action='append', default=[], help='--set override for both sides')
    ap.add_argument('--b-set', action='append', default=[], help='--set override for the judged side only (e.g. sim.solve_mode)')
    args = ap.parse_args()
    binary = os.path.abspath(args.bin)
    os.makedirs(os.path.join(args.out, 'a'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'b'), exist_ok=True)
    todo = scenes(args.only)

    def both(scene):
        name = os.path.basename(scene)[:-5]
        res = {}
        for side, methods, sets in [('a', args.a, args.set), ('b', args.b, args.set + args.b_set)]:
            out = os.path.join(args.out, side, name + '.json')
            res[side] = run(binary, methods, scene, out, sets)
        return name, res

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(both, todo))
    keys = ['fails', 'where', 'when', 'pieces', 'debris']
    passed = 0
    print('%-30s %s  %9s %9s  %s' % ('scene', ' '.join('%-6s' % k for k in keys), 'substeps', 'cpu', 'fragments, debris m/s, detached (A -> B)'))
    for name, res in results:
        try:
            a = json.load(open(os.path.join(args.out, 'a', name + '.json')))
            b = json.load(open(os.path.join(args.out, 'b', name + '.json')))
        except (OSError, ValueError):
            print('%-30s run failed (A status %s, B status %s)' % (name, res['a'][1], res['b'][1]))
            continue
        m, (fa, fb, sa, sb, na, nb) = judge(a, b)
        ok = all(m.values())
        passed += ok
        sub = b['values'].get('substeps', 0) / max(a['values'].get('substeps', 1), 1)
        cpu = res['b'][0] / max(res['a'][0], 1e-9)
        print('%-30s %s  %8.2fx %8.2fx  %g -> %g, %.2f -> %.2f, %d -> %d' % (name, ' '.join('%-6s' % ('ok' if m[k] else 'DIFF') for k in keys), sub, cpu, fa, fb, sa, sb, na, nb))
    print('fidelity: %d/%d scenes pass every metric; CPU A %.1f s, B %.1f s' % (passed, len(results), sum(r['a'][0] for _, r in results), sum(r['b'][0] for _, r in results)))


if __name__ == '__main__':
    sys.exit(main())
