#!/usr/bin/env python3
"""The asserted criteria of two crash reference reports side by side (engine/docs/crash-reference.md):
the CPU reference's (`crash_ref -- ID --json`, one report a line) and the web page's
(`crash-ref-record.mjs`'s report.json).

    python3 engine/tools/crash-reference/compare.py CPU.json PAGE.json [--all]

`--all`: every criterion, not only the asserted ones.
"""
import json
import sys


def load(path):
    text = open(path).read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # crash_ref --json prints one report a line (one per run): the first.
        return json.loads(text.splitlines()[0])


def fmt(v):
    return '—' if v is None else f'{v:.4g}'


def band(r):
    lo, hi = r.get('lo'), r.get('hi')
    return f"[{fmt(lo) if lo is not None else '-inf'}, {fmt(hi) if hi is not None else 'inf'}]"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 2:
        sys.exit(__doc__)
    every = '--all' in sys.argv
    a, b = load(args[0]), load(args[1])
    print(f"{a['id']} contact {a['contact']} run {a['run']}  (CPU)  vs  {b['id']} contact {b['contact']} run {b['run']}  (page)")
    rb = {r['name']: r for r in b['results']}
    w = max(len(r['name']) for r in a['results'])
    print(f"{'criterion':<{w}}  {'metric':<28} {'band':<18} {'CPU':>10} {'page':>10}  CPU/page")
    for r in a['results']:
        if not (every or r['asserted']):
            continue
        p = rb.get(r['name'], {})
        ok = lambda x: 'pass' if x.get('pass') else 'FAIL' if x else '—'
        print(f"{r['name']:<{w}}  {r['metric']:<28} {band(r):<18} {fmt(r.get('value')):>10} {fmt(p.get('value')):>10}  {ok(r)}/{ok(p)}")


if __name__ == '__main__':
    main()
