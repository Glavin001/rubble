#!/usr/bin/env python3
"""How big each WGSL module's entry point is once every call is inlined.

    engine/scripts/wgsl-inline-size.py <dir-or-files...> [--top N] [--callers N] [--subtrees N] [--entry NAME]

Metal's compiler (and so Chrome on macOS) inlines every function into the entry
point, and its compile time grows faster than linearly with that one function's
size. A module's bytes say little: a helper called from five places counts five
times. This prints, per module, its bytes, functions, and the entry point's
inlined size (lines, loops, barriers, buffer accesses), then the functions that
contribute most (own lines x the number of times they end up inlined), and with
--callers the most copied functions with the callers that multiply them, with
--subtrees the largest functions as inlined (a function with all it calls).

Pure text analysis (no GPU, no compiler): function bodies are found by brace
matching, calls by name. Good for ranking and for seeing what a change does to
the inlined size before paying for a compile.
"""
import os
import re
import sys
from functools import lru_cache

FN = re.compile(r'^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', re.M)
ENTRY = re.compile(r'@compute[^\n]*\n(?:@[^\n]*\n)*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def functions(src):
    """name -> body text (from its opening brace to the matching one)."""
    out = {}
    for m in FN.finditer(src):
        i = src.index('{', m.end())
        depth, j = 0, i
        while True:
            c = src[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(1)] = src[i:j + 1]
    return out


def analyse(path, top, entry_name, callers=0, subtrees=0):
    src = open(path).read()
    fns = functions(src)
    names = set(fns)
    own, calls = {}, {}
    for name, body in fns.items():
        c = {}
        for w in WORD.findall(body):
            if w in names and w != name:
                c[w] = c.get(w, 0) + 1
        calls[name] = c
        own[name] = {
            'lines': body.count('\n'),
            'loops': len(re.findall(r'\b(?:loop|for|while)\b', body)),
            'barriers': body.count('workgroupBarrier'),
            'access': body.count('['),
        }

    @lru_cache(maxsize=None)
    def inlined(name):
        t = dict(own[name])
        for callee, n in calls[name].items():
            sub = dict(inlined(callee))
            for k in t:
                t[k] += n * sub[k]
        return tuple(sorted(t.items()))

    def times(root):
        """How many copies of each function end up in `root`."""
        count = {}

        def walk(name, mult):
            count[name] = count.get(name, 0) + mult
            for callee, n in calls[name].items():
                walk(callee, mult * n)
        walk(root, 1)
        return count

    entries = [m.group(1) for m in ENTRY.finditer(src)]
    if entry_name:
        entries = [e for e in entries if e == entry_name]
    rows = []
    for e in entries:
        t = dict(inlined(e))
        rows.append((os.path.basename(path), e, len(src), len(fns), t))
        if top:
            copies = times(e)
            heavy = sorted(((own[f]['lines'] * n, n, own[f]['lines'], f) for f, n in copies.items()), reverse=True)[:top]
            for total, n, lines, f in heavy:
                print(f'        {total:8d} lines = {lines:5d} x {n:4d} copies  {f}')
        if subtrees:
            # The biggest functions as inlined (their whole subtree), with their copies.
            copies = times(e)
            big = sorted(((dict(inlined(f))['lines'], f) for f in copies), reverse=True)[:subtrees]
            for lines, f in big:
                print(f'        {lines:8d} inlined lines x {copies[f]:4d} copies  {f}')
        if callers:
            # Why a function has its copies: each caller's call sites x its own copies.
            copies = times(e)
            many = sorted(((n, f) for f, n in copies.items() if own[f]['lines'] >= 6 and n > 3), reverse=True)[:callers]
            for n, f in many:
                who = sorted(((calls[c][f] * copies[c], calls[c][f], copies[c], c) for c in copies if f in calls[c]), reverse=True)[:6]
                print(f'        {f} ({own[f]["lines"]} lines) x {n}: ' + ', '.join(f'{c} [{k} sites x {cp}]' for _, k, cp, c in who))
    return rows


def main():
    args = sys.argv[1:]
    top, entry, callers, subtrees = 0, None, 0, 0
    if '--subtrees' in args:
        i = args.index('--subtrees')
        subtrees = int(args[i + 1])
        del args[i:i + 2]
    if '--callers' in args:
        i = args.index('--callers')
        callers = int(args[i + 1])
        del args[i:i + 2]
    if '--top' in args:
        i = args.index('--top')
        top = int(args[i + 1])
        del args[i:i + 2]
    if '--entry' in args:
        i = args.index('--entry')
        entry = args[i + 1]
        del args[i:i + 2]
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths += sorted(os.path.join(a, f) for f in os.listdir(a) if f.endswith('.wgsl'))
        else:
            paths.append(a)
    if not paths:
        print(__doc__, file=sys.stderr)
        return 2
    sys.setrecursionlimit(10000)
    print(f'{"module":44s} {"KB":>6s} {"fns":>5s} {"inlined lines":>14s} {"loops":>7s} {"barriers":>9s} {"accesses":>9s}')
    all_rows = []
    for p in paths:
        if top or callers or subtrees:
            print(os.path.basename(p))
        all_rows += analyse(p, top, entry, callers, subtrees)
    for name, e, size, nf, t in sorted(all_rows, key=lambda r: r[4]['lines']):
        print(f'{name[:-5]:44s} {size / 1024:6.0f} {nf:5d} {t["lines"]:14d} {t["loops"]:7d} {t["barriers"]:9d} {t["access"]:9d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
