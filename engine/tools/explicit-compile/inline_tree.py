#!/usr/bin/env python3
"""The inlined-size tree of a WGSL entry point: each call's callee with its inlined size,
recursing into subtrees above --min lines (measurement only; see inline_size.py)."""
import sys
from functools import lru_cache
from collections import defaultdict
sys.path.insert(0, __import__("os").path.dirname(__file__))
from inline_size import parse, IDENT_CALL

def main():
    path = sys.argv[1]
    entry = sys.argv[2]
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    funcs = parse(open(path).read())
    calls, own = {}, {}
    for n, b in funcs.items():
        cs = defaultdict(int)
        for m in IDENT_CALL.finditer(b):
            if m.group(1) in funcs and m.group(1) != n:
                cs[m.group(1)] += 1
        calls[n] = dict(cs)
        own[n] = sum(1 for l in b.splitlines() if l.strip())

    @lru_cache(None)
    def inl(n):
        return own[n] + sum(k * inl(c) for c, k in calls[n].items())

    def show(n, depth, mult):
        print(f"{'  '*depth}{n} own {own[n]} inl {inl(n)}" + (f" x{mult}" if mult > 1 else ""))
        for c, k in sorted(calls[n].items(), key=lambda x: -x[1] * inl(x[0])):
            if k * inl(c) >= lo:
                show(c, depth + 1, k)
    show(entry, 0, 1)

main()
