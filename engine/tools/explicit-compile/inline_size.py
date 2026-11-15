#!/usr/bin/env python3
"""The size of a WGSL entry point after full inlining (what a GPU driver that inlines every
call compiles), and which functions make it up: per function its own size (non-blank
lines, bytes), its copies after inlining (call paths from the entry) and its share.

  python3 inline_size.py MODULE.wgsl [ENTRY] [--top N]

Measurement only. WGSL as slangc emits it: top-level `fn name(...) ... {` with braces
balanced in the body.
"""
import re
import sys
from collections import defaultdict
from functools import lru_cache

FN = re.compile(r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
IDENT_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse(text):
    funcs = {}
    for m in FN.finditer(text):
        name = m.group(1)
        i = text.index("{", m.end())
        depth = 0
        j = i
        while True:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i : j + 1]
        funcs[name] = body
    return funcs


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 30
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    text = open(args[0]).read()
    funcs = parse(text)
    entry = args[1] if len(args) > 1 and args[1] in funcs else None
    if entry is None:
        # The entry: the function after @compute.
        m = re.search(r"@compute[^\n]*\n(?:@[^\n]*\n)*fn\s+([A-Za-z_0-9]+)", text)
        entry = m.group(1)
    calls = {}
    own = {}
    for name, body in funcs.items():
        cs = defaultdict(int)
        for m in IDENT_CALL.finditer(body):
            if m.group(1) in funcs and m.group(1) != name:
                cs[m.group(1)] += 1
        calls[name] = dict(cs)
        own[name] = sum(1 for l in body.splitlines() if l.strip())

    @lru_cache(None)
    def inlined(n):
        return own[n] + sum(k * inlined(c) for c, k in calls[n].items())

    copies = defaultdict(int)

    def walk_counts():
        # copies[f] = number of call paths from entry to f (topological accumulate).
        order = []
        seen = set()

        def dfs(n):
            if n in seen:
                return
            seen.add(n)
            for c in calls[n]:
                dfs(c)
            order.append(n)

        dfs(entry)
        order.reverse()
        copies[entry] = 1
        for n in order:
            for c, k in calls[n].items():
                copies[c] += copies[n] * k
        return order

    order = walk_counts()
    total = inlined(entry)
    print(f"{args[0]}: entry {entry}, {len(order)} reachable functions of {len(funcs)}, "
          f"own lines {sum(own[n] for n in order)}, inlined lines {total}")
    rows = sorted(order, key=lambda n: -copies[n] * own[n])
    print(f"{'function':48} {'own':>6} {'copies':>7} {'own*copies':>11} {'share':>6} {'inlined(each)':>13}")
    for n in rows[:top]:
        print(f"{n:48} {own[n]:6} {copies[n]:7} {own[n]*copies[n]:11} {100*own[n]*copies[n]/total:5.1f}% {inlined(n):13}")


if __name__ == "__main__":
    main()
