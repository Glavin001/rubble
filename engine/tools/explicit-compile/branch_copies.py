#!/usr/bin/env python3
"""Per function of a WGSL entry: copies after inlining x its own branches (if/loop/
switch) and workgroup/storage reads: where the inlined body's control flow comes from.
  branch_copies.py MODULE.wgsl [ENTRY] [--top N]   (measurement only)"""
import re, sys
from collections import defaultdict
sys.path.insert(0, __import__("os").path.dirname(__file__))
from inline_size import parse, IDENT_CALL

args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and sys.argv[i] != "--top"]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
text = open(args[0]).read()
f = parse(text)
entry = args[1] if len(args) > 1 else re.search(r"@compute[^\n]*\n(?:@[^\n]*\n)*fn\s+([A-Za-z_0-9]+)", text).group(1)
calls = {n: defaultdict(int) for n in f}
for n, b in f.items():
    for m in IDENT_CALL.finditer(b):
        if m.group(1) in f and m.group(1) != n:
            calls[n][m.group(1)] += 1
order, seen = [], set()
def dfs(n):
    if n in seen: return
    seen.add(n)
    for c in calls[n]: dfs(c)
    order.append(n)
dfs(entry); order.reverse()
cp = defaultdict(int); cp[entry] = 1
for n in order:
    for c, k in calls[n].items(): cp[c] += cp[n] * k
br = {n: len(re.findall(r"\bif\s*\(|\bloop\b|\bswitch\b", f[n])) for n in order}
tot = sum(cp[n] * br[n] for n in order)
print(f"{entry}: branches after inlining {tot}")
for n in sorted(order, key=lambda n: -cp[n] * br[n])[:top]:
    print(f"{n:36} copies {cp[n]:6} branches {br[n]:4}  x {cp[n]*br[n]:7} ({100*cp[n]*br[n]/tot:4.1f}%)")
