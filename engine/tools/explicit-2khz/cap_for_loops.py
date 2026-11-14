#!/usr/bin/env python3
"""Rewrites a Slang file's counted `for` loops into explicit_loop_cap.slang's clamped
form (EB_END), in place: `for (uint i = a; i < n; i += s)` becomes
`for (uint i = a, iStop = EB_END(i, n, s); i < iStop; i += s)`; a condition's further
`&& ...` terms are kept after the clamp. Loops with a constant bound (a literal or an
ALL_CAPS constant) are left as written. Prints each rewritten line. (Used once on
vehicle_coupling.slang; barrier_loops.py then marks the loops that reach a barrier.)
Usage: cap_for_loops.py <file.slang>
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()
pat = re.compile(
    r"for \(uint (\w+) = ([^;,]+); \1 < ([^;&]+?)((?:\s*&&[^;]*)?); (\1\+\+|\1 \+= [^)]+)\)"
)


def const(b):
    b = b.strip()
    return bool(re.fullmatch(r"\d+u?|[A-Z][A-Z0-9_]*", b))


def rewrite(m):
    var, start, bound, rest, step = m.groups()
    if const(bound):
        return m.group(0)
    s = "1u" if step.endswith("++") else step.split("+=", 1)[1].strip()
    out = f"for (uint {var} = {start}, {var}Stop = EB_END({var}, {bound.strip()}, {s}); {var} < {var}Stop{rest}; {step})"
    line = src.count("\n", 0, m.start()) + 1
    print(f"{line}: {out}")
    return out


open(path, "w").write(pat.sub(rewrite, src))
