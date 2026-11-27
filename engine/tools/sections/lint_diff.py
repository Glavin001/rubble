#!/usr/bin/env python3
"""The generated-vehicle lint's failing rules, two gate runs side by side (front_gates.sh logs):
which (design, rule) fail in one and not the other, and the lines that changed.

    python3 -I tools/sections/lint_diff.py out/gates/none/generated_vehicle_lint.log out/gates/cell/generated_vehicle_lint.log
"""
import re
import sys


def rules(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^(\w+) (\w+) car: (.*)$", line.strip())
        if m:
            out[(m.group(1), m.group(2))] = m.group(3)
    return out


a, b = rules(sys.argv[1]), rules(sys.argv[2])
for k in sorted(set(a) | set(b)):
    if k not in a:
        print(f"NEW   {k[0]} {k[1]}: {b[k][:220]}")
    elif k not in b:
        print(f"GONE  {k[0]} {k[1]}: {a[k][:220]}")
    elif a[k] != b[k]:
        print(f"CHANGED {k[0]} {k[1]}:\n   before {a[k][:220]}\n   after  {b[k][:220]}")
