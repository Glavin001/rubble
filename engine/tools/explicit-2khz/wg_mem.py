#!/usr/bin/env python3
"""Workgroup memory per explicit WGSL module (bytes declared with var<workgroup>).

Usage: python3 engine/tools/explicit-2khz/wg_mem.py <wgsl dir> [module ...]
Counts every workgroup variable a module declares (an upper bound of what an entry uses).
"""
import os
import re
import sys

SIZES = {"u32": 4, "i32": 4, "f32": 4, "vec2<f32>": 8, "vec3<f32>": 16, "vec4<f32>": 16,
         "vec4<u32>": 16, "atomic<u32>": 4, "atomic<i32>": 4}


def size(t):
    t = t.strip()
    a = re.match(r"array<(.+),\s*(?:i32\(|u32\()?(\d+)[iu]?\)?\s*>$", t)
    if a:
        return size(a.group(1)) * int(a.group(2))
    return SIZES.get(t, 16)


def main():
    d = sys.argv[1]
    names = sys.argv[2:] or sorted(f for f in os.listdir(d) if f.startswith("explicit_beams") and f.endswith(".wgsl"))
    for f in names:
        s = open(os.path.join(d, f)).read()
        vars_ = re.findall(r"var<workgroup>\s+(\w+)\s*:\s*([^;]+);", s)
        tot = sum(size(t) for _, t in vars_)
        print(f"{f:50} {tot:6} bytes ({len(vars_)} vars)")
        if os.environ.get("WG_VERBOSE"):
            for n, t in vars_:
                print(f"    {n:30} {size(t):6} {t}")


main()
