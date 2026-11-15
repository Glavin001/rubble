#!/usr/bin/env python3
"""Measurement only: copy a Slang CUDA module with the named functions __noinline__ (a
proxy for "one copy of this function" in nvcc's compile time).
  cu_noinline.py IN.cu OUT.cu fn1,fn2,...   (names as in the WGSL/CUDA, e.g. tryContact_0)"""
import re, sys, shutil
src, dst, names = sys.argv[1], sys.argv[2], [n for n in sys.argv[3].split(",") if n]
t = open(src).read()
for n in names:
    t2, k = re.subn(r"^(static )?__device__ ([^\n(]*?\b" + re.escape(n) + r"\()", r"\1__device__ __noinline__ \2", t, flags=re.M)
    if k == 0:
        print("not found", n)
    t = t2
open(dst, "w").write(t)
j = src[:-3] + ".json"
shutil.copy(j, dst[:-3] + ".json")
