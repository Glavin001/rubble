#!/usr/bin/env python3
"""Machine-code size of compute functions in a .metallib, from Xcode's offline AGX
translator (applegpu-nt, PRIVATE/undocumented tooling shipped inside Xcode).

One translator run compiles every requested function into one Mach-O whose __TEXT,__compute
section holds a nested Mach-O per function (symbols __<fn>_begin/_end); the size of each
nested __TEXT,__text (main program + constant program) is the function's code bytes.

usage: codesize.py [--arch applegpu_g15s] [--platform 14.4] <lib.metallib> <function> [...]
prints: <function> <text_bytes>   (functions that failed are omitted; exit 0 if any printed)
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

NT = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/applegpu-nt"


def load_commands(buf, base):
    magic = struct.unpack_from("<I", buf, base)[0]
    if magic != 0xFEEDFACF:
        raise ValueError(f"not a 64-bit Mach-O at {base}")
    ncmds = struct.unpack_from("<I", buf, base + 16)[0]
    off = base + 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", buf, off)
        yield cmd, off
        off += size


def sections(buf, base):
    for cmd, off in load_commands(buf, base):
        if cmd == 0x19:  # LC_SEGMENT_64
            nsect = struct.unpack_from("<I", buf, off + 64)[0]
            for s in range(nsect):
                so = off + 72 + 80 * s
                sect = buf[so:so + 16].rstrip(b"\0").decode()
                seg = buf[so + 16:so + 32].rstrip(b"\0").decode()
                _addr, size, fileoff = struct.unpack_from("<QQI", buf, so + 32)
                yield seg, sect, size, fileoff


def symbols(buf, base):
    for cmd, off in load_commands(buf, base):
        if cmd == 0x2:  # LC_SYMTAB
            symoff, nsyms, stroff, _ = struct.unpack_from("<IIII", buf, off + 8)
            for i in range(nsyms):
                strx, _t, _s, _d, val = struct.unpack_from("<IBBHQ", buf, base + symoff + 16 * i)
                name = buf[base + stroff + strx:].split(b"\0", 1)[0].decode(errors="replace")
                yield name, val


def text_sizes(binary, functions):
    d = open(binary, "rb").read()
    syms = dict(symbols(d, 0))
    out = {}
    for fn in functions:
        begin = syms.get(f"__{fn}_begin")
        if begin is None:
            continue
        for seg, sect, size, _ in sections(d, begin):
            if seg == "__TEXT" and sect == "__text":
                out[fn] = size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="applegpu_g15s")
    ap.add_argument("--platform", default=None, help="macOS version for -platform_version (default: this OS)")
    ap.add_argument("lib")
    ap.add_argument("functions", nargs="+")
    a = ap.parse_args()
    if not os.path.exists(NT):
        print(f"codesize: {NT} not found (Xcode needed)", file=sys.stderr)
        return 1
    plat = a.platform or subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip() or "14.4"
    lib = os.path.abspath(a.lib)
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "p.mtlp-json")
        with open(script, "w") as f:
            json.dump({"libraries": {"paths": [{"label": "lib", "path": lib}]},
                       "pipelines": {"compute_pipelines": [{"compute_function": f"alias:lib#{fn}"} for fn in a.functions]}}, f)
        out = os.path.join(tmp, "p.bin")
        r = subprocess.run([NT, "-arch", a.arch, "-platform_version", "macos", plat, plat, "-N", script, lib, "-o", out],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print(f"codesize: applegpu-nt failed ({r.returncode}): {r.stderr.strip()[-400:]}", file=sys.stderr)
            return 1
        sizes = text_sizes(out, a.functions)
    for fn in a.functions:
        if fn in sizes:
            print(f"{fn} {sizes[fn]}")
    return 0 if sizes else 1


if __name__ == "__main__":
    sys.exit(main())
