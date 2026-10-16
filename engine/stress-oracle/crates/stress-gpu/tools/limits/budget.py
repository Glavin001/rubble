#!/usr/bin/env python3
"""Static kernel budget: per-kernel driver statistics and machine-code size against the
measured limits (instruction-cache knee, stack knee) and a committed baseline.

Driven by scripts/kernel_budget.sh, which builds world.metallib exactly as build.rs does and
collects the inputs. Statistics come from a PRIVATE Metal API (pipeline option 0x40000 ->
MTLComputePipelineReflection.performanceStatistics); a value of -1 means it was unavailable
and that comparison is skipped.

usage: budget.py --stats stats.tsv --code code.txt --slang world.slang --limits limits.toml
                 --baseline kernel_budget.toml [--update] [--margin 5] [--meta k=v ...] [--json out.json]
exit: 0 ok, 1 regression beyond the margin (unless --update), 2 usage/input error
"""
import argparse
import datetime
import json
import os
import re
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11: a minimal reader for the flat files used here
    tomllib = None


def read_toml(path):
    if not os.path.exists(path):
        return {}
    if tomllib:
        with open(path, "rb") as f:
            return tomllib.load(f)
    data, cur = {}, None
    for line in open(path):
        line = line.split("#", 1)[0].strip() if not line.strip().startswith('"') else line.strip()
        m = re.match(r"^\[([^\]]+)\]$", line)
        if m:
            cur = data
            for part in m.group(1).split("."):
                cur = cur.setdefault(part, {})
            continue
        m = re.match(r"^(\w+)\s*=\s*(.+)$", line)
        if m and cur is not None:
            k, v = m.groups()
            if v.startswith('"'):
                cur[k] = v.strip('"')
            elif v.startswith("[") and v.endswith("]"):
                try:
                    cur[k] = [float(x) for x in v[1:-1].split(",") if x.strip()]
                except ValueError:
                    pass
            else:
                try:
                    cur[k] = float(v) if "." in v else int(v)
                except ValueError:
                    cur[k] = v
    return data


def numthreads(slang_path):
    """[numthreads(X, Y, Z)] followed by the entry point's name."""
    src = open(slang_path).read()
    out = {}
    for m in re.finditer(r"\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]\s*(?:\[[^\]]*\]\s*)*void\s+(\w+)\s*\(", src):
        out[m.group(4)] = int(m.group(1)) * int(m.group(2)) * int(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--slang", required=True)
    ap.add_argument("--limits", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--margin", type=float, default=5.0, help="percent growth allowed for instructions and code bytes")
    ap.add_argument("--meta", action="append", default=[])
    ap.add_argument("--json", help="also write the per-kernel numbers and the limits used (tests/kernel_limits.rs)")
    a = ap.parse_args()

    threads = numthreads(a.slang)
    code = {}
    if os.path.exists(a.code):
        for line in open(a.code):
            p = line.split()
            if len(p) == 2:
                code[p[0]] = int(p[1])
    kernels = {}
    for line in open(a.stats):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        name = f[0]
        instr, tregs, spill, _tgmem, maxthr, _alu, _br, _dl, static_tg = (int(x) for x in f[1:10])
        t = threads.get(name, -1)
        kernels[name] = {
            "instructions": instr,
            "temp_registers": tregs,
            "spilled_bytes": spill,
            "threads_per_threadgroup": t,
            "spilled_bytes_x_threads": spill * t if spill >= 0 and t > 0 else -1,
            "static_threadgroup_memory": static_tg,
            "max_threads_per_threadgroup": maxthr,
            "code_bytes": code.get(name, -1),
        }
    if not kernels:
        print("budget: no kernels in stats", file=sys.stderr)
        return 2
    stats_ok = any(k["instructions"] >= 0 for k in kernels.values())
    if not stats_ok:
        print("budget: WARNING driver statistics unavailable (private pipeline option 0x40000); "
              "only machine-code size is compared", file=sys.stderr)

    lim = read_toml(a.limits)
    ic = lim.get("instruction_cache", {})
    st = lim.get("stack", {})
    icache_bytes = (ic.get("alu_knee_code_bytes") or [None])[0]
    icache_instr = (ic.get("alu_knee_instructions") or [None])[0]
    stack_bytes = (st.get("array_sweep_knee_bytes_per_threadgroup") or [None])[0]
    if isinstance(icache_bytes, str):
        icache_bytes = None
    if isinstance(stack_bytes, str):
        stack_bytes = None

    def ratio(v, knee):
        return f"{v / knee:5.1f}x" if knee and v is not None and v >= 0 else "    -"

    print(f"limits: {a.limits}  instruction-cache knee {icache_bytes} code bytes ({icache_instr} instructions), "
          f"stack knee {stack_bytes} bytes per threadgroup")
    hdr = f"{'kernel':20s} {'instr':>6s} {'regs':>4s} {'spillB':>6s} {'thr':>4s} {'spill*thr':>9s} {'tgmem':>5s} {'maxthr':>6s} {'codeB':>7s} {'code/icache':>11s} {'stack/knee':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(kernels):
        k = kernels[name]
        print(f"{name:20s} {k['instructions']:6d} {k['temp_registers']:4d} {k['spilled_bytes']:6d} {k['threads_per_threadgroup']:4d} "
              f"{k['spilled_bytes_x_threads']:9d} {k['static_threadgroup_memory']:5d} {k['max_threads_per_threadgroup']:6d} {k['code_bytes']:7d} "
              f"{ratio(k['code_bytes'], icache_bytes):>11s} {ratio(k['spilled_bytes_x_threads'], stack_bytes):>10s}")

    if a.json:
        ic_mixed = (ic.get("mixed_knee_code_bytes") or [None])[0]
        lr = lim.get("live_registers", {})
        dev = lim.get("device", {})
        with open(a.json, "w") as f:
            json.dump({
                "limits_file": a.limits,
                "limits": {
                    "icache_code_bytes": icache_bytes,
                    "icache_instructions": icache_instr,
                    "icache_mixed_code_bytes": ic_mixed if not isinstance(ic_mixed, str) else None,
                    "stack_bytes_per_threadgroup": stack_bytes,
                    "max_temp_registers": lr.get("max_temp_registers_seen"),
                    "max_threadgroup_memory_bytes": dev.get("max_threadgroup_memory_bytes"),
                },
                "kernels": kernels,
            }, f, indent=1)

    base = read_toml(a.baseline).get("kernels", {})
    problems, notes = [], []
    m = 1 + a.margin / 100
    for name, k in sorted(kernels.items()):
        b = base.get(name)
        if b is None:
            notes.append(f"{name}: new kernel (not in baseline)")
            continue
        for key in ("instructions", "code_bytes"):
            if k[key] >= 0 and b.get(key, -1) >= 0 and k[key] > b[key] * m:
                problems.append(f"{name}: {key} {b[key]} -> {k[key]} (+{(k[key] / b[key] - 1) * 100:.1f}% > {a.margin}%)")
        if k["spilled_bytes"] >= 0 and b.get("spilled_bytes", -1) >= 0 and k["spilled_bytes"] > b["spilled_bytes"]:
            problems.append(f"{name}: spilled_bytes {b['spilled_bytes']} -> {k['spilled_bytes']}")
        if k["temp_registers"] >= 0 and b.get("temp_registers", -1) >= 0 and k["temp_registers"] > b["temp_registers"]:
            notes.append(f"{name}: temp_registers {b['temp_registers']} -> {k['temp_registers']}")
        if k["max_threads_per_threadgroup"] < b.get("max_threads_per_threadgroup", 0) and k["max_threads_per_threadgroup"] < max(k["threads_per_threadgroup"], 1):
            problems.append(f"{name}: maxTotalThreadsPerThreadgroup {k['max_threads_per_threadgroup']} below its threadgroup size")
    for name in sorted(set(base) - set(kernels)):
        notes.append(f"{name}: in baseline, no longer built")
    for n in notes:
        print(f"note: {n}")

    if a.update:
        meta = dict(x.split("=", 1) for x in a.meta)
        meta.setdefault("date", datetime.date.today().isoformat())
        with open(a.baseline, "w") as f:
            f.write("# Kernel budget baseline: world.metallib built like build.rs (precise math).\n"
                    "# Regenerate with scripts/kernel_budget.sh --update after an intended change.\n"
                    "# instructions/temp_registers/spilled_bytes: driver statistics (PRIVATE Metal pipeline option 0x40000);\n"
                    "# code_bytes: __TEXT,__text of the AGX binary from Xcode's applegpu-nt; -1 = unavailable.\n\n[meta]\n")
            for key, v in meta.items():
                f.write(f'{key} = "{v}"\n')
            for name in sorted(kernels):
                f.write(f"\n[kernels.{name}]\n")
                for key, v in kernels[name].items():
                    f.write(f"{key} = {v}\n")
        print(f"baseline written: {a.baseline}")
        return 0
    if not base:
        print(f"budget: no baseline at {a.baseline}; run with --update to create it", file=sys.stderr)
        return 1
    if problems:
        print("REGRESSION:")
        for p in problems:
            print(f"  {p}")
        print("(intended? rerun with --update to accept)")
        return 1
    print(f"ok: no kernel regressed beyond {a.margin}% instructions/code bytes or gained spilled bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
