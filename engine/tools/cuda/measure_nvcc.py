#!/usr/bin/env python3
"""Per-module nvcc cost of engine-cuda's kernels: wall time, peak RSS and nvcc's own
per-phase times (cicc = front end + NVVM per virtual arch, ptxas per SASS target).

Runs nvcc with the flags `engine_slang_build::cuda_layout::fatbin_threads` uses, on
CUDA sources engine-cuda's build.rs already generated (copy its OUT_DIR's `cuda`,
`cuda_explicit` and `cuda_terrain` folders into SRC). Measurement only: nothing it
writes is used by a build. Run where nvcc is (the CI image:
engine/tools/cuda/build_tests_in_docker.sh's image), e.g.

  docker run --rm --platform linux/amd64 -v "$PWD:/repo" rubble-cuda-build:slang-2026.19-cuda-12.8 \
    python3 engine/tools/cuda/measure_nvcc.py SRC OUT.csv --jobs 3 --sass 75,89

`--only a,b*` limits the modules (a trailing `*` is a prefix); `--extra "-O1"` adds
flags (reports only); `--split` measures each SASS target on its own run as well.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def threads(cu: Path) -> int:
    j = cu.with_suffix(".json")
    if not j.exists():
        return 0
    try:
        eps = json.loads(j.read_text())["entryPoints"]
        t = eps[0].get("threadGroupSize", [0, 1, 1])
        return t[0] * t[1] * t[2]
    except Exception:
        return 0


def nvcc_cmd(nvcc, cu, dest, sass, ptx, cap_threads, extra, timing):
    cmd = [nvcc, "-fatbin", "-std=c++17", "-lineinfo"]
    if cap_threads > 256:
        regs = min(65536 // cap_threads, 255) // 8 * 8
        cmd.append(f"-maxrregcount={regs}")
    cmd += ["-diag-suppress", "20044", "-Xptxas", "-dlcm=cg"]
    for sm in sass:
        cmd.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    if ptx:
        cmd.append(f"-gencode=arch={ptx},code={ptx}")
    cmd += extra
    cmd += ["-time", str(timing), "-o", str(dest), str(cu)]
    return cmd


WRAP = r"""
import os, resource, subprocess, sys, time
t0 = time.monotonic()
r = subprocess.run(sys.argv[1:], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
wall = time.monotonic() - t0
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
sys.stderr.write(r.stderr.decode(errors="replace"))
print(f"{r.returncode} {wall:.2f} {rss:.0f}")
"""


def measure(nvcc, cu, sass, ptx, extra, tmp, tag):
    timing = tmp / f"{cu.stem}.{tag}.time.csv"
    if timing.exists():
        timing.unlink()
    dest = tmp / f"{cu.stem}.{tag}.fatbin"
    cap = threads(cu) if cu.parent.name == "cuda_explicit" else 0
    cmd = nvcc_cmd(nvcc, cu, dest, sass, ptx, cap, extra, timing)
    r = subprocess.run([sys.executable, "-c", WRAP, *cmd], capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "1 0 0"
    code, wall, rss = last.split()
    phases = {}
    if timing.exists():
        for row in csv.reader(timing.open()):
            # source, phase, inputs, output, arch, tool, time, unit
            if len(row) < 8 or row[0].strip().startswith("source file"):
                continue
            name, arch, ms = row[1].strip(), row[4].strip(), row[6].strip()
            try:
                phases[f"{name}:{arch}"] = phases.get(f"{name}:{arch}", 0.0) + float(ms) / 1000.0
            except ValueError:
                pass
    size = dest.stat().st_size if dest.exists() else 0
    if dest.exists():
        dest.unlink()
    return {
        "module": cu.stem,
        "dir": cu.parent.name,
        "tag": tag,
        "cu_kb": cu.stat().st_size // 1024,
        "threads": threads(cu),
        "ok": code == "0",
        "wall_s": float(wall),
        "rss_mb": float(rss),
        "fatbin_kb": size // 1024,
        "phases": phases,
        "err": "" if code == "0" else r.stderr[-400:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--sass", default="75,89")
    ap.add_argument("--ptx", default="compute_75")
    ap.add_argument("--only", default="")
    ap.add_argument("--dirs", default="cuda,cuda_explicit,cuda_terrain")
    ap.add_argument("--extra", default="")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    a = ap.parse_args()
    src = Path(a.src)
    only = [x for x in a.only.split(",") if x]

    def wanted(name):
        return not only or any(name.startswith(p[:-1]) if p.endswith("*") else name == p for p in only)

    files = [cu for d in a.dirs.split(",") for cu in sorted((src / d).glob("*.cu")) if wanted(cu.stem)]
    # Largest first, so the long ones do not end the run alone.
    files.sort(key=lambda p: -p.stat().st_size)
    sass = [s for s in a.sass.split(",") if s]
    extra = a.extra.split()
    work = []
    for cu in files:
        work.append((cu, sass, a.ptx, "all"))
        if a.split:
            for i, sm in enumerate(sass):
                work.append((cu, [sm], a.ptx if i == 0 else "", f"sm_{sm}"))
    tmp = Path(tempfile.mkdtemp(prefix="nvcc-measure-"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with out.open("w") as f, ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(measure, a.nvcc, cu, s, p, extra, tmp, tag) for cu, s, p, tag in work]
        for fut in futs:
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(
                f"{r['dir']:14} {r['module']:44} {r['tag']:6} {r['cu_kb']:6} KB "
                f"{r['wall_s']:8.1f} s {r['rss_mb']:7.0f} MB {'ok' if r['ok'] else 'FAIL ' + r['err']}",
                flush=True,
            )
    print(f"total {time.monotonic() - t0:.0f} s for {len(work)} nvcc runs at {a.jobs} jobs", flush=True)


if __name__ == "__main__":
    main()
