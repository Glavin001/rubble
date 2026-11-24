#!/usr/bin/env bash
# Car-to-car contact variants judged over an ensemble: the same head-on with car B moved
# sideways by 0..N-1 mm (the crash is chaotic: a 1 mm nudge moves a single run's edge count
# from 51 to 179), each run's worst-moment numbers, then per variant the min / median / max.
#
#   tools/c2c/ensemble.sh KMH SECONDS N "NAME:VAR=x VAR=y" ["NAME:..."] ...
#
# From engine/. Uses the release vehicle_c2c_overlap example (built here). Logs and the
# summary go to out/c2c-fuse/ens/.
set -euo pipefail
kmh="$1"; secs="$2"; n="$3"; shift 3
out=out/c2c-fuse/ens
mkdir -p "$out"
cargo build --release -p engine-softbody-lab --example vehicle_c2c_overlap >/dev/null 2>&1
bin="$out/probe"
cp target/release/examples/vehicle_c2c_overlap "$bin"
pids=()
for spec in "$@"; do
  name="${spec%%:*}"; vars="${spec#*:}"
  for ((k = 0; k < n; k++)); do
    off=$(printf '0.%03d' "$k")
    # shellcheck disable=SC2086
    env $vars LEDGER=1 OFFSET="$off" "$bin" sedan "$kmh" "$secs" > "$out/${name}_${kmh}_${k}.log" 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done
python3 - "$out" "$kmh" "$n" "$@" <<'EOF'
import re, statistics, sys
out, kmh, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(f"head on {kmh} + {kmh} km/h, {n} runs each (car B 0..{n-1} mm aside): min / median / max")
print(f"{'variant':<14} {'nodes inside A+B':>22} {'edges through A+B':>22} {'overlap L':>22} {'energy rise kJ (20 ms)':>22}")
for spec in sys.argv[4:]:
    name = spec.split(":")[0]
    rows = []
    for k in range(n):
        t = open(f"{out}/{name}_{kmh}_{k}.log").read()
        a = re.search(r"worst A in B: nodes (\d+).*skin edges through triangles (\d+)", t)
        b = re.search(r"worst B in A: nodes (\d+).*skin edges through triangles (\d+)", t)
        v = re.search(r"largest volume ([\d.]+) L", t)
        mech = [float(x) for x in re.findall(r"mechanical ([\d.]+) \(start", t)]
        start = float(re.search(r"\(start ([\d.]+)\)", t).group(1))
        rows.append((int(a[1]) + int(b[1]), int(a[2]) + int(b[2]), float(v[1]), max(m - start for m in mech)))
    def f(i):
        x = sorted(r[i] for r in rows)
        return f"{x[0]:.0f} / {statistics.median(x):.0f} / {x[-1]:.0f}" if i < 3 else f"{x[0]:.2f} / {statistics.median(x):.2f} / {x[-1]:.2f}"
    print(f"{name:<14} {f(0):>22} {f(1):>22} {f(2):>22} {f(3):>22}")
EOF
