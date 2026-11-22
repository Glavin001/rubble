"""The main body's spin ledger from `door_orbit`'s spin.csv (engine-world-lab/examples/door_orbit.rs):
for every span of at least 1/3 s with the body's centre-of-mass acceleration less gravity
under THRESHOLD m/s² (no contact beyond what the air gives) and no piece leaving, the change
of its angular momentum about its centre of mass against the air's torque integrated (the
solver's per-triangle law, evaluated from the GPU's state each World step). Residual = what
else turned it (contact the threshold let through, or a bug).

    python3 tools/free-flight/spin_ledger.py out/free-flight/seed1/spin.csv [THRESHOLD]
"""
import csv
import math
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
f = lambda r, k: float(r[k])
V = lambda r, p: [f(r, p + "x"), f(r, p + "y"), f(r, p + "z")]
n = lambda v: math.sqrt(sum(x * x for x in v))
spans, cur = [], []
for a, b in zip(rows, rows[1:]):
    if f(b, "accel_less_g") < thr and a["nodes"] == b["nodes"]:
        cur.append((a, b))
    else:
        if len(cur) >= 20:
            spans.append(cur)
        cur = []
if len(cur) >= 20:
    spans.append(cur)
print(f"{sys.argv[1]}: {len(spans)} spans of 1/3 s or more with the body's |a - g| < {thr} m/s²")
for s in spans:
    a0, b1 = s[0][0], s[-1][1]
    l0, l1 = V(a0, "l"), V(b1, "l")
    T = f(b1, "t") - f(a0, "t")
    tq = [0.0, 0.0, 0.0]
    for a, b in s:
        dt = f(b, "t") - f(a, "t")
        for i, k in enumerate("xyz"):
            tq[i] += 0.5 * (f(a, "aero_t" + k) + f(b, "aero_t" + k)) * dt
    dl = [l1[i] - l0[i] for i in range(3)]
    res = [dl[i] - tq[i] for i in range(3)]
    worst = max(f(b, "accel_less_g") for _, b in s)
    print(
        f"  t {f(a0, 't'):6.2f}-{f(b1, 't'):6.2f}  |L| {n(l0):8.0f} -> {n(l1):8.0f} kg m²/s ({100 * (n(l1) / n(l0) - 1) / T:+6.1f} %/s)"
        f"  |dL| {n(dl):7.1f}  air {n(tq):7.1f}  residual {n(res):7.1f} ({100 * n(res) / max(n(l0), 1e-9) / T:5.2f} % of |L| a second)  max |a-g| {worst:4.1f}"
    )
