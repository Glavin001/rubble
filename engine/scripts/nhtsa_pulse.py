# NHTSA 56 km/h NCAP frontal: rear-seat crossmember X acceleration (G's, global X)
# -> dynamic crush, time of max crush, rebound velocity, restitution, peak (CFC60-like) g.
# Velocity and displacement integrate the raw signal (filtering does not change the
# integral); the peak uses a 100 Hz 2-pole Butterworth run forward and backward.
# python3 -I engine/scripts/nhtsa_pulse.py <dir of NHTSA <test>tsv.093/.094 files>
# The files are public-domain NHTSA data, fetched locally (never committed); see
# engine/docs/crash-reference-targets.md section 8.
import sys, os, math, statistics as st
d = sys.argv[1]
V0 = 56.3 / 3.6
G = 9.80665
def load(p):
    t, a = [], []
    for line in open(p):
        s = line.split()
        if len(s) >= 2:
            t.append(float(s[0])); a.append(float(s[1]))
    return t, a
def biquad(x, dt, fc):
    w = math.tan(math.pi * fc * dt); n = 1 / (1 + math.sqrt(2) * w + w * w)
    b0 = w * w * n; b1 = 2 * b0; b2 = b0; a1 = 2 * (w * w - 1) * n; a2 = (1 - math.sqrt(2) * w + w * w) * n
    y = []; x1 = x2 = x[0]; y1 = y2 = x[0]
    for v in x:
        o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, v, y1, o; y.append(o)
    return y
def metrics(p):
    t, a = load(p); dt = t[1] - t[0]
    pre = [a[i] for i in range(len(t)) if t[i] < 0]
    base = sum(pre) / len(pre) if pre else 0.0
    a = [x - base for x in a]
    i0 = next(i for i in range(len(t)) if t[i] >= 0)
    dv = sum(a[i0:i0 + int(0.25 / dt)]) * dt * G
    s = -1.0 if dv > 0 else 1.0           # deceleration as negative
    a = [s * x for x in a]
    f = biquad(biquad(a, dt, 100)[::-1], dt, 100)[::-1]
    v = V0; x = 0.0; xmax = 0.0; tm = 0.0; vmin = V0
    for i in range(i0, len(t)):
        v += a[i] * G * dt; x += v * dt
        if x > xmax: xmax, tm = x, t[i]
        if t[i] < 0.3: vmin = min(vmin, v)
    peak = -min(f[i0:i0 + int(0.2 / dt)])
    return xmax, tm, -vmin, peak
rows = []
for fn in sorted(os.listdir(d)):
    if not fn.endswith('.093'): continue
    ms = [metrics(os.path.join(d, fn[:-4] + c)) for c in ('.093', '.094') if os.path.exists(os.path.join(d, fn[:-4] + c))]
    rows.append((fn[:6], *[sum(m[k] for m in ms) / len(ms) for k in range(4)]))
print(f"{'test':8} {'dyn crush m':>11} {'t_m ms':>7} {'rebound m/s':>11} {'e':>6} {'peak g':>7} {'mean g':>7}")
for test, xm, tm, vr, pk in rows:
    print(f"{test:8} {xm:11.3f} {tm*1000:7.1f} {vr:11.2f} {vr/V0:6.3f} {pk:7.1f} {V0/(G*tm):7.1f}")
for k, name in ((1, 'dyn crush m'), (2, 't_m s'), (3, 'rebound m/s'), (4, 'peak g')):
    vals = [r[k] for r in rows]
    print(f"{name:12} median {st.median(vals):.3f}  min {min(vals):.3f}  max {max(vals):.3f}")
print('restitution median', round(st.median([r[3] / V0 for r in rows]), 3))
