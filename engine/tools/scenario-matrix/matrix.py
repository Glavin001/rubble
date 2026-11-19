#!/usr/bin/env python3
"""The scenario matrix (run through run.sh, which builds and takes the GPU lock).

Each row is a saved state of a use case (one soft vehicle driving, tumbling, into a wall;
piles of 2, 8 and 32; car against car; car into rigid bricks and under falling rocks; car
into a node-beam barrier; a loaded truss alone). A row replays its state with the switch
under test on and off (`vehicle_lab --from-fixture FILE --ab-switch SWITCH`: one process,
one state, the same build), compares every state word and times each frame apart.

    matrix.py --make [ROW ...]          save the states into out/scenario-matrix/
    matrix.py SWITCH [ROW ...]          the A/B table and the verdict
    matrix.py --baseline [ROW ...]      each row's ms a frame against its target
Options (environment): FRAMES (default 10) frames a half, ROUNDS (default 3) the halves in turn, PROFILE_SPLIT=1 (each row's
`--ab-profile`: the frame by dispatch kind; slower), THRESHOLD (% slower that fails,
default below), WARM_MS (default 150: the GPU warmed from the state before the timed halves), LAB_BIN,
CAR, STATES (the states' directory), RETRIES (default 2: a row past the threshold runs again,
the median run decides), BIT_EXACT (default 1: the rows run with bit-exact results, the
engine-wide setting that is off by default since 2026-11-19, so that the verdict is bit for
bit; 0 runs them in the fast setting).

The switch `bitexact` is bit-exact results themselves (on: strict arithmetic; off: the
fast setting). Its two sides are not bit for bit by design, so its rows are judged by
tolerance instead (TOL_MM, default 1000: the largest node distance between the sides
after the frames, a blow-up detector only: a crash amplifies last bits to centimetres and
a torn piece to decimetres within 10 frames; TOL_BROKEN, default 2, plus 1 % of the
broken beams: their difference; every position
finite), and a row fails as slower when the fast side (off) is slower than the strict one.
The tolerances come from the fast setting's own spread: vehicle-perf-first-principles.md
§8.38.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, "..", ".."))
SCENES = os.path.join(HERE, "scenes")
STATES = os.environ.get("STATES", os.path.join(ENGINE, "out", "scenario-matrix"))
BIN = os.environ.get("LAB_BIN", os.path.join(ENGINE, "target", "fast", "examples", "vehicle_lab"))
CAR = os.environ.get("CAR", os.path.expanduser("~/Development/rubble-artifacts/sedan-live/mods/sedan_new"))
FRAMES = int(os.environ.get("FRAMES", "10"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
# A row past the threshold runs RETRIES more times; the median run decides.
RETRIES = int(os.environ.get("RETRIES", "2"))
# A row is slower when its on-frame median is more than this much over its off-frame
# median, and more than NOISE_MS: calibrated from the same-vs-same spread (perf=0, the
# switch that changes nothing; scenario-matrix.md "Noise").
THRESHOLD = float(os.environ.get("THRESHOLD", "5"))
NOISE_MS = float(os.environ.get("NOISE_MS", "0.05"))
BIT_EXACT = os.environ.get("BIT_EXACT", "1")
TOL_MM = float(os.environ.get("TOL_MM", "1000"))
TOL_BROKEN = int(os.environ.get("TOL_BROKEN", "2"))
PHYSICAL = re.compile(r"A/B physical: largest node difference ([\d.]+) mm .*broken beams on (\d+) / off (\d+); "
                      r"kinetic energy on ([\d.]+) / off ([\d.]+) J; finite (true|false)")

# (row, soft bodies in it, scenario file or None for the mountain, extra lab args, the
# moment saved (s), what it is). The mountain rows are the lab's multi-car run (the
# fast-check fixtures' moments).
ROWS = [
    ("drive", 1, "drive.json", [], 1.0, "one car driving on flat ground, 60 km/h, throttle on"),
    ("tumble", 1, None, ["--cars", "1"], 2.5, "one car in free flight, tumbling down the mountain"),
    ("wall", 1, "wall.json", [], 0.1, "one car into a rigid wall at 56 km/h, mid-crush"),
    ("pile2", 2, None, ["--cars", "2"], 3.7, "2 cars, first collisions on the mountain"),
    ("pile8", 8, None, ["--cars", "8"], 8.0, "8 cars piling up on the mountain"),
    ("pile32", 32, None, ["--cars", "32"], 4.0, "32 cars piling up on the mountain"),
    ("headon", 2, "headon.json", ["--cars", "2"], 0.7, "2 cars head-on, 50 km/h each, mid-crush"),
    ("side", 2, "side.json", [], 1.0, "a car at 50 km/h into a parked car's side"),
    ("bricks", 1, "bricks.json", [], 0.7, "a car at 40 km/h into a wall of 60 rigid bricks"),
    ("rocks", 1, "rocks.json", [], 1.0, "4 rigid rocks (0.5 to 1 t) falling on a parked car"),
    ("barrier", 2, "barrier.json", [], 0.65, "a car at 50 km/h into a node-beam steel barrier"),
    ("truss", 1, "truss.json", ["--car", os.path.join(SCENES, "truss.truck")], 2.0,
     "a node-beam bridge truss sagging under 6 t, no vehicle"),
]


def target_ms(bodies):
    """The owner's targets: one soft body 1 ms a frame (worst), 8 bodies in contact 3 ms;
    between and past them a straight line through both (an extrapolation, marked ~)."""
    return 1.0 + (bodies - 1) * 2.0 / 7.0, bodies in (1, 8)


def lab_args(row):
    name, _, scenario, extra, _, _ = row
    args = [BIN, "--mode", "gpu", "--measure", "mountain-perf", "--rockslide", "0",
            "--self-split", "1", "--timing", "step", "--bit-exact", BIT_EXACT]
    if "--car" not in extra:
        args += ["--car", CAR]
    if "--cars" not in extra:
        args += ["--cars", "1"]
    if scenario:
        args += ["--scenario", os.path.join(SCENES, scenario)]
    return args + extra


def state_file(row):
    return os.path.join(STATES, row[0] + ".fx")


def pick(names):
    if not names:
        return ROWS
    rows = [r for r in ROWS if any(n == r[0] or n in r[0] for n in names)]
    if not rows:
        sys.exit(f"no row matches {names}; rows: {', '.join(r[0] for r in ROWS)}")
    return rows


def make(rows):
    os.makedirs(STATES, exist_ok=True)
    for row in rows:
        t0 = time.time()
        d = os.path.join(STATES, row[0] + ".d")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        at = row[4]
        cmd = lab_args(row) + ["--ab", str(at), "--ab-switch", "perf=0", "--ab-frames", "1",
                               "--save-fixtures", d, "--secs", f"{at + 0.1:.2f}"]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE)
        saved = [f for f in os.listdir(d) if f.endswith(".fx")]
        if p.returncode != 0 or len(saved) != 1:
            print(f"FAIL {row[0]}: exit {p.returncode}, {saved}")
            print("\n".join(p.stdout.splitlines()[-8:] + p.stderr.splitlines()[-8:]))
            continue
        os.replace(os.path.join(d, saved[0]), state_file(row))
        os.rmdir(d)
        print(f"saved {row[0]:<8} {saved[0]} ({os.path.getsize(state_file(row)) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


BEST = re.compile(r"frames \((on|off)\): best of \d+ rounds a frame, mean ([\d.]+) ms")
FRAME = re.compile(r"frames \((on|off)\): median ([\d.]+) ms, min ([\d.]+), max ([\d.]+)")


def replay(row, switch):
    cmd = lab_args(row) + ["--from-fixture", state_file(row), "--ab-switch", switch,
                           "--ab-frames", str(FRAMES)]
    if os.environ.get("PROFILE_SPLIT") == "1":
        cmd.append("--ab-profile")
    t0 = time.time()
    # (The GPU warmed for WARM_MS from the state first: a small row's first half otherwise
    # runs at the idle clock; scenario-matrix.md "Noise".)
    # (And the halves in turn ROUNDS times, on, off, on, off, ..., frames pooled by half.)
    env = dict(os.environ, AB_WARM_MS=os.environ.get("WARM_MS", "150"), AB_ROUNDS=str(ROUNDS))
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE, env=env)
    out = p.stdout + p.stderr
    med = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4))) for m in FRAME.finditer(out)}
    # The number compared: each frame's fastest round, their mean (the median of every
    # frame when there is one round).
    for m in BEST.finditer(out):
        if m.group(1) in med:
            med[m.group(1)] = (float(m.group(2)),) + med[m.group(1)][1:]
    verdict = re.search(r"A/B at t [\d.]+: .*state bit for bit: (true|false)", out)
    bfb = verdict is not None and verdict.group(1) == "true"
    phys = PHYSICAL.search(out)
    tol = None
    if phys:
        mm, b_on, b_off, finite = float(phys.group(1)), int(phys.group(2)), int(phys.group(3)), phys.group(6) == "true"
        tol = {"mm": mm, "broken": (b_on, b_off), "ke": (float(phys.group(4)), float(phys.group(5))),
               "pass": finite and mm <= TOL_MM and abs(b_on - b_off) <= TOL_BROKEN + b_on // 100}
    if switch == "bitexact":
        # (Not bit for bit by design: the tolerance decides.)
        bfb = tol is not None and tol["pass"]
    return {"on": med.get("on"), "off": med.get("off"), "bfb": bfb and p.returncode == 0,
            "secs": time.time() - t0, "out": out, "ok": verdict is not None, "tol": tol}


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--make":
        make(pick(args[1:]))
        return 0
    baseline = args[0] == "--baseline"
    switch = "perf=0" if baseline else args[0]
    rows = pick(args[1:])
    missing = [r[0] for r in rows if not os.path.isfile(state_file(r))]
    if missing:
        sys.exit(f"no saved state for {', '.join(missing)} in {STATES}: run.sh --make")
    t_all = time.time()
    bad = []
    if baseline:
        print(f"scenario matrix baseline ({ROUNDS} x {FRAMES} frames of 34 steps, each frame's best round, mean ms a frame; {os.path.basename(CAR)})")
        print(f"{'row':<8} {'bodies':>6} {'ms/frame':>9} {'target':>7} {'':>6} {'run s':>5}  what")
    else:
        if switch == "bitexact":
            print(f"(bitexact: 'on' is bit-exact, 'off' the fast setting; 'bit4bit' is the tolerance verdict: <= {TOL_MM:g} mm, <= {TOL_BROKEN} + 1 % broken beams' difference, finite; the scene-calibrated tolerance is tests/fast_arithmetic_gpu.rs)")
        print(f"scenario matrix: --ab-switch {switch} ({ROUNDS} x {FRAMES} frames of 34 steps each way in turn, each frame's best round, mean ms a frame; slower fails past {THRESHOLD:g} % and {NOISE_MS:g} ms)")
        print(f"{'row':<8} {'bit4bit':>7} {'on ms':>8} {'off ms':>8} {'delta':>8}  {'':<6} {'run s':>5}  what")
    for row in rows:
        r = replay(row, switch)
        if not r["ok"] or r["on"] is None or r["off"] is None:
            bad.append(row[0])
            print(f"{row[0]:<8} FAILED to run ({r['secs']:.1f} s)")
            print("\n".join("    " + l for l in r["out"].splitlines()[-10:]))
            continue
        on, off = r["on"][0], r["off"][0]
        if baseline:
            # (Both halves run the same path: the faster is the row's time.)
            ms = min(on, off)
            tgt, stated = target_ms(row[1])
            mark = "ok" if ms <= tgt else "over"
            print(f"{row[0]:<8} {row[1]:>6} {ms:>9.3f} {('' if stated else '~') + format(tgt, '.2f'):>7} {mark:>6} {r['secs']:>5.1f}  {row[5]}")
            continue
        delta = 100.0 * (on - off) / off
        past = lambda on, off: 100.0 * (on - off) / off > THRESHOLD and on - off > NOISE_MS
        if switch == "bitexact":
            # (The fast side is "off": it must not be slower than the strict one.)
            past = lambda on, off: 100.0 * (off - on) / on > THRESHOLD and off - on > NOISE_MS
        note = ""
        if r["bfb"] and past(on, off) and RETRIES > 0:
            # A row past the threshold is run again RETRIES times: the median run (by its
            # delta) decides. A lone slow run is the machine; a change is slow every time.
            runs = [(on, off, r["secs"])]
            for _ in range(RETRIES):
                q = replay(row, switch)
                if q["ok"] and q["on"] and q["off"] and q["bfb"]:
                    runs.append((q["on"][0], q["off"][0], q["secs"]))
            runs.sort(key=lambda x: x[0] / x[1])
            on, off, _ = runs[len(runs) // 2]
            delta = 100.0 * (on - off) / off
            note = f" ({len(runs)} runs: " + ", ".join(f"{100.0 * (a - b) / b:+.1f}%" for a, b, _ in runs) + ")"
            r["secs"] = sum(x[2] for x in runs)
        slower = past(on, off)
        flag = "SLOWER" if slower else ("faster" if delta < -THRESHOLD and off - on > NOISE_MS else "")
        if not r["bfb"] or slower:
            bad.append(row[0])
        if switch == "bitexact" and flag == "faster":
            flag = ""
        if switch == "bitexact" and delta > THRESHOLD and on - off > NOISE_MS:
            flag = "faster"
        print(f"{row[0]:<8} {('yes' if r['bfb'] else 'NO'):>7} {on:>8.3f} {off:>8.3f} {delta:>+7.1f}%  {flag:<6} {r['secs']:>5.1f}  {row[5]}{note}")
        if r.get("tol"):
            t = r["tol"]
            print(f"         tolerance: largest node difference {t['mm']:.3f} mm, broken beams {t['broken'][0]} / {t['broken'][1]}, kinetic energy {t['ke'][0]:.1f} / {t['ke'][1]:.1f} J")
        if not r["bfb"]:
            for l in r["out"].splitlines():
                if "differ" in l or "Error" in l or "panicked" in l:
                    print("    " + l.strip())
        if os.environ.get("PROFILE_SPLIT") == "1":
            for l in r["out"].splitlines():
                if re.search(r"profile at|^\s{5}\S.*\d+\.\d+%$|untimed frame|self split profile", l):
                    print("    " + l.rstrip())
    wall = time.time() - t_all
    if baseline:
        print(f"baseline: {len(rows)} rows in {wall:.1f} s" + (f"; failed: {', '.join(bad)}" if bad else ""))
        return 1 if bad else 0
    verdict = "PASS" if not bad else "FAIL (" + ", ".join(bad) + ")"
    print(f"verdict: {verdict}; {len(rows)} rows in {wall:.1f} s")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
