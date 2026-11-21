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
    matrix.py --world [ROW ...]         the World's own cost a step (the rigid step, the
        coupling's passes, the host): each row's scene run in the World up to its moment,
        then `--end-world` (N World steps, then the same explicit steps as frames alone from
        the same state); fails a row where the World adds more than WORLD_MS. The rows with
        a scenario file by default (the mountain rows drive for seconds first: name them).
    matrix.py --rows ARGS...            the row names ARGS pick (for the Modal runner)
    matrix.py --make-portable [ROW ...] each row's saved state as a portable fixture
        (out/scenario-matrix/portable/ROW.pfx) for the CUDA replay (REPLAY_BIN=explicit_replay
        runs the rows on it: tools/modal/gpu_run.sh --matrix-cuda -- SWITCH [ROW ...])
    matrix.py --two-builds [ROW ...]    two builds against each other on CUDA (BASE_BIN and
        REPLAY_BIN, explicit_replay --run): each row's fixture replayed by one build's
        process, then the other's, PROCS (default 3) times in alternating order; the median
        process each, the base's spread as the noise, the end states word by word (a changed
        layout: the bindings both have). ALLOW_DIFF=1 reports state differences without
        failing (a change that reorders arithmetic). run.sh --cuda --base REF [--head REF].
    matrix.py --end-state DIR [ROW ...] each row's state after END_FRAMES (default 20)
        frames from its moment, into DIR/ROW.fx (one build's; a host refactor's gate)
    matrix.py --compare-end A B [ROW ...]  two builds' end states, every state word:
        PASS only if each row is bit for bit (a host change the A/B cannot see, since
        it changes both halves alike)
Options (environment): FRAMES (default 10) frames a half, ROUNDS (default 3) the halves in turn, PROFILE_SPLIT=1 (each row's
`--ab-profile`: the frame by dispatch kind; slower), THRESHOLD (% slower that fails,
default below), WARM_MS (default 150: the GPU warmed from the state before the timed halves), LAB_BIN,
CAR, STATES (the states' directory), RETRIES (default 2: a row past the threshold runs again,
the median run decides). --world: WORLD_STEPS (default 20) steps each way, WORLD_MS (default
2.5) the most the World may add a step, WORLD_ARGS extra lab arguments.
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
# CUDA: engine-cuda's explicit_replay replays each row from its portable fixture
# (STATES/portable/ROW.pfx, `--make-portable`) in place of vehicle_lab (gpu_run.sh
# --matrix-cuda sets it on Modal).
REPLAY = os.environ.get("REPLAY_BIN")
# Two builds (`--two-builds`): the base's explicit_replay, compared with REPLAY_BIN (the
# head) process by process (gpu_run.sh --base REF builds and ships both).
BASE = os.environ.get("BASE_BIN")
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
            "--self-split", "1", "--timing", "step"]
    if "--car" not in extra:
        args += ["--car", CAR]
    if "--cars" not in extra:
        args += ["--cars", "1"]
    if scenario:
        args += ["--scenario", os.path.join(SCENES, scenario)]
    return args + extra


def state_file(row):
    return os.path.join(STATES, row[0] + ".fx")


def portable_file(row):
    return os.path.join(STATES, "portable", row[0] + ".pfx")


def make_portable(rows):
    """Each row's saved state as a portable fixture (every binding, the layout, the
    planner, the host's paths), for a host with no World: the CUDA replay."""
    os.makedirs(os.path.join(STATES, "portable"), exist_ok=True)
    bad = []
    for row in rows:
        cmd = lab_args(row) + ["--from-fixture", state_file(row), "--save-portable", portable_file(row)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE)
        line = next((l for l in p.stdout.splitlines() if l.startswith("portable ")), None)
        if p.returncode != 0 or line is None:
            bad.append(row[0])
            print(f"FAIL {row[0]}: exit {p.returncode}")
            print("\n".join((p.stdout + p.stderr).splitlines()[-8:]))
            continue
        print(f"{row[0]:<8} {line}")
    return 0 if not bad else 1


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
    if REPLAY:
        cmd = [REPLAY, "--fixture", portable_file(row), "--ab-switch", switch, "--ab-frames", str(FRAMES)]
    else:
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
    return {"on": med.get("on"), "off": med.get("off"), "bfb": bfb and p.returncode == 0,
            "secs": time.time() - t0, "out": out, "ok": verdict is not None}


RUN = re.compile(r"run at t [\d.]+: ([\d.]+) ms/frame")


def read_state(path):
    """An end state `explicit_replay --run --save-state` wrote: [(name, words bytes)]."""
    out, b, i = [], open(path, "rb").read(), 0
    while i < len(b):
        j = b.index(0, i)
        name = b[i:j].decode()
        n = int.from_bytes(b[j + 1:j + 5], "little")
        out.append((name, b[j + 5:j + 5 + 4 * n]))
        i = j + 5 + 4 * n
    return out


def compare_states(a, b):
    """(bit for bit, note). The same layout: every word. A changed layout (an architecture
    change): the bindings both builds have by name and size, and which they are."""
    if [(n, len(w)) for n, w in a] == [(n, len(w)) for n, w in b]:
        bad = [n for (n, x), (_, y) in zip(a, b) if x != y]
        return not bad, ("differ: " + ", ".join(bad)) if bad else ""
    bmap = dict(b)
    common = [(n, w) for n, w in a if n in bmap and len(bmap[n]) == len(w)]
    same = [n for n, w in common if bmap[n] == w]
    return False, (f"layout differs: {len(common)} of {len(a)} bindings comparable, "
                   f"{len(same)} equal ({', '.join(n for n, _ in common if n not in same) or 'none differ'})")


def run_build(binary, row, state_out):
    """One process of one build on the row's fixture: (ms a frame, end state path or
    None when that build predates `--run`: then its no-op A/B's mean, no state)."""
    cmd = [binary, "--fixture", portable_file(row), "--run", "--ab-frames", str(FRAMES), "--save-state", state_out]
    env = dict(os.environ, AB_WARM_MS=os.environ.get("WARM_MS", "150"), AB_ROUNDS=str(ROUNDS))
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE, env=env)
    out = p.stdout + p.stderr
    m = RUN.search(out)
    if m and p.returncode == 0:
        return float(m.group(1)), state_out, out
    best = [float(x.group(2)) for x in BEST.finditer(out)]
    if best:
        return sum(best) / len(best), None, out
    return None, None, out


def two_builds(rows):
    """The A/B of two builds (BASE_BIN against REPLAY_BIN; gpu_run.sh --base REF): each row's
    fixture replayed by one build's process, then the other's, PROCS times, the order
    alternating (A B, B A, ...). A build's time is the median of its processes; the base's
    own spread between processes is printed as that row's noise. The end states: every
    word, base against head; each build against itself too (a nondeterministic build is
    no reference)."""
    import tempfile
    procs = int(os.environ.get("PROCS", "3"))
    allow_diff = os.environ.get("ALLOW_DIFF") == "1"
    tmp = tempfile.mkdtemp(prefix="two-builds-")
    print(f"scenario matrix: two builds, base against head ({procs} processes each, alternating; {ROUNDS} x {FRAMES} "
          f"frames of 34 steps a process, each frame's best round, mean ms a frame; the median process; slower fails "
          f"past {THRESHOLD:g} % and {NOISE_MS:g} ms{'; state differences allowed' if allow_diff else ''})")
    print(f"{'row':<8} {'bit4bit':>7} {'head ms':>8} {'base ms':>8} {'delta':>8}  {'':<6} {'noise':>6} {'run s':>5}  what")
    t_all = time.time()
    bad = []
    for row in rows:
        t0 = time.time()
        times = {"base": [], "head": []}
        states = {"base": [], "head": []}
        outs = []
        for i in range(procs):
            order = [("base", BASE), ("head", REPLAY)] if i % 2 == 0 else [("head", REPLAY), ("base", BASE)]
            for side, binary in order:
                ms, st, out = run_build(binary, row, os.path.join(tmp, f"{row[0]}.{side}{i}.st"))
                outs.append(f"[{side} #{i}] " + out[-600:])
                if ms is not None:
                    times[side].append(ms)
                if st:
                    states[side].append(read_state(st))
        secs = time.time() - t0
        if not times["base"] or not times["head"]:
            bad.append(row[0])
            print(f"{row[0]:<8} FAILED to run ({secs:.1f} s)")
            print("\n".join("    " + l for l in "\n".join(outs).splitlines()[-12:]))
            continue
        med = lambda v: sorted(v)[len(v) // 2]
        head, base = med(times["head"]), med(times["base"])
        noise = 100.0 * (max(times["base"]) - min(times["base"])) / base
        delta = 100.0 * (head - base) / base
        notes = []
        if states["base"] and states["head"]:
            same, why = compare_states(states["base"][0], states["head"][0])
            for side in ("base", "head"):
                if any(s != states[side][0] for s in states[side][1:]):
                    same, why = False, why + f"; {side} differs between its own processes"
            bfb = "yes" if same else "NO"
            if why:
                notes.append(why)
        else:
            same, bfb = True, "n/a"
            notes.append("no end state: the base predates explicit_replay --run")
        slower = delta > THRESHOLD and head - base > NOISE_MS
        flag = "SLOWER" if slower else ("faster" if delta < -THRESHOLD and base - head > NOISE_MS else "")
        if slower or (not same and not allow_diff):
            bad.append(row[0])
        print(f"{row[0]:<8} {bfb:>7} {head:>8.3f} {base:>8.3f} {delta:>+7.1f}%  {flag:<6} {noise:>5.1f}% {secs:>5.1f}  {row[5]}")
        for n in notes:
            print("    " + n)
    verdict = "PASS" if not bad else "FAIL (" + ", ".join(bad) + ")"
    print(f"verdict: {verdict}; {len(rows)} rows in {time.time() - t_all:.1f} s")
    return 0 if not bad else 1


WORLD = re.compile(r"end world: \d+ steps, World step ([\d.]+) ms, frame alone ([\d.]+) ms, the World adds (-?[\d.]+) ms")


def world(rows):
    """The World's own cost a step on each row (`vehicle_lab --end-world`)."""
    n = os.environ.get("WORLD_STEPS", "20")
    limit = float(os.environ.get("WORLD_MS", "2.5"))
    extra = os.environ.get("WORLD_ARGS", "").split()
    print(f"scenario matrix, the World's own cost: each row's scene to its moment, then 3 rounds of {n} World steps "
          f"and the same explicit steps as frames alone (medians, wall ms); fails past {limit:g} ms")
    print(f"{'row':<8} {'step ms':>8} {'frame ms':>9} {'World ms':>9} {'':<6} {'run s':>5}  what")
    t_all = time.time()
    bad = []
    for row in rows:
        t0 = time.time()
        cmd = lab_args(row)
        # (No GPU timestamps: wall time only, as a page waits for it.)
        i = cmd.index("--timing")
        del cmd[i:i + 2]
        cmd += ["--secs", f"{row[4]:.2f}", "--end-world", n] + extra
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE)
        out = p.stdout + p.stderr
        m = WORLD.search(out)
        secs = time.time() - t0
        if p.returncode != 0 or not m:
            bad.append(row[0])
            print(f"{row[0]:<8} FAILED to run ({secs:.1f} s)")
            print("\n".join("    " + l for l in out.splitlines()[-10:]))
            continue
        step, frame, adds = (float(m.group(k)) for k in (1, 2, 3))
        over = adds > limit
        if over:
            bad.append(row[0])
        print(f"{row[0]:<8} {step:>8.3f} {frame:>9.3f} {adds:>+9.3f} {'OVER' if over else '':<6} {secs:>5.1f}  {row[5]}")
    verdict = "PASS" if not bad else "FAIL (" + ", ".join(bad) + ")"
    print(f"verdict: {verdict}; {len(rows)} rows in {time.time() - t_all:.1f} s")
    return 0 if not bad else 1


def end_state(out_dir, rows):
    """Each row's state after END_FRAMES (default 20) frames of 34 steps from its saved
    moment, saved as OUT_DIR/ROW.fx: one build's end states, for --compare-end against
    another build's (a host change that changes both halves of the A/B alike, which the
    A/B cannot see)."""
    n = os.environ.get("END_FRAMES", "20")
    os.makedirs(out_dir, exist_ok=True)
    bad = []
    for row in rows:
        t0 = time.time()
        d = os.path.join(out_dir, row[0] + ".d")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        cmd = lab_args(row) + ["--from-fixture", state_file(row), "--ab-switch", "perf=0",
                               "--ab-frames", "1", "--save-fixtures", d]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE, env=dict(os.environ, AB_PRE=n))
        saved = [f for f in os.listdir(d) if f.endswith(".fx")]
        if p.returncode != 0 or len(saved) != 1:
            bad.append(row[0])
            print(f"FAIL {row[0]}: exit {p.returncode}, {saved}")
            print("\n".join((p.stdout + p.stderr).splitlines()[-8:]))
            continue
        os.replace(os.path.join(d, saved[0]), os.path.join(out_dir, row[0] + ".fx"))
        os.rmdir(d)
        print(f"end state {row[0]:<8} after {n} frames ({time.time() - t0:.1f} s)")
    return 0 if not bad else 1


def compare_end(a_dir, b_dir, rows):
    """Every state word of each row's end state in A_DIR against B_DIR's (the A/B's
    masking), and the host state (piece labels, retired nodes) as bytes."""
    bad = []
    for row in rows:
        a, b = os.path.join(a_dir, row[0] + ".fx"), os.path.join(b_dir, row[0] + ".fx")
        if not (os.path.isfile(a) and os.path.isfile(b)):
            bad.append(row[0])
            print(f"{row[0]:<8} MISSING ({a if not os.path.isfile(a) else b})")
            continue
        cmd = lab_args(row) + ["--from-fixture", a, "--compare-fixture", b]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ENGINE)
        out = p.stdout + p.stderr
        host = re.search(r"host state the same: (true|false)", out)
        same = p.returncode == 0 and "state bit for bit: true" in out and host is not None and host.group(1) == "true"
        if not same:
            bad.append(row[0])
        print(f"{row[0]:<8} {'bit for bit' if same else 'DIFFERS'}")
        if not same:
            for l in out.splitlines():
                if "differ" in l or "Error" in l or "panicked" in l or "host state" in l:
                    print("    " + l.strip())
    print(f"verdict: {'PASS' if not bad else 'FAIL (' + ', '.join(bad) + ')'}; {len(rows)} rows")
    return 0 if not bad else 1


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--make-portable":
        return make_portable(pick(args[1:]))
    if args[0] == "--end-state" and len(args) >= 2:
        return end_state(args[1], pick(args[2:]))
    if args[0] == "--compare-end" and len(args) >= 3:
        return compare_end(args[1], args[2], pick(args[3:]))
    if args[0] == "--rows":
        # The rows the rest of the arguments pick (SWITCH | --baseline | --make, then
        # names): the Modal runner gives each its own container.
        if len(args) < 2:
            sys.exit("matrix.py --rows SWITCH|--baseline|--make [ROW ...]")
        print(" ".join(r[0] for r in pick(args[2:])))
        return 0
    if args[0] == "--make":
        make(pick(args[1:]))
        return 0
    if args[0] == "--world":
        return world(pick(args[1:]) if args[1:] else [r for r in ROWS if r[2]])
    if args[0] == "--two-builds":
        if not (REPLAY and BASE):
            sys.exit("--two-builds needs REPLAY_BIN (head) and BASE_BIN (base): "
                     "run.sh --cuda --base REF [--head REF] [ROW ...]")
        rows = pick(args[1:])
        missing = [r[0] for r in rows if not os.path.isfile(portable_file(r))]
        if missing:
            sys.exit(f"no portable fixture for {', '.join(missing)} in {STATES}: matrix.py --make-portable")
        return two_builds(rows)
    baseline = args[0] == "--baseline"
    switch = "perf=0" if baseline else args[0]
    rows = pick(args[1:])
    missing = [r[0] for r in rows if not os.path.isfile(portable_file(r) if REPLAY else state_file(r))]
    if missing:
        sys.exit(f"no saved state for {', '.join(missing)} in {STATES}: run.sh --make"
                 + (" then matrix.py --make-portable" if REPLAY else ""))
    t_all = time.time()
    bad = []
    if baseline:
        print(f"scenario matrix baseline ({ROUNDS} x {FRAMES} frames of 34 steps, each frame's best round, mean ms a frame; {os.path.basename(CAR)})")
        print(f"{'row':<8} {'bodies':>6} {'ms/frame':>9} {'target':>7} {'':>6} {'run s':>5}  what")
    else:
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
        print(f"{row[0]:<8} {('yes' if r['bfb'] else 'NO'):>7} {on:>8.3f} {off:>8.3f} {delta:>+7.1f}%  {flag:<6} {r['secs']:>5.1f}  {row[5]}{note}")
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
