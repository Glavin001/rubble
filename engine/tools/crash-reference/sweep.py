#!/usr/bin/env python3
"""The crash reference suite swept: every scenario x {discrete, edge, continuous} x 8
jittered runs, on the CPU reference or on CUDA on Modal (engine/docs/crash-reference.md,
"Results").

    python3 tools/crash-reference/sweep.py --cpu  [--jobs 8] [--ids S01,S02] [--contacts edge]   (from engine/)
    python3 tools/crash-reference/sweep.py --cuda [--gpu T4] [--ids S01,S02] [--contacts edge] [--run-from 4 --runs 2]
    python3 tools/crash-reference/sweep.py --summarize          (the tables again, no runs)

Writes out/crash-reference/sweep/<backend>/<ID>-<contact>-<run>.json (each run's
CrashReport without its time histories), summary-<backend>.md (scenario x contact: each
asserted criterion pass / known failure (k of 8) / FAIL, headline metrics min-max) and,
when both backends are there, compare.md (CUDA against the CPU reference: each asserted
criterion's value and the headline metrics as min-max over the runs, and every scenario
where the two disagree on a verdict).

--cpu runs crash_ref (engine-softbody-lab) once per scenario and contact setting, --jobs at
a time, each bounded by a 300 s alarm. --cuda runs engine-cuda's crash_reference tests
through tools/modal/gpu_run.sh: one container per test (--fanout-tests), the three contact
settings as variants in each, built on Modal (--build modal), each run's report read back
from the log (CRASH_REPORT_LINES=1: files written in a container do not come back).

Pass/fail here reads what Rust judged (each result's pass, asserted, known); the verdict
that fails a run is crash_reference::verdict's, which crash_ref and the tests apply.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2]
OUT = ENGINE / "out" / "crash-reference" / "sweep"
CONTACTS = ["discrete", "edge", "continuous"]
RUNS = 8
MARK = "CRASH_REPORT_JSON "


def ids_all(crash_ref: Path) -> list[str]:
    out = subprocess.run([str(crash_ref), "--list"], capture_output=True, text=True, check=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.strip()]


def cpu(args) -> None:
    subprocess.run(["perl", "-e", "alarm 590; exec @ARGV", "cargo", "build", "--release", "-p", "engine-softbody-lab",
                    "--example", "crash_ref"], cwd=ENGINE, check=True, stdin=subprocess.DEVNULL)
    exe = ENGINE / "target" / "release" / "examples" / "crash_ref"
    ids = args.ids or ids_all(exe)
    d = OUT / "cpu"
    (d / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(job):
        sid, c, k = job
        log = d / "logs" / f"{sid}-{c}{'' if k is None else f'-{k}'}.log"
        runs = ["--runs", str(RUNS)] if k is None else ["--run", str(k)]
        t = time.time()
        with open(log, "w") as f:
            code = subprocess.run(["perl", "-e", "alarm 300; exec @ARGV", str(exe), sid, *runs, "--contact", c,
                                   "--out", str(d)], cwd=ENGINE, stdout=f, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL).returncode
        return sid, c, k, code, time.time() - t

    # --per-run: one process a run (scenarios whose 8 runs take longer than the bound); its
    # exit code is then one run's verdict, and the summary's is the whole setting's.
    jobs = [(s, c, k) for c in args.contacts for s in ids for k in (range(RUNS) if args.per_run else [None])]
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        for sid, c, k, code, secs in ex.map(one, jobs):
            print(f"cpu {sid} {c}{'' if k is None else f' run {k}'}: exit {code}, {secs:.0f} s", flush=True)
    with open(d / "wall.txt", "a") as f:
        f.write(f"{','.join(args.contacts)}: {time.time() - t0:.0f} s wall, {len(jobs)} jobs, {args.jobs} at a time\n")


def cuda(args) -> None:
    d = OUT / "cuda"
    d.mkdir(parents=True, exist_ok=True)
    cmd = ["perl", "-e", "alarm 590; exec @ARGV", "tools/modal/gpu_run.sh", "--build", args.build, "--fanout-tests",
           "--gpu", args.gpu, "--cpus", str(args.cpus), "--env", "CRASH_REPORT_LINES=1", "--env", f"CRASH_RUNS={args.runs}", "--env", f"CRASH_RUN_FROM={args.run_from}"]
    for c in args.contacts:
        cmd += ["--variant", f"{c}:CRASH_CONTACT={c}"]
    cmd += ["--", "-p", "engine-cuda", "--test", "crash_reference"]
    if args.ids:
        # (libtest filters by name: the scenarios' tests are sNN.)
        cmd += ["--"] + [s.lower() for s in args.ids]
    t0 = time.time()
    tag = time.strftime("%H%M%S")
    ids = ",".join(args.ids or ["all"])
    log = d / f"modal-{','.join(args.contacts)}-{ids}-{args.run_from}-{tag}.log"
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=ENGINE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             stdin=subprocess.DEVNULL)
        for line in p.stdout:
            f.write(line)
            if MARK not in line and PART not in line:
                sys.stdout.write(line)
        code = p.wait()
    n = read_log(log, d)
    print(f"cuda: exit {code}, {time.time() - t0:.0f} s wall, {n} reports read back", flush=True)
    with open(d / "wall.txt", "a") as f:
        f.write(f"{log.name}: {time.time() - t0:.0f} s wall (gpu_run.sh exit {code}), {n} reports\n")


PART = "CRASH_REPORT_PART "


def read_log(log: Path, d: Path) -> int:
    """Reports from a log: numbered parts (`CRASH_REPORT_PART key k/n text`), or whole lines
    (`CRASH_REPORT_JSON text`). A report with a part missing or cut is skipped and said."""
    n = 0
    parts: dict = {}
    whole = []
    # (A fan-out's containers share one stream: a line can run on into another container's,
    # which starts with its label.)
    lines = [x for line in log.read_text(errors="replace").splitlines() for x in re.split(r"(?=\[crash_reference::)", line)]
    for line in lines:
        i = line.find(PART)
        if i >= 0:
            key, kn, text = (line[i + len(PART):].rstrip("\n").split(" ", 2) + ["", ""])[:3]
            k, _, total = kn.partition("/")
            parts.setdefault(key, {})[int(k)] = (int(total), text)
            continue
        i = line.find(MARK)
        if i >= 0:
            whole.append(line[i + len(MARK):])
    for key, ps in parts.items():
        total = next(iter(ps.values()))[0]
        if sorted(ps) != list(range(1, total + 1)):
            print(f"skipped {key}: {len(ps)} of {total} parts", file=sys.stderr)
            continue
        whole.append("".join(ps[k][1] for k in range(1, total + 1)))
    for text in whole:
        try:
            r = json.loads(text)
        except json.JSONDecodeError:
            print(f"skipped a cut report ({len(text)} chars)", file=sys.stderr)
            continue
        (d / f"{r['id']}-{r['contact']}-{r['run']}.json").write_text(json.dumps(r))
        n += 1
    return n


# ---- tables ----

def load(backend: str) -> dict:
    """{(id, contact): [report, ...] by run}"""
    out: dict = {}
    for f in sorted((OUT / backend).glob("*-*-*.json")):
        r = json.loads(f.read_text())
        out.setdefault((r["id"], r["contact"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["run"])
    return out


def headline_keys(metrics: dict) -> list[str]:
    body = next((b for b in ("car", "car_a") if f"{b}.crush_peak" in metrics), None)
    keys = ["all.energy_gain_j", "all.plastic_work_j"]
    if body:
        keys += [f"{body}.crush_peak", f"{body}.rebound_ratio", f"{body}.speed_end"]
    keys += sorted(k for k in metrics if k.startswith("pair.") and k.endswith(".crossings_max"))
    keys += [k for k in ("all.continuous_engaged",) if k in metrics]
    return keys


def fmt(x) -> str:
    if x is None:
        return "-"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


def span(vals) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "not measured"
    lo, hi = min(vals), max(vals)
    return fmt(lo) if lo == hi else f"{fmt(lo)} to {fmt(hi)}"


def verdicts(reports: list) -> dict:
    """{criterion name: (status text, verdict)} for asserted criteria; verdict is
    pass / known / FAIL (an unknown failure, or an every-run known failure that passed)."""
    out = {}
    n = len(reports)
    for i, c in enumerate(reports[0]["results"]):
        if not c["asserted"]:
            continue
        fails = sum(1 for r in reports if not r["results"][i]["pass"])
        known = c.get("known", False)
        occurs = c.get("known_occurs")
        if fails == 0 and known and occurs == "every_run":
            out[c["name"]] = (f"FAIL: known failure passes in all {n} (remove the entry)", "FAIL")
        elif fails == 0:
            out[c["name"]] = ("pass" + (" (known intermittent, did not occur)" if known else ""), "pass")
        elif known:
            out[c["name"]] = (f"known failure ({fails} of {n})", "known")
        else:
            out[c["name"]] = (f"FAIL ({fails} of {n})", "FAIL")
    return out


def summary(backend: str) -> str:
    data = load(backend)
    if not data:
        return ""
    lines = [f"### {backend.upper()}: every scenario x contact setting, {RUNS} jittered runs", "",
             "| Scenario | Contact | Runs | Asserted criteria | Headline metrics (min to max over the runs) |",
             "|---|---|---|---|---|"]
    for (sid, c), reps in sorted(data.items(), key=lambda kv: (kv[0][0], CONTACTS.index(kv[0][1]))):
        v = verdicts(reps)
        bad = [f"**{k}: {t}**" for k, (t, s) in v.items() if s == "FAIL"]
        known = [f"{k}: {t}" for k, (t, s) in v.items() if s == "known"]
        npass = sum(1 for _, s in v.values() if s == "pass")
        crit = "; ".join(bad + known + [f"{npass} pass"])
        keys = headline_keys(reps[0]["metrics"])
        head = "; ".join(f"{k} {span([r['metrics'].get(k) for r in reps])}" for k in keys)
        lines.append(f"| {sid} | {c} | {len(reps)} | {crit} | {head} |")
    return "\n".join(lines) + "\n"


def compare() -> str:
    a, b = load("cpu"), load("cuda")
    if not a or not b:
        return ""
    lines = ["### CUDA against the CPU reference (min to max over the runs: CPU / CUDA)", "",
             "| Scenario | Contact | Verdicts | Asserted criteria's values (CPU / CUDA) | Headline metrics (CPU / CUDA) |",
             "|---|---|---|---|---|"]
    disagree = []
    for key in sorted(set(a) | set(b), key=lambda k: (k[0], CONTACTS.index(k[1]))):
        sid, c = key
        if key not in a or key not in b:
            lines.append(f"| {sid} | {c} | only on {'CPU' if key in a else 'CUDA'} | | |")
            continue
        ra, rb = a[key], b[key]
        va, vb = verdicts(ra), verdicts(rb)
        diff = [k for k in va if vb.get(k, ("", "?"))[1] != va[k][1]]
        for k in diff:
            disagree.append(f"- {sid} {c}, {k}: CPU {va[k][0]}, CUDA {vb.get(k, ('not judged', ''))[0]}")
        ver = "agree" if not diff else f"**disagree on {len(diff)}**"
        crit = []
        for i, cr in enumerate(ra[0]["results"]):
            if not cr["asserted"]:
                continue
            x = span([r["results"][i]["value"] for r in ra])
            y = span([r["results"][i]["value"] for r in rb if i < len(r["results"])])
            mark = "**" if cr["name"] in diff else ""
            crit.append(f"{mark}{cr['name']}: {x} / {y}{mark}")
        keys = headline_keys(ra[0]["metrics"])
        head = "; ".join(f"{k} {span([r['metrics'].get(k) for r in ra])} / {span([r['metrics'].get(k) for r in rb])}" for k in keys)
        lines.append(f"| {sid} | {c} | {ver} | {'; '.join(crit)} | {head} |")
    lines += ["", "Verdict disagreements:", ""] + (disagree or ["- none"])
    return "\n".join(lines) + "\n"


def known_table(backend: str = "cpu") -> str:
    """The recorded known failures (each result's known, known_occurs, known_note) beside what
    the sweep measured: the doc's "Currently failing" table."""
    data = load(backend)
    if not data:
        return ""
    lines = [f"| Scenario | Criterion | Contact | Recorded as | Failed ({backend.upper()}) | Value over the runs | Band | New, for review |",
             "|---|---|---|---|---|---|---|---|"]
    for (sid, c), reps in sorted(data.items(), key=lambda kv: (kv[0][0], CONTACTS.index(kv[0][1]))):
        for i, r0 in enumerate(reps[0]["results"]):
            if not (r0["asserted"] and r0.get("known")):
                continue
            fails = sum(1 for r in reps if not r["results"][i]["pass"])
            band = f"{fmt(r0['lo']) if r0['lo'] is not None else ''} .. {fmt(r0['hi']) if r0['hi'] is not None else ''}"
            occurs = {"every_run": "every run", "some_runs": "some runs"}.get(r0.get("known_occurs"), "?")
            vals = span([r["results"][i]["value"] for r in reps])
            note = "**yes**" if "new in the" in (r0.get("known_note") or "") else ""
            lines.append(f"| {sid} | {r0['name']} | {c} | {occurs} | {fails} of {len(reps)} | {vals} | {band} | {note} |")
    return "\n".join(lines) + "\n"


def write_tables() -> None:
    t = known_table()
    if t:
        (OUT / "known-cpu.md").write_text(t)
        print(f"wrote {OUT / 'known-cpu.md'}")
    for b in ("cpu", "cuda"):
        t = summary(b)
        if t:
            (OUT / f"summary-{b}.md").write_text(t)
            print(f"wrote {OUT / f'summary-{b}.md'}")
    t = compare()
    if t:
        (OUT / "compare.md").write_text(t)
        print(f"wrote {OUT / 'compare.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cpu", action="store_true")
    g.add_argument("--cuda", action="store_true")
    g.add_argument("--summarize", action="store_true")
    g.add_argument("--read-log", metavar="LOG", help="read a --cuda run's log back into reports")
    ap.add_argument("--ids", type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()])
    ap.add_argument("--contacts", type=lambda s: s.split(","), default=CONTACTS, help="the settings to run (default all three; --cuda: one variant each)")
    ap.add_argument("--run-from", type=int, default=0, help="--cuda: the first run (CRASH_RUN_FROM)")
    ap.add_argument("--runs", type=int, default=RUNS, help="--cuda: runs a container (CRASH_RUNS), from --run-from")
    ap.add_argument("--per-run", action="store_true", help="--cpu: one process a run")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--build", default="modal", choices=["modal", "local"], help="--cuda: where gpu_run.sh compiles")
    ap.add_argument("--cpus", default="2", help="--cuda: CPU cores a container (the templates rig on the CPU)")
    args = ap.parse_args()
    if args.cpu:
        cpu(args)
    elif args.cuda:
        cuda(args)
    elif args.read_log:
        print(read_log(Path(args.read_log), OUT / "cuda"), "reports")
    write_tables()


if __name__ == "__main__":
    main()
