"""Measured generation throughput and the projected bulk budget (plan P1 "Budget").

    python -m learned_impact.outcome_budget RUN_DIR [RUN_DIR ...] [--target 30000] [--ensemble-share 0.15]
        [--members 8] [--threads 8] [--json OUT.json]

Per job: wall-clock span of the job (`progress.log` start to end), events, threads (from the
start line), events/min, and the mean single-thread cost per event. The projection uses the
measured events/min per thread of each family (no scaling assumption beyond the threads the
job ran with is made silently: `--threads` rescales linearly and is labelled as such).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .outcome_audit import event_dirs, load_event

TIME = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]")


def _secs(line):
    m = TIME.match(line)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) if m else None


def job_throughput(run: Path) -> dict:
    lines = (run / "progress.log").read_text().splitlines()
    start = next(l for l in lines if "[gen] start" in l)
    end = next((l for l in reversed(lines) if "[gen] end" in l), lines[-1])
    threads = int(re.search(r"(\d+) threads", start)[1])
    t0, t1 = _secs(start), _secs(end)
    span = (t1 - t0) % 86400
    evs = [load_event(p) for p in event_dirs(run)]
    fam = defaultdict(list)
    for e in evs:
        fam[f"{e.context['family']}/t{e.context['tier']}"].append(e.outcome["cost"]["event"])
    return {"job": run.name, "threads": threads, "span_s": span, "events": len(evs),
            "events_per_min": 60 * len(evs) / max(span, 1), "cpu_s_per_event": {k: sum(v) / len(v) for k, v in fam.items()},
            "families": sorted(fam)}


def project(jobs, target, ensemble_share, members, threads):
    runs = target * (1 - ensemble_share) + target * ensemble_share * members
    rows = []
    for j in jobs:
        rate = j["events_per_min"] / j["threads"] * threads
        rows.append({**j, "projected_threads": threads, "projected_events_per_min": rate,
                     "days_for_target_runs": runs / rate / 60 / 24})
    return {"target_configurations": target, "ensemble_share": ensemble_share, "members": members, "oracle_runs": runs, "jobs": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--target", type=int, default=30000)
    ap.add_argument("--ensemble-share", type=float, default=0.15)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    r = project([job_throughput(p) for p in a.runs], a.target, a.ensemble_share, a.members, a.threads)
    print(f"oracle runs for {a.target} configurations ({a.ensemble_share:.0%} with {a.members}-member ensembles): {r['oracle_runs']:.0f}")
    print("| Job | Threads | Events | Span s | Events/min | CPU s/event | Days at", a.threads, "threads |")
    print("|---|---|---|---|---|---|---|")
    for j in r["jobs"]:
        cpu = ", ".join(f"{k} {v:.1f}" for k, v in j["cpu_s_per_event"].items())
        print(f"| {j['job']} | {j['threads']} | {j['events']} | {j['span_s']} | {j['events_per_min']:.1f} | {cpu} | {j['days_for_target_runs']:.1f} |")
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
