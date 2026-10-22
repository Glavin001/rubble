"""Relabel a DAgger round's jobs in chunks of events, keeping only the calls: P0's binary also
writes each job's start/end island states (`<i>_start/_end.msgpack`, ~6× the calls), which
training does not read; they are deleted after each chunk, and so are the calls the energy
admission rejects, so the round's disk use stays at its admitted calls (DAgger output counts
toward the shared data budget).

    python ../scripts/relabel_chunked.py --round DIR --bin RELABEL [--full-state-columns M] [--events-per-chunk 10]
    python ../scripts/relabel_chunked.py --round DIR --bin X --prune-only
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from learned_impact.outcome import schema as S  # noqa: E402
from learned_impact.outcome.dagger import admissible, relabel  # noqa: E402


def prune(out: Path) -> dict:
    """Delete the relabelled calls the energy admission rejects (training never reads them);
    their counts stay in <chunk>/summary.json."""
    n = {"relabelled": 0, "admitted": 0}
    for d in sorted(out.glob("*_call")):
        e = S.load_event(d, in_memory=True)
        ok = all(admissible(c) for c in e.calls if c.routed)
        n["relabelled"] += 1
        n["admitted"] += ok
        if not ok:
            shutil.rmtree(d)
    return n

ap = argparse.ArgumentParser()
ap.add_argument("--round", required=True)
ap.add_argument("--bin", required=True)
ap.add_argument("--full-state-columns", default=None)
ap.add_argument("--events-per-chunk", type=int, default=10)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--prune-only", action="store_true",
                help="only prune finished chunks that have no summary yet (an older run's)")
a = ap.parse_args()
rd = Path(a.round)
if a.prune_only:
    for cd in sorted((rd / "chunks").iterdir()):
        if (cd / "done").exists() and not (cd / "summary.json").exists():
            for f in (cd / "calls").glob("*.msgpack"):
                f.unlink()
            n = prune(cd / "calls")
            f = cd / "calls" / "failures.jsonl"
            (cd / "summary.json").write_text(json.dumps({**n, "rejected_jobs": sum(1 for _ in open(f)) if f.exists() else 0}))
            print(cd.name, n, flush=True)
    sys.exit(0)
jobs = json.loads((rd / "jobs.json").read_text())
events = list(dict.fromkeys(j["event"] for j in jobs))
for k in range(0, len(events), a.events_per_chunk):
    part = set(events[k:k + a.events_per_chunk])
    cd = rd / "chunks" / f"{k // a.events_per_chunk:03d}"
    if (cd / "done").exists():
        continue
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "jobs.json").write_text(json.dumps([j for j in jobs if j["event"] in part], indent=1))
    out = relabel(cd, Path(a.bin), a.threads, None, a.full_state_columns)
    for f in out.glob("*.msgpack"):
        f.unlink()
    n = prune(out)
    fails = sum(1 for _ in open(out / "failures.jsonl")) if (out / "failures.jsonl").exists() else 0
    (cd / "summary.json").write_text(json.dumps({**n, "rejected_jobs": fails}))
    (cd / "done").write_text("")
    print(f"chunk {cd.name}: {len(part)} events, {n['admitted']} of {n['relabelled']} calls admitted, "
          f"{fails} jobs rejected", flush=True)
print("RELABEL_DONE", flush=True)
